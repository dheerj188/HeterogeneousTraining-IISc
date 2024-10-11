import argparse
import os
from contextlib import nullcontext
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.model_zoo import model_builder
from commons.performance_evaluator import get_profile_context
from commons.utils import get_data, get_tflops, get_time_stamp, DataManager
from packaging import version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam


import torch.profiler 

CAI_VERSION = colossalai.__version__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distplan",
        type=str,
        default="CAI_Gemini",
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=100,
        help="training iterations for test",
    )

    args = parser.parse_args()
    return args


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=""):
    return f"{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB"


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f"{numel / GB_SIZE:.1f}B"
    elif numel >= MB_SIZE:
        return f"{numel / MB_SIZE:.1f}M"
    elif numel >= KB_SIZE:
        return f"{numel / KB_SIZE:.1f}K"
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split("\n")[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = True  # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch()

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion
    criterion = GPTLMLoss()
    torch.manual_seed(123)
    if args.distplan.startswith("CAI"):
        ctx = (
            LazyInitContext(default_device=get_accelerator().get_current_device())
            if args.distplan == "CAI_Gemini"
            else nullcontext()
        )
        # build GPT model
        with ctx:

            model = model_builder(args.model_type)(checkpoint=True)
    
        # ae gradient of the loss with respect to the activations and parameters of these layers is indicated by the nodes marked with b. During the backward pass, all these nodes are evaluated in the reversed order. The results obtained for the f nodes are needed to compute the b nodes, and hence all f nodes are kept in memory after the forward pass. Only when backpropagation has progressed far enough to have computed all dependencies, or children, of an f node, can it be erased from memory. This means that the memory required by simple backprop grows linearly with the number of neural net layers n. Below we show the order in which these nodes are computed. The purple shaded circles indicate which of the nodes need to be held in memory at any given time.ssign running configurations
        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        plugin = None
        if args.distplan.startswith("CAI_ZeRO"):
            plugin = LowLevelZeroPlugin(
                stage=zero_stage, reduce_bucket_size_in_m=12, overlap_communication=True, verbose=True
            )
        elif args.distplan == "CAI_Gemini":
            plugin = GeminiPlugin(search_range_m=128, hidden_dim=model.config.n_embd, offload_optim_frac=1.0, offload_param_frac=1.0,placement_policy="static")
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        logger.info(get_mem_info(prefix="After init optim, "), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        plugin = TorchDDPPlugin()
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer

            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)

    else:
        raise RuntimeError
    # wrap your model and optimizer
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    # model is shared after TP
    numel = get_model_size(model)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix="After init model, "), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()
    model.train()
    tflops_list = []

    def train_step(batch = None):
        # we just use randomly generated data here

        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        #input_ids, attn_mask = (batch['input_ids'], batch['attention_mask'])
        
        optimizer.zero_grad()

        start = time()

        with torch.profiler.record_function("forward pass"):
            outputs = model(input_ids, attn_mask)

        with torch.profiler.record_function("Executions post FWD and pre BWD"):
            loss = criterion(outputs, input_ids)
            torch.cuda.synchronize()
            fwd_end = time()
            fwd_time = fwd_end - start

        logger.info(get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Forward "), ranks=[0])

        with torch.profiler.record_function("backward pass"):
            booster.backward(loss, optimizer)

        with torch.profiler.record_function("Executions post BWD pre PU"):
            torch.cuda.synchronize()
            bwd_end = time()
            bwd_time = bwd_end - fwd_end
        
        logger.info(get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Backward "), ranks=[0])

        with torch.profiler.record_function("Parameter Update"):
            optimizer.step()
        
        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Optimizer step "), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)
    
    """
    demo_profiler = get_profile_context(
        PROF_FLAG, WARMUP_STEPS, NUM_STEPS - WARMUP_STEPS, save_dir=f"profile/{get_time_stamp()}-demo"
    )"""

    profile_rank = int(os.environ['LOCAL_RANK'])

    manager = DataManager()
    
    train_loader = manager.load_data(SEQ_LEN)
    
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log5bauto/gemgptx'),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
        ) as prof:
        """
        for n in range(NUM_STEPS):
            (_, batch) = next(enumerate(train_loader), None)
            if batch is None:
                break

            train_step(batch)
            prof.step()

        """
        for n in range(NUM_STEPS):
            train_step()
            prof.step()
            
    prof.export_chrome_trace(f"./logchrome5bauto/cai_gem_ha_trace-rank{profile_rank}.json")
    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()

if __name__ == "__main__":
    
    main()