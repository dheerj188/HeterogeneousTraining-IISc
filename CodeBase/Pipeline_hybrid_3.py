import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import torch.optim as optim

import torch.distributed as dist

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

import functools

#from comm_enabled_gpipe import GPipe

from torchgpipe import GPipe

import os

from collections import OrderedDict

from torch.distributed.distributed_c10d import (
    _get_default_group,
    _rank_not_in_group,
    ReduceOp,
)

class list_to_module(nn.Module):

    def __init__(self, module_list):

        super(list_to_module, self).__init__()

        self.module_list = module_list

    def forward(self, x):

        for module in self.module_list:

            x = module(x)

        return x
    
class PipelineCPUGPUTrainer:

    def __init__(self, model: nn.Sequential, dataloader, rank, world_size, balance: list, process_group = None):

        self.dataloader = dataloader 

        self.rank = rank 

        self.world_size = world_size

        self.balance = balance 

        self.cpu_device = None

        self.gpu_devices = None

        if process_group is None:

            self.process_group = _get_default_group()

        if(self.rank == 0):

            j = 0

            partitions = nn.ModuleList([])

            layers = OrderedDict()

            for name, layer in model.named_children():
                
                layers[name] = layer

                if len(layers) == self.balance[j]:

                    # Group buffered layers as a partition.

                    partition = nn.Sequential(layers)

                    partitions.append(partition)

                    # Prepare for the next partition.

                    layers.clear()

                    j += 1
            
            self.model = list_to_module(partitions).to("cpu")

        if(self.rank == 1):

            self.model = GPipe(model, balance = self.balance, chunks = 1)

            self.gpu_devices = self.model.devices

        self._accum_grad_hooks = []

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

        self._attach_all_reduce_hook()

    def _attach_all_reduce_hook(self):

        import torch.distributed._functional_collectives as fcol

        def all_reduce_hook_on_cpu(param, *, param_index:int):

            if param.grad is None:

                return 
            
            else:

                gradient = param.grad / self.world_size

                grad_device = gradient.get_device()

                if(self.rank == 1):

                    gradient = gradient.to("cpu")

                else:

                    pass

                gradient = fcol.all_reduce(gradient, "sum", self.process_group)

                if(self.rank == 1):

                    gradient = gradient.to(f"cuda:{grad_device}")

                param.grad.copy_(gradient)


        for index, param in enumerate(self.model.parameters()):

            if not param.requires_grad:

                continue 

            self._accum_grad_hooks.append(param.register_post_accumulate_grad_hook(functools.partial(all_reduce_hook_on_cpu, param_index = index)))

    def train(self, epochs):

        for epoch in range(epochs):

            for id, (images, labels) in enumerate(self.dataloader):

                if(self.rank == 1):

                    images = images.reshape(-1, 28*28).to(self.gpu_devices[0])

                    labels = labels.to(self.gpu_devices[-1])

                else:

                    images = images.reshape(-1, 28*28).to("cpu")

                    labels = labels.to("cpu")

                outs = self.model(images)

                loss = self.criterion(outs, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            print(f"Process {self.rank} || Loss = {loss.item()}")

        return None
    
"""
def main():

    model = nn.Sequential(nn.Linear(784,100), nn.ReLU(), nn.Linear(100,100), nn.ReLU(), nn.Linear(100,10))

    rank = int(os.environ["RANK"])

    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend = "gloo", rank = rank, world_size = world_size)

    train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)

    batch_size = 64

    d_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = d_sampler)

    trainer = PipelineCPUGPUTrainer(model, train_loader, rank, world_size, balance = [3,2])

    epochs = 20

    trainer.train(epochs)

"""

def load_dataset():

    return 

def initialize_distributed_process_group():

    return 

def main(rank: int, world_size, total_epochs: int):

    return  

if __name__ == "__main__":

    pass

