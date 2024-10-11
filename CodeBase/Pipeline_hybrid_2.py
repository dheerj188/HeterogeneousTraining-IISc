import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import torch.optim as optim

import torch.distributed as dist

import torchvision

from torchvision.models import vgg19

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

# ResNet Libraries 

from typing import TYPE_CHECKING, Optional, Tuple, Union

from torch import Tensor

from torchgpipe.skip import Namespace, pop, skippable, stash

#------------------------------------------------ Test DNNs ---------------------------------------------------------

def model_prep(org_model, paras):

    org_model.classifier = nn.Sequential(nn.Flatten(), *list(org_model.classifier._modules.values()))
    
    model = nn.Sequential()
    
    for para in paras:
        
        if type(org_model._modules[para])==torch.nn.modules.container.Sequential:
            
            for layer in org_model._modules[para]:
                
                print(f"{para}: {layer}")
                
                model.add_module(str(len(model)), layer)
        
        else:
           
            model.add_module(str(len(model)), org_model._modules[para])

    num_layers = len(list(model.modules()))

    return (model, num_layers)

def load_CIFAR_model_dataset():

    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True,
				transform=transforms.Compose([
				transforms.Resize((256,256)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]))
    
    model = vgg19(num_classes = 10)

    model, _ = model_prep(model, [a for a in model._modules])

    return (model, train_set)
#------------------------------------------------ Helper Classes ----------------------------------------------------

class list_to_module(nn.Module):

    def __init__(self, module_list):

        super(list_to_module, self).__init__()

        self.module_list = module_list

    def forward(self, x):

        for module in self.module_list:

            x = module(x)

        return x
    
class PreProcessModel(nn.Module):

    def __init__(self, model):

        self.model = model 

    def _flatten_sequential(self):

        def _flatten(module):

            for name, child in self.model.named_children():

                if isinstance(child, nn.Sequential):

                    for sub_name, sub_child in _flatten(child):

                        yield (f'{name}_{sub_name}', sub_child)

                else:

                    yield (name, child)

        return nn.Sequential(OrderedDict(_flatten(self.model)))
    
    def forward(self, x):

        return self.model(x)
    
#------------------------------------------------- Trainer Class Implementation ----------------------------------------
       
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

            torch.set_num_threads(32)

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

                    images = images.to(self.gpu_devices[0])

                    labels = labels.to(self.gpu_devices[-1])

                else:
                    
                    images = images.to("cpu")

                    labels = labels.to("cpu")

                outs = self.model(images)

                loss = self.criterion(outs, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            print(f"Process {self.rank} || Loss = {loss.item()}")

        return None
    
#----------------------------------------- main() function called by both the processes ------------------------------------------
    
def main():

    #model = nn.Sequential(nn.Linear(784,100), nn.ReLU(), nn.Linear(100,100), nn.ReLU(), nn.Linear(100,10))

    model, dataset = load_CIFAR_model_dataset()

    num_layers = len(list(model.modules()))

    rank = int(os.environ["RANK"])

    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend = "gloo", rank = rank, world_size = world_size)

    #dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)

    batch_size = 64

    d_sampler = DistributedSampler(dataset)

    train_loader = DataLoader(dataset = dataset, batch_size=batch_size, sampler = d_sampler)

    balance_int = 10

    trainer = PipelineCPUGPUTrainer(model, train_loader, rank, world_size, balance = [23, 23])

    epochs = 200

    trainer.train(epochs)

if __name__ == "__main__":

    main()
