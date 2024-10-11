import torch 

import torch.nn as nn

import torch.optim as optim

import torch.distributed as dist

import torchvision

from torchvision.models import vgg19

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

#from comm_enabled_gpipe import GPipe

from torchgpipe import GPipe

import os

from collections import OrderedDict

import time 

from torch.distributed.distributed_c10d import (
   
    _get_default_group,
    _rank_not_in_group,
    ReduceOp,

)

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

def load_CIFAR_model_test_dataset():

    test_set = torchvision.datasets.CIFAR10('./data', train=False,
				transform=transforms.Compose([
				transforms.Resize((256,256)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]))
    
    return test_set

class list_to_module(nn.Module):

    def __init__(self, module_list):

        super(list_to_module, self).__init__()

        self.module_list = module_list

    def forward(self, x):

        for module in self.module_list:

            x = module(x)

        return x
    
class PipelineCPUGPUTrainer:

    def __init__(self, model: nn.Sequential, train_dataloader, test_dataloader, rank, world_size, balance: list, process_group = None):

        self.train_dataloader = train_dataloader 

        self.test_dataloader = test_dataloader 

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

            self.model = GPipe(model, balance = self.balance, chunks = 4)

            self.gpu_devices = self.model.devices

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

    def _synchronize(self):

        for index, param in enumerate(self.model.parameters()):

            if param.requires_grad and param.grad is not None:

                grad = param.grad

                if(self.rank == 1):

                    device = grad.get_device()

                    grad = grad.to("cpu")

                dist.all_reduce(grad, group = self.process_group)

                if(self.rank == 1):

                    grad = grad.to(f"cuda:{device}")

                    param.grad.copy_(grad)

        return None 
    
    def train(self, epochs):

        self.model.train()

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

                self._synchronize()

                self.optimizer.step()

            print(f"Process: {self.rank} || Epoch Completed: {epoch} || Loss: {loss.item()}")
    
        return None
    
    def validate(self):

        self.model.eval()

        test_loss, correct = 0,0

        total = 0

        with torch.no_grad():

            for images, labels in self.test_dataloader:

                if self.rank == 1: 

                    images, labels = images.to(self.gpu_devices[0]), labels.to(self.gpu_devices[-1])

                out = self.model(images)

                test_loss += nn.CrossEntropyLoss()(out,labels).item()

                correct += (out.argmax(1) == labels).type(torch.float).sum().item()

                total += 1

        print("Validation Results")
                
        print(f"Process ID = {self.rank} || Test Error .Accuracy: {((correct/total)*100)}% || Avg loss: {test_loss:>8f} \n")

        return None 

def main():

    model, train_dataset = load_CIFAR_model_dataset()

    test_dataset = load_CIFAR_model_test_dataset()

    rank = int(os.environ["RANK"])

    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend = "gloo", rank = rank, world_size = world_size)

    batch_size = 512

    d_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, sampler = d_sampler)

    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    trainer = PipelineCPUGPUTrainer(model, train_loader, test_loader, rank, world_size, balance = [23, 23])

    epochs = 20

    trainer.train(epochs)

    trainer.validate()

if __name__ == "__main__":

    main()