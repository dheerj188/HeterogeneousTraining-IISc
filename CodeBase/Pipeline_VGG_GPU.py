
import torch 

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision.models import vgg19

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torchgpipe import GPipe

import matplotlib.pyplot as plt 

import time

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

class PipelineGPUTrainer:

    def __init__(self, model, dataloader, balance):

        self.balance = balance

        self.model = GPipe(model, self.balance, chunks = 4)

        self.dataloader = dataloader

        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) 

        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):

        #total_time_start = time.time()

        times_per_epoch = []

        losses = []

        for epoch in range(epochs):

            es_time = time.time()

            for id, (images, labels) in enumerate(self.dataloader):

                images = images.to(self.model.devices[0])

                labels = labels.to(self.model.devices[-1])

                outs = self.model(images)

                loss = self.criterion(outs, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            ee_time = time.time()

            times_per_epoch.append((ee_time-es_time)*10**3)

            losses.append(loss.item())

            print(f"Epoch: {epoch} || Loss: {loss.item()}")

        #total_time = (total_time_end-total_time_start)*10**3

        return (times_per_epoch,losses)
    
def main():

    model, dataset = load_CIFAR_model_dataset()

    batch_size = 512

    train_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle = True)

    trainer = PipelineGPUTrainer(model, train_loader, [23,23])

    epochs = 10

    x_axis = [i for i in range(epochs)]

    times,losses = trainer.train(epochs)

    print(f"Total execution time for {epochs} epochs is: {sum(times)}")

    print(f"Average Epoch time is: {sum(times)/epochs}")

    plt.plot(x_axis, losses)

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.title(f"VGG Net Loss for Pipeline GPU executions")

    plt.savefig("Pipeline_GPU_plot.pdf")

if __name__ == "__main__":

    main()

