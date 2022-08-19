import numpy as np
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,(5,5))
        self.subsample2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(6,16,(5,5))
        self.subsample2 = nn.MaxPool2d((2,2))
        self.conv5 = nn.Conv2d(16,1,(5,5))
        self.fullyconnected6 = nn.Linear(120,84)
        self.fullyconnected7 = nn.Linear(84,10)

    def forward(self,x):
        output = self.conv1(x)
        output = nn.Tanh()(output)
        output = self.subsample2(output)
        output = self.conv2(output)
        output = nn.Tanh()(output)
        output = self.subsample2(output)
        output = self.conv5(output)
        output = nn.Tanh()(output)
        output = torch.flatten(output)
        output = self.fullyconnected6(output)
        output = nn.Tanh()(output)
        output = self.fullyconnected7(output)
        output = nn.Softmax()(output)
        return output
    
    def fit(self, data_loader, epoch, lr=None, optimizer=None, loss_function=None):
        if optimizer is None:
            if lr is None:
                print('Need to provide "lr" argument if optimizer aurgument is not provided')
            optimizer = torch.optim.SGD(self.parameters(),lr=lr)
        if loss_function is None:
            print('"loss" argument is missing')
        for i in range(epoch):
            output = self(data_loader)
            loss = loss_function(output, )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    model = LeNet()