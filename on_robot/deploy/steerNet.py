import numpy as np 
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


class SteerNet(nn.Module):
    def __init__(self, num_classes):
        super(SteerNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*18*18, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # to prevent overfitting
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    


def test():
    mynet = SteerNet(5)
    print(mynet)
    #mynet = tr.load("steerNet.pt")

if __name__ == "__main__":
    test()  
