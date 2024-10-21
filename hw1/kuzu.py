"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.flatten = nn.Flatten()
        # INSERT CODE HERE
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x 
        # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28*28, 510)
        self.fc2 = nn.Linear(510, 250)
        self.fc3 = nn.Linear(250, 10)
        self.flatten = nn.Flatten()
        # INSERT CODE HERE

    def forward(self, x):
        x = self.flatten(x)  
        x = self.fc1(x)
        x = F.tanh(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(x)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()
        
        # INSERT CODE HERE

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x # CHANGE CODE HERE
