import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN_CNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN_CNN,self).__init__()

        #convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        #fully connected linear layers
        self.fc4 = nn.Linear(7 * 7 * 64, 512)#find smarter way to find in_channels instead of 7 * 7 * 64
        self.fc5 = nn.Linear(512, num_actions)#output layer

    def forward(self, x):
        #convolutional pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #linear pass and reshaping with view
        x = F.relu(self.fc4(x.view(x.size[0], -1)))
        return self.fc5(x)#no activation for output layer. Maybe consider softmax?


#note: Gym has Discrete or Box inputs from the environment. CNN's should be used with Box ??? observation types only ???
