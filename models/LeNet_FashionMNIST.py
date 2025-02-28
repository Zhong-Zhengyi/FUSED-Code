# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel

class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mid_val = F.relu(self.fc2(x))
        output = self.fc3(mid_val)
        return output