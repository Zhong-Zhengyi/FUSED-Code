# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision
import timm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out


# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
#         self.fc = nn.Linear(512, config.num_classes)
#
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         mid_val = out.view(out.size(0), -1)
#         out = self.fc(mid_val)
#         return out, mid_val

class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)


    def forward(self, x):
        return self.model(x)

# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.num_classes = config.num_classes
#         self.model = timm.create_model('vit_small_patch16_224', pretrained=True)
#         # self.model = ViTForImageClassification.from_pretrained('google/vit-small-patch16-224-in21k', num_labels=self.num_classes)
        
#         # processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
#         # self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
#         self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

#     def forward(self, x):
#         return self.model(x)
