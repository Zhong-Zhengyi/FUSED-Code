#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import torchvision
from peft import LoraConfig, get_peft_model
import timm

# from utils import *
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @staticmethod
    def split_weight_name(name):
        if 'weight' or 'bias' in name:
            return ''.join(name.split('.')[:-1])
        return name

    def save_params(self):
        for param_name, param in self.named_parameters():
            if 'alpha' in param_name or 'beta' in param_name:
                continue
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name, param.data.clone())

    def compute_diff(self):
        diff_mean = dict()
        for param_name, param in self.named_parameters():
            layer_name = self.split_weight_name(param_name)
            _buff_param_name = param_name.replace('.', '__')
            old_param = getattr(self, _buff_param_name, default=0.0)
            diff = (param - old_param) ** 2
            diff = diff.sum()
            total_num = reduce(lambda x, y: x*y, param.shape)
            diff /= total_num
            diff_mean[layer_name] = diff
        return diff_mean

    def remove_grad(self, name=''):
        for param_name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False


class Lora(nn.Module):
    def __init__(self, args, global_model):
        super(Lora, self).__init__()
        if args.data_name == 'cifar10':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]

#             target_modules = [
#     "to_patch_embedding.1",  # nn.Linear in to_patch_embedding
#     "transformer.layers.0.0.to_qkv",  # Attention layer's to_qkv in the first transformer block
#     "transformer.layers.0.0.to_out",  # Attention layer's to_out in the first transformer block
#     "transformer.layers.0.1.net.1",  # First Linear layer in FeedForward in the first transformer block
#     "transformer.layers.0.1.net.3",  # Second Linear layer in FeedForward in the first transformer block
#     "transformer.layers.1.0.to_qkv",  # Attention layer's to_qkv in the second transformer block
#     "transformer.layers.1.0.to_out",  # Attention layer's to_out in the second transformer block
#     "transformer.layers.1.1.net.1",  # First Linear layer in FeedForward in the second transformer block
#     "transformer.layers.1.1.net.3",  # Second Linear layer in FeedForward in the second transformer block
#     # Add more layers as needed for deeper transformer blocks
#     "linear_head.1"  # nn.Linear in linear_head
# ]

        elif args.data_name == 'cifar100':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
           
        elif args.data_name == 'fashionmnist':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))

            target_modules = ['conv1', 'fc3']
        elif args.data_name == 'adult':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            target_modules = ['fc3']
        elif args.data_name == 'text':
            global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(args.data_name)))
            target_modules = ["encoder.layer.11.output.dense"]

        config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        target_modules = target_modules,
        lora_dropout = 0.1,
        bias = "none",
        )
        self.lora_model = get_peft_model(global_model, config)
        for name, param in self.lora_model.named_parameters():
            if not any(target in name for target in config.target_modules):
                param.requires_grad = False

    def forward(self, x):
        return self.lora_model(x)