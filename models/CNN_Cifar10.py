# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision


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
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
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
#         return out

class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        return self.model(x)
    


# import torch
# from torch import nn

# from einops import rearrange
# from einops.layers.torch import Rearrange

# # helpers

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
#     _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

#     y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
#     assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
#     omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
#     omega = 1. / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
#     return pe.type(dtype)

# # classes

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)

#     def forward(self, x):
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head),
#                 FeedForward(dim, mlp_dim)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

# class Model(MyModel):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
#         super(MyModel, self).__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

#         self.to_latent = nn.Identity()
#         self.linear_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, img):
#         *_, h, w, dtype = *img.shape, img.dtype

#         x = self.to_patch_embedding(img)
#         pe = posemb_sincos_2d(x)
#         x = rearrange(x, 'b ... d -> b (...) d') + pe

#         x = self.transformer(x)
#         x = x.mean(dim = 1)

#         x = self.to_latent(x)
#         return self.linear_head(x)
