#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Pytorch_learn 
@File    ：Myattention.py
@IDE     ：PyCharm 
@Author  ：咋
@Date    ：2023/7/14 17:42
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d

class Channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(Channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.max_pool = nn.AdaptiveMaxPool2d(1).cuda()

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False).cuda()

        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class My_attention(nn.Module):
    def __init__(self, channel_x, channel_y, ratio=8, kernel_size=7):
        super(My_attention, self).__init__()
        self.channel_attention = Channel_attention(channel_x)  # 假设channel_x是x的通道数
        self.spatial_attention = Spatial_attention()
        # 用于降维到y的通道数
        self.conv_reduce = nn.Conv2d(channel_x + channel_y, channel_x, 1, bias=False).cuda()

    def forward(self, x, y):
        # x和y可能有不同的通道数，先将它们拼接
        # print('x',x.size())
        # print('y',y.size())
        input = torch.cat([x, y], dim=1)
        # print(f'After concatenation: {input.size()}')  # 打印拼接后的尺寸

        # 将拼接后的特征图降维到y的通道数
        input = self.conv_reduce(input)
        #print(f'After 1x1 conv reduction: {input.size()}')  # 打印降维后的尺寸
        
        # 应用空间注意力和通道注意力
        SA_x = self.spatial_attention(input) * input
        CA_x = self.channel_attention(input) * input
        # print(f'After spatial attention: {SA_x.size()}')  # 打印空间注意力处理后的尺寸
        # print(f'After channel attention: {CA_x.size()}')  # 打印通道注意力处理后的尺寸
        
        # 将注意力加权的特征图与降维后的输入特征图相加，形成残差连接
        out = SA_x + CA_x + input
        #print(f'Final output: {out.size()}')  # 打印最终输出的尺寸
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        # print(f"in_planes: {in_planes}, ratio: {ratio}, in_planes // ratio: {in_planes // ratio}")
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.max_pool = nn.AdaptiveMaxPool2d(1).cuda()

        
        mid_channels = max(in_planes // ratio, 2)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, mid_channels, 1, bias=False).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Conv2d(mid_channels, in_planes, 1, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MyAttention(nn.Module):
    def __init__(self, channel_x, channel_y, ratio=8, kernel_size=7):
        super(MyAttention, self).__init__()
        self.channel_attention = ChannelAttention(channel_x)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        
        # 卷积层降维并添加BatchNorm和ReLU
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channel_x + channel_y, channel_x, 1, bias=False),
            nn.BatchNorm2d(channel_x),
            nn.ReLU()
        ).cuda()

    def forward(self, x, y):
        # 拼接x和y
        combined = torch.cat([x, y], dim=1)

        # 降维处理
        reduced = self.conv_reduce(combined)
        
        # 应用空间注意力和通道注意力
        sa_output = self.spatial_attention(reduced) * reduced
        ca_output = self.channel_attention(reduced) * reduced
        
        # 残差连接
        out = ca_output + sa_output + reduced
        return out
