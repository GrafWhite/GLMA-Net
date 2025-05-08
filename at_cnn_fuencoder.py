import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc1(self.avg_pool(x))
#         max_out = self.fc1(self.max_pool(x))
#         out = self.relu(avg_out + max_out)
#         return self.sigmoid(self.fc2(out))

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 分别处理两个分支的卷积层
        self.fc1_avg = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.fc1_max = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        # 分别处理两个分支的卷积层
        self.fc2_avg = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.fc2_max = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.fc2_max_avg = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1_avg(self.avg_pool(x))  # 平均池化分支的卷积
        max_out = self.fc1_max(self.max_pool(x))  # 最大池化分支的卷积
        out = self.relu(avg_out + max_out)
        avg_attention = self.sigmoid(self.fc2_avg(avg_out))  # 平均池化分支的注意力权重
        max_attention = self.sigmoid(self.fc2_max(max_out))  # 最大池化分支的注意力权重
        # 将两个分支的注意力权重结合起来
        # combined_attention = self.sigmoid(self.fc2_max_avg(out))
        combined_attention = avg_attention + max_attention
        return combined_attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionModuleCA(nn.Module):
    def __init__(self, channel_low, channel_high, out_channels):
        super(FusionModuleCA, self).__init__()
        # 初始化通道注意力和空间注意力模块
        self.channel_attention_high = ChannelAttention(channel_high)
        self.spatial_attention_low = SpatialAttention()
                
        # 降维到输出通道数的卷积层
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(channel_high + channel_low, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, low, high):
        # 分别应用通道和空间注意力
        # print("low",low.size())
        # print("high",high.size())
        high_att = self.channel_attention_high(high) * high
        # print("high_att",high_att.size())
        low_att = self.spatial_attention_low(low) * low
        # print("low_att",low_att.size())

        # 合并加权后的特征
        fused = torch.cat([high_att, low_att], dim=1)
        # print("fused",fused.size())
        
        # 降维处理
        output = self.reduce_dim(fused)

        return output

