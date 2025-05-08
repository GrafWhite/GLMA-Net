import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import pdb
import matplotlib.pyplot as plt
import torchvision.models as models
import random
from lib.models.resnet import resnet50




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values 

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False) 
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups )

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W,self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()


        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

#end of attn definition

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)

        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#end of block definition

resnet50_model = resnet50(pretrained=False)  # 如果需要预训练的权重

class AxialFeatureFusionModule(nn.Module):
    def __init__(self, main_channels, aux_channels, num_heads, kernel_size=56):
        super(AxialFeatureFusionModule, self).__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels
        self.num_heads = num_heads
        self.combined_channels = main_channels + aux_channels

        # 使用轴向注意力层，沿水平和垂直轴处理
        self.axial_attention_h = AxialAttention(
            in_planes=self.combined_channels, 
            out_planes=self.combined_channels, 
            groups=num_heads, 
            kernel_size=kernel_size, 
            width=True  # 水平方向
        )

        self.axial_attention_w = AxialAttention(
            in_planes=self.combined_channels, 
            out_planes=self.combined_channels, 
            groups=num_heads, 
            kernel_size=kernel_size, 
            width=False  # 垂直方向
        )

        self.conv1x1 = nn.Conv2d(self.combined_channels, main_channels, kernel_size=1)

    def forward(self, feature_main, feature_aux):
        # 组合特征图
        combined_feature = torch.cat((feature_main, feature_aux), dim=1)

        # 应用轴向注意力（水平和垂直）
        attention_out = self.axial_attention_h(combined_feature)
        attention_out = self.axial_attention_w(attention_out)

        # 将通道数还原到 feature_main 的通道数
        output = self.conv1x1(attention_out)
        return output


class AxialFeatureFusionModule(nn.Module):
    def __init__(self, main_channels, aux_channels, num_heads, kernel_size=56):
        super(AxialFeatureFusionModule, self).__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels
        self.num_heads = num_heads
        self.combined_channels = main_channels + aux_channels

        # 使用轴向注意力层，沿水平和垂直轴处理
        self.axial_attention_h = AxialAttention(
            in_planes=self.combined_channels, 
            out_planes=self.combined_channels, 
            groups=num_heads, 
            kernel_size=kernel_size, 
            width=True  # 水平方向
        )

        self.axial_attention_w = AxialAttention(
            in_planes=self.combined_channels, 
            out_planes=self.combined_channels, 
            groups=num_heads, 
            kernel_size=kernel_size, 
            width=False  # 垂直方向
        )

        self.conv1x1 = nn.Conv2d(self.combined_channels, main_channels, kernel_size=1)

    def forward(self, feature_main, feature_aux):
        # 组合特征图
        combined_feature = torch.cat((feature_main, feature_aux), dim=1)

        # 应用轴向注意力（水平和垂直）
        attention_out = self.axial_attention_h(combined_feature)
        attention_out = self.axial_attention_w(attention_out)

        # 将通道数还原到 feature_main 的通道数
        output = self.conv1x1(attention_out)
        return output

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
    
    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.split_into_heads(self.W_q(x), batch_size)
        key = self.split_into_heads(self.W_k(x), batch_size)
        value = self.split_into_heads(self.W_v(x), batch_size)

        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        out = self.fc_out(out)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, main_channels, aux_channels, num_heads):
        super(FeatureFusionModule, self).__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels
        self.num_heads = num_heads
        self.combined_channels = main_channels + aux_channels

        self.attention = MultiHeadSelfAttention(self.combined_channels, num_heads)
        self.conv1x1 = nn.Conv2d(self.combined_channels, main_channels, kernel_size=1)

    def forward(self, feature_main, feature_aux):
        # 组合特征图
        combined_feature = torch.cat((feature_main, feature_aux), dim=1)

        # 调整形状以适应多头自注意力模块
        batch_size, channels, height, width = combined_feature.size()
        combined_feature = combined_feature.view(batch_size, channels, -1).permute(0, 2, 1)

        # 应用多头自注意力
        attention_out = self.attention(combined_feature)

        # 恢复原始形状
        attention_out = attention_out.permute(0, 2, 1).view(batch_size, -1, height, width)

        # 将通道数还原到 feature_main 的通道数
        output = self.conv1x1(attention_out)
        return output


    
class AxialResUNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128, imgchan = 3):
        super(AxialResUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024 *2*s), int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024 *2*s), int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s), int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(int(128*s), num_classes, kernel_size=1, stride=1, padding=0)

        self.resnet_branch = resnet50(pretrained=False)  # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])
        
        # 根据每个解码器层的输入通道数设置 d_model
#         self.attention_fusion_2 = MultiHeadAttentionFusion(d_model=int(1024 * 2 * 0.125), num_heads=8)
#         self.attention_fusion_3 = MultiHeadAttentionFusion(d_model=int(1024 * 0.125), num_heads=8)
#         self.attention_fusion_4 = MultiHeadAttentionFusion(d_model=int(512 * 0.125), num_heads=8)
#         self.attention_fusion_5 = MultiHeadAttentionFusion(d_model=int(256 * 0.125), num_heads=8)
#         self.fusion_module1 = FeatureFusionModule(main_channels=256, aux_channels=2048, num_heads=4)
#         self.fusion_module2 = FeatureFusionModule(main_channels=128, aux_channels=1024, num_heads=4)
#         self.fusion_module3 = FeatureFusionModule(main_channels=64, aux_channels=512, num_heads=4)
#         self.fusion_module4 = FeatureFusionModule(main_channels=32, aux_channels=256, num_heads=4)
        self.fusion_module1 = AxialFeatureFusionModule(main_channels=256, aux_channels=2048, num_heads=16, kernel_size=16)
        self.fusion_module2 = AxialFeatureFusionModule(main_channels=128, aux_channels=1024, num_heads=16, kernel_size=32)
        self.fusion_module3 = AxialFeatureFusionModule(main_channels=64, aux_channels=512, num_heads=16, kernel_size=64)
        self.fusion_module4 = AxialFeatureFusionModule(main_channels=32, aux_channels=256, num_heads=16, kernel_size=128)
#         self.attention_fusion_2 = fusion_module1(feature_main1, feature_aux1)
#         self.attention_fusion_3 = fusion_module2(feature_main2, feature_aux2)
#         self.attention_fusion_4 = fusion_module3(feature_main3, feature_aux3)
#         self.attention_fusion_5 = fusion_module4(feature_main4, feature_aux4)
        self.upsample = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2)
  
    def get_resnet_feature(self, x, size):
#         print(f"Requested size: {size}")
        found = False
        layer_index = 0

        x = self.resnet_branch[0](x)  # conv1
        x = self.resnet_branch[1](x)  # bn1
        x = self.resnet_branch[2](x)  # relu
        x = self.resnet_branch[3](x)  # maxpool

        for layer in range(4, 8):  # layer1 to layer4
            x = self.resnet_branch[layer](x)
            layer_index = layer
            if x.size(2) == size or x.size(3) == size:
                found = True
                break

        if found:
#             print(f"Matching size found in ResNet layer {layer_index}: {x.size()}")
            return x
        else:
#             print("Matching size not found in any ResNet layer.")
            raise ValueError("Requested size not found in ResNet features")


    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 克隆 x 以创建一个未经修改的副本，用于 ResNet 分支
        x_cnn = self.upsample(x.clone())
#         x_cnn = F.interpolate(x.clone(), scale_factor=2, mode='bilinear', align_corners=False)

        # 主路径前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 初始特征图来自最深层的解码器输出
        current_feature = x4

        # 解码器层列表
        decoders = [self.decoder2, self.decoder3, self.decoder4, self.decoder5]
        attention_fusion_modules = [self.fusion_module1, self.fusion_module2, self.fusion_module3, self.fusion_module4]

        for idx, decoder in enumerate(decoders):
            resnet_feature = self.get_resnet_feature(x_cnn, current_feature.size()[2])

            # 注意力融合并打印融合后的尺寸和通道数
            fused_feature = attention_fusion_modules[idx](current_feature, resnet_feature)
#             print(f"After fusion {idx+2}, shape: {fused_feature.shape}")

            current_feature = decoder(fused_feature)
#             print(f"After decoder {idx+2}, shape: {current_feature.shape}")

            if idx < len(decoders) :
                current_feature = F.interpolate(current_feature, scale_factor=2, mode='bilinear', align_corners=False)
#                 print(f"After upsampling {idx+2}, shape: {current_feature.shape}")

        # 调整最终输出尺寸以匹配类别数量
        x_final = self.adjust(current_feature)
        return x_final


    def forward(self, x):
        return self._forward_impl(x)



def ARUNet(pretrained=False, **kwargs):
    model = AxialResUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

