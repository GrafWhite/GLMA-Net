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
from lib.models.resnet import resnet50, resnet34, resnet18 , resnet34_2
from lib.models.ca_sa import My_attention, MyAttention
# from lib.models.resnet import resnet50_1, resnet34_1, resnet18_1
# from lib.models.snake_resnet import resnet34
from lib.models.snake_resnet import resnet50_s
from lib.models.aftermodel import FusionModule,ReduceDimModule,ReduceDimModule_1,EdgeEnhancementModule
from lib.models.S3_DSConv_pro import DSConv_pro ,My_DSConv
from lib.models.at_cnn_fuencoder import FusionModuleCA

Reduceinput = ReduceDimModule()
Reduceinput1 = ReduceDimModule_1()
FusionIO = FusionModule()
FusionIO_1 = FusionModule().cuda('cuda:1')

EdgeEnhanceinput = EdgeEnhancementModule().cuda('cuda:1')
afterFusion = MyAttention(2,4).cuda('cuda:1')

class SnakeConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, extend_scope, if_offset, device):
        super(SnakeConvBlock, self).__init__()
        
        # 第一个蛇卷积单元
        self.conv20 = EncoderConv(input_channel, output_channel)
        self.conv2x = DSConv_pro(input_channel, output_channel, 9, extend_scope, 0, if_offset, device)
        self.conv2y = DSConv_pro(input_channel, output_channel, 9, extend_scope, 1, if_offset, device)
        # 第一个蛇卷积单元后的降维卷积
        self.conv_reduce1 = EncoderConv(3*output_channel, output_channel)
        
        # 第二个蛇卷积单元
        self.conv30 = EncoderConv(output_channel, output_channel)
        self.conv3x = DSConv_pro(output_channel, output_channel, 9, extend_scope, 0, if_offset, device)
        self.conv3y = DSConv_pro(output_channel, output_channel, 9, extend_scope, 1, if_offset, device)
        # 第二个蛇卷积单元后的降维卷积
        self.conv_reduce2 = EncoderConv(3*output_channel, output_channel)

        # 批标准化和ReLU激活函数
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # 通过第一个蛇卷积单元
        x_20 = self.conv20(x)
        x_2x = self.conv2x(x)
        x_2y = self.conv2y(x)
        # 合并特征图并降维
        x1 = torch.cat([x_20, x_2x, x_2y], dim=1)
        x1 = self.conv_reduce1(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        
        # 通过第二个蛇卷积单元
        x_30 = self.conv30(x1)
        x_3x = self.conv3x(x1)
        x_3y = self.conv3y(x1)
        # 合并特征图并降维
        x2 = torch.cat([x_30, x_3x, x_3y], dim=1)
        x2 = self.conv_reduce2(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        
        # 残差连接和ReLU激活
        x_out = self.relu(x2 + identity)
        
        return x_out


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownSampleBlock, self).__init__()
        # 使用普通卷积进行下采样
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSampleBlock, self).__init__()
        # 使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, norm_layer=nn.BatchNorm2d):
        super(DecoderLayer, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.upsample:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip=None):
        if self.upsample:
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=3, kernel_size=56,
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
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=bias)
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

        # print("q shape:", q.shape)
        # print("q_embedding shape:", q_embedding.shape)
        
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

class AxialAttention_dynamic_L(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic_L, self).__init__()
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

        # print("q shape:", q.shape)
        # print("all_embeddings:", all_embeddings.shape)
        
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


class AxialBlock_dynamic_L(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic_L, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic_L(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic_L(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
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



class AxialAttention_wof(nn.Module):
    def __init__(self, in_planes, out_planes, groups=4, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wof, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=bias)
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

        # print("q size:", q.size())  # 应显示 [N, W, C, H] 形状
        # print("q_embedding size:", q_embedding.size())  # 应显示 [C, kernel_size, kernel_size] 或类似形状``
        

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # Removed multiplication by dynamic factors

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # Removed multiplication by dynamic factors

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
        nn.init.kaiming_normal_(self.qkv_transform.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.relative, 0., math.sqrt(1. / (self.group_planes * 2)))


class AxialBlock_wof(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wof, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv_down and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wof(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wof(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
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

#end of block definition

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=4, kernel_size=56,
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

        out = self.conv_down(x)
#         print("After conv_down: ", out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.hight_block(out)
#         print("After hight_block: ", out.shape)
        out = self.width_block(out)
#         print("After width_block: ", out.shape)
        out = self.relu(out)
        out = self.conv_up(out)
#         print("After conv_up: ", out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
#             print("After downsample: ", identity.shape)

        out += identity
        out = self.relu(out)

        return out


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
        self.batch_norm = nn.BatchNorm2d(main_channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(main_channels, 4 * main_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4 * main_channels, main_channels, kernel_size=1)
        )

    def forward(self, feature_main, feature_aux):
        # 组合特征图
        combined_feature = torch.cat((feature_main, feature_aux), dim=1)
    
        # 应用轴向注意力（水平和垂直）
        attention_out = self.axial_attention_h(combined_feature)
        attention_out = self.axial_attention_w(attention_out)
    
        # 残差连接
        attention_out += combined_feature
    
        # 将通道数还原到 feature_main 的通道数
        reduced_feature = self.conv1x1(attention_out)

        # 应用BatchNorm
        norm_feature = self.batch_norm(reduced_feature)
    
        # 在ReLU之后应用FFN
        output = self.ffn(nn.ReLU()(norm_feature))

        output = self.conv1x1(attention_out)
    
        return output


    
    
class AxialResUNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128, imgchan = 1):
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
        

        self.fusion_module1 = AxialFeatureFusionModule(main_channels=256, aux_channels=2048, num_heads=8, kernel_size=16)
        self.fusion_module2 = AxialFeatureFusionModule(main_channels=128, aux_channels=1024, num_heads=16, kernel_size=32)
        self.fusion_module3 = AxialFeatureFusionModule(main_channels=64, aux_channels=512, num_heads=16, kernel_size=64)
        self.fusion_module4 = AxialFeatureFusionModule(main_channels=32, aux_channels=256, num_heads=24, kernel_size=128)

        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
  
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




class AxialResUNet_2(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=1):
        super(AxialResUNet_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        # Other initializations...
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Using DepthwiseSeparableConv for initial convolutions
        self.conv1 = DepthwiseSeparableConv(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.conv2 = DepthwiseSeparableConv(self.inplanes, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = DepthwiseSeparableConv(128, self.inplanes, kernel_size=3, stride=1, padding=1)

        # BatchNorm and ReLU remain unchanged
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

        # Initializations...
        self.decoder1 = DepthwiseSeparableConv(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = DepthwiseSeparableConv(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)  # Assuming skip connection from layer3 of the encoder
        self.decoder3 = DecoderLayer(int(1024 * s) + int(512 * s), int(512 * s))  # Assuming skip connection from layer2
        self.decoder4 = DecoderLayer(int(512 * s) + int(256 * s), int(256 * s))  # Assuming skip connection from layer1
        self.decoder5 = DecoderLayer(int(256 * s) + int(128 * s), int(128 * s))  # Assuming skip connection from initial conv layers
        # Assuming the adjust layer remains unchanged
        # Add any additional layers or modifications as needed
        # Final adjust layer remains unchanged
        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        # self.resnet_branch = resnet34(pretrained=False)  # 不使用预训练权重
        # self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.fusion_module1 = AxialFeatureFusionModule(main_channels=256, aux_channels=512, num_heads=24, kernel_size=16)  # Assuming this corresponds to the last stage of ResNet34
        # self.fusion_module2 = AxialFeatureFusionModule(main_channels=128, aux_channels=256, num_heads=24, kernel_size=32)  # And so on...
        # self.fusion_module3 = AxialFeatureFusionModule(main_channels=64, aux_channels=128, num_heads=24, kernel_size=64)
        # self.fusion_module4 = AxialFeatureFusionModule(main_channels=32, aux_channels=64, num_heads=48, kernel_size=128)

        self.upsample = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=2, stride=2)


#      def get_resnet_feature(self, x, size):
# #         print(f"Requested size: {size}")
#         found = False
#         layer_index = 0

#         x = self.resnet_branch[0](x)  # conv1
#         x = self.resnet_branch[1](x)  # bn1
#         x = self.resnet_branch[2](x)  # relu
#         x = self.resnet_branch[3](x)  # maxpool

#         for layer in range(4, 8):  # layer1 to layer4
#             x = self.resnet_branch[layer](x)
#             layer_index = layer
#             if x.size(2) == size or x.size(3) == size:
#                 found = True
#                 break

#         if found:
# #             print(f"Matching size found in ResNet layer {layer_index}: {x.size()}")
#             return x
#         else:
# #             print("Matching size not found in any ResNet layer.")
#             raise ValueError("Requested size not found in ResNet features")


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
        # Encoder
        # Assuming you have the encoder part implemented as before, up to layer4
        # Skip connections would typically come from the output of the encoder blocks
        # 克隆 x 以创建一个未经修改的副本，用于 ResNet 分支
        # x_cnn = self.upsample(x.clone())
        # x_cnn = F.interpolate(x.clone(), scale_factor=2, mode='bilinear', align_corners=False)
        # x_cnn = self.upsample(x)
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

        # Encoder
        
        # 解码器部分 - 使用跳跃连接
        d1 = self.decoder2(x4)  # 第一层解码器不需要跳跃连接输入
        d2 = self.decoder3(d1, x3)  # 将x3作为跳跃连接输入到decoder2
        d3 = self.decoder4(d2, x2)  # 将x2作为跳跃连接输入到decoder3
        d4 = self.decoder5(d3, x1)  # 将x1作为跳跃连接输入到decoder4
        
        x_final = self.adjust(d4)
        
        return x_final

    def forward(self, x):
        return self._forward_impl(x)


class AxialResUNet_3(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(AxialResUNet_3, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet34(pretrained=False)  # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))
        
        
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

        x_resnet = x.clone()
        
        # AxialAttention Encoder
        # pdb.set_trace()
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
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)



        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x4)  # 跳跃连接
        # print("x, x4 size:", x.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = torch.cat([x, resnet_feature1], dim=1)  # 拼接特征图
        # print("cat", x.size())
        x = self.conv_reduce1(x) 
        # print("conv_reduce1", x.size())
        
    
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x3)
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = torch.cat([x, resnet_feature2], dim=1)
        x = self.conv_reduce2(x)
        
    
        # 继续这个过程直到所有的解码层都处理完毕
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x2)
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = torch.cat([x, resnet_feature3], dim=1)
        x = self.conv_reduce3(x)
        
    
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        # print("resnet_feature4:", resnet_feature4.size())
        x = torch.cat([x, resnet_feature4], dim=1)
        x = self.conv_reduce4(x)
        

        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        x = self.adjust(F.relu(x))  # 最终调整   
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)

class AxialResUNet_4(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(AxialResUNet_4, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet34_2(pretrained=False)  # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = AxialFeatureFusionModule(main_channels=256, aux_channels=512, num_heads=24, kernel_size=16)
        self.fusion_module2 = AxialFeatureFusionModule(main_channels=128, aux_channels=256, num_heads=24, kernel_size=32)
        self.fusion_module3 = AxialFeatureFusionModule(main_channels=64, aux_channels=128, num_heads=24, kernel_size=64)
        self.fusion_module4 = AxialFeatureFusionModule(main_channels=32, aux_channels=64, num_heads=48, kernel_size=128)
        
        
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

        x_resnet = x.clone()
        
        # AxialAttention Encoder
        # pdb.set_trace()
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
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)



        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode='bilinear'))
        # x = torch.add(x, x4)  # 跳跃连接
        # print("x, x4 size:", x.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        # print("cat", x.size())
        x = self.fusion_module1(x,resnet_feature1) 
        # print("fusion_module1", x.size())
        
    
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x3)
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = self.fusion_module2(x,resnet_feature2) 
        
    
        # 继续这个过程直到所有的解码层都处理完毕
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x2)
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = self.fusion_module3(x,resnet_feature3) 
        
    
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        

        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        x = self.adjust(F.relu(x))  # 最终调整   
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)

class AAFSR_net(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(AAFSR_net, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet34(num_classes=1000)  # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = AxialFeatureFusionModule(main_channels=256, aux_channels=512, num_heads=24, kernel_size=16)
        self.fusion_module2 = AxialFeatureFusionModule(main_channels=128, aux_channels=256, num_heads=24, kernel_size=32)
        self.fusion_module3 = AxialFeatureFusionModule(main_channels=64, aux_channels=128, num_heads=24, kernel_size=64)
        self.fusion_module4 = AxialFeatureFusionModule(main_channels=32, aux_channels=64, num_heads=24, kernel_size=128)
        
        
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

        x_resnet = x.clone()
        
        # AxialAttention Encoder
        # pdb.set_trace()
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
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)



        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x4)  # 跳跃连接
        # print("x, x4 size:", x.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        # print("cat", x.size())
        x = self.fusion_module1(x,resnet_feature1) 
        # print("fusion_module1", x.size())
        
    
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x3)
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = self.fusion_module2(x,resnet_feature2) 
        
    
        # 继续这个过程直到所有的解码层都处理完毕
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x2)
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = self.fusion_module3(x,resnet_feature3) 
        
    
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        

        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        x = self.adjust(F.relu(x))  # 最终调整   
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)


class casanet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(casanet, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet34(pretrained=False)   # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = My_attention(channel_x=256, channel_y=512)
        self.fusion_module2 = My_attention(channel_x=128, channel_y=256)
        self.fusion_module3 = My_attention(channel_x=64, channel_y=128)
        self.fusion_module4 = My_attention(channel_x=32, channel_y=64)
        
        
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

        x_resnet = x.clone()
        
        # AxialAttention Encoder
        # pdb.set_trace()
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
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)

        # x = torch.add(x, x4)  # 跳跃连接
        #print("x4 size:", x4.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = self.fusion_module1(x4,resnet_feature1) 
        #print("fusion_module:", x.size())
        x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
        # print("cat", x.size())
        
        # print("fusion_module1", x.size())
        #print("x size:", x.size())
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        #print("resnet_feature2:", resnet_feature2.size())
        # x = torch.add(x, x3)  
        x = self.fusion_module2(x,resnet_feature2) 
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))

        #print("x size:", x.size())
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        #print("resnet_feature3:", resnet_feature3.size())
        # 继续这个过程直到所有的解码层都处理完毕
        x = self.fusion_module3(x,resnet_feature3)
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        # x = torch.add(x, x2)
        
        # print("xsize:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

        x = self.adjust(F.relu(x))  # 最终调整  
        
        # x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        # print("resnet_feature4:", resnet_feature4.size())
        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)


class casanet_2(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(casanet_2, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet50(pretrained=False)   # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = My_attention(channel_x=512, channel_y=2048)
        self.fusion_module2 = My_attention(channel_x=256, channel_y=1024)
        self.fusion_module3 = My_attention(channel_x=128, channel_y=512)
        self.fusion_module4 = My_attention(channel_x=64, channel_y=256)
        
        
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

        x_resnet = x.clone()
        input = Reduceinput(x)
        
        # AxialAttention Encoder
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        print("x",x.shape)

        x1 = self.layer1(x)
        
        print("x1",x1.shape)
        x2 = self.layer2(x1)
        print("x2",x2.shape)
        x3 = self.layer3(x2)
        print("x3",x3.shape)
        x4 = self.layer4(x3)
        print("x4",x4.shape)

        # x = torch.add(x, x4)  # 跳跃连接
        # print("x4 size:", x4.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = self.fusion_module1(x4,resnet_feature1) 
        #print("fusion_module:", x.size())
        x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
        # x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("cat", x.size())
        
        # print("fusion_module1", x.size())
        #print("x size:", x.size())
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature2 = F.max_pool2d(resnet_feature2, kernel_size=3, stride=1, padding=1)
        #print("resnet_feature2:", resnet_feature2.size())
        x = torch.add(x, x3)  
        x = self.fusion_module2(x,resnet_feature2) 
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=5, stride=1, padding=1)

        #print("x size:", x.size())
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature3 = F.max_pool2d(resnet_feature3, kernel_size=5, stride=1, padding=2)
        #print("resnet_feature3:", resnet_feature3.size())
        # 继续这个过程直到所有的解码层都处理完毕
        x = torch.add(x, x2)
        x = self.fusion_module3(x,resnet_feature3)          
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)

        
        # print("xsize:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature4 = F.max_pool2d(resnet_feature4, kernel_size=5, stride=1, padding=3)
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

        x = self.adjust(F.relu(x))  # 最终调整  
        
        # x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        # print("resnet_feature4:", resnet_feature4.size())
        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()s
        # x = FusionIO(input, x)
        output_2 = FusionIO(input, x)
        output_3 = FusionIO(input, output_2)
        # output_4 = FusionIO(output_3, x)w
        
        
        return x, output_2, output_3

    def forward(self, x):
        return self._forward_impl(x)

class casanet_3(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3):
        super(casanet_3, self).__init__()
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
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)

        self.decoder_e2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet50(pretrained=False)   # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = My_attention(channel_x=512, channel_y=2048)
        self.fusion_module2 = My_attention(channel_x=256, channel_y=1024)
        self.fusion_module3 = My_attention(channel_x=128, channel_y=512)
        self.fusion_module4 = My_attention(channel_x=64, channel_y=256)
        
        
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

        x_resnet = x.clone()
        input = Reduceinput(x)
        
        # AxialAttention Encoder
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("x",x.shape)

        x1 = self.layer1(x)
        
        # print("x1",x1.shape)
        x2 = self.layer2(x1)
        # print("x2",x2.shape)
        x3 = self.layer3(x2)
        # print("x3",x3.shape)
        x4 = self.layer4(x3)
        # print("x4",x4.shape)

        # x = torch.add(x, x4)  # 跳跃连接
        # print("x4 size:", x4.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = self.fusion_module1(x4,resnet_feature1) 
        # print("fusion_module:", x.size())
        x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
        # x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("cat", x.size())
        x_e = self.fusion_module1(x4,resnet_feature1)
        x_e = F.relu(F.interpolate(self.decoder_e2(x4), scale_factor=(2,2), mode='bilinear'))
        # print("fusion_module1", x.size())
        # print("x2 size:", x.size())
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature2 = F.max_pool2d(resnet_feature2, kernel_size=3, stride=1, padding=1)
        # print("resnet_feature2:", resnet_feature2.size())
        x = torch.add(x, x3)  
        x = self.fusion_module2(x,resnet_feature2) 
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        x_e = self.fusion_module2(x_e,resnet_feature2)
        x_e = F.relu(F.interpolate(self.decoder_e3(x_e), scale_factor=(2,2), mode='bilinear'))

        # print("x size:", x.size())
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature3 = F.max_pool2d(resnet_feature3, kernel_size=5, stride=1, padding=2)
        # print("resnet_feature3:", resnet_feature3.size())
        # 继续这个过程直到所有的解码层都处理完毕
        x = torch.add(x, x2)
        x = self.fusion_module3(x,resnet_feature3)          
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)

        x_e = self.fusion_module3(x_e,resnet_feature3)
        x_e = F.relu(F.interpolate(self.decoder_e4(x_e), scale_factor=(2,2), mode='bilinear'))
        
        # print("xsize:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature4 = F.max_pool2d(resnet_feature4, kernel_size=5, stride=1, padding=2)
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

        x_e = self.fusion_module4(x_e,resnet_feature4)
        x_e = F.relu(F.interpolate(self.decoder_e5(x_e), scale_factor=(2,2), mode='bilinear'))
        # x_e = torch.add(x, x_e)

        x = self.adjust(F.relu(x)) 
        x_e = self.adjust(F.relu(x_e)) # 最终调整  
        
        # x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        # print("resnet_feature4:", resnet_feature4.size())
        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()s
        # x = FusionIO(input, x)
        output_2 = FusionIO(input, x)
        output_3 = FusionIO(input, output_2)
        output_e = FusionIO(input, x_e)
        # output_4 = FusionIO(output_3, x)w
        
        
        return x, output_2, output_3, output_e 

    def forward(self, x):
        return self._forward_impl(x)

class casanet_4(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3, extend_scope=1, morph=0, if_offset=True ,device = 'cuda:1' ):
        super(casanet_4, self).__init__()
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
        self.extend_scope = 3
        self.if_offset = True
        # self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv00 = EncoderConv(imgchan, imgchan)
        self.conv0x = DSConv_pro(
            imgchan,
            imgchan,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 0,
            if_offset = self.if_offset,
            device = 'cuda:1'
            
        )
        self.conv0y = DSConv_pro(
            imgchan,
            imgchan,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 1,
            if_offset = self.if_offset,
            device = 'cuda:1'
        )
        self.conv1 = EncoderConv(3*imgchan, imgchan)

        self.conv20 = EncoderConv(imgchan, self.inplanes)
        self.conv2x = DSConv_pro(
            imgchan,
            self.inplanes,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 0,
            if_offset = self.if_offset,
            device = 'cuda:1'
        )
        self.conv2y = DSConv_pro(
            imgchan,
            self.inplanes,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 1,
            if_offset = self.if_offset,
            device = 'cuda:1'
        )
        self.conv3 = EncoderConv(3*self.inplanes, self.inplanes)

        self.maxpooling = nn.MaxPool2d(2)
        
        # self.bn1 = norm_layer(self.inplanes)
        # self.bn2 = norm_layer(128)
        # self.bn3 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)

        self.decoder_e2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet50_s(pretrained=False)   # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = My_attention(channel_x=512, channel_y=2048)
        self.fusion_module2 = My_attention(channel_x=256, channel_y=1024)
        self.fusion_module3 = My_attention(channel_x=128, channel_y=512)
        self.fusion_module4 = My_attention(channel_x=64, channel_y=256)
        
        
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

        x_resnet = x.clone()
        # input = Reduceinput1(x)
        input = Reduceinput1(x)
        # input = input.to(device='cuda:1')
        
        
        # AxialAttention Encoder
        # pdb.set_trace()
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # print("x",x.shape)
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # block1
        # x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x_0_1)
        x_2x_0 = self.conv2x(x_0_1)
        x_2y_0 = self.conv2y(x_0_1)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        x0 = self.maxpooling(x_1_1)

        x1 = self.layer1(x0)
        
        # print("x1",x1.shape)
        x2 = self.layer2(x1)
        # print("x2",x2.shape)
        x3 = self.layer3(x2)
        # print("x3",x3.shape)
        x4 = self.layer4(x3)
        # print("x4",x4.shape)

        # x = torch.add(x, x4)  # 跳跃连接
        # print("x4 size:", x4.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = self.fusion_module1(x4,resnet_feature1) 
        # print("fusion_module:", x.size())
        x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
        # x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("cat", x.size())
        # x_e = self.fusion_module1(x4,resnet_feature1)
        # x_e = F.relu(F.interpolate(self.decoder_e2(x4), scale_factor=(2,2), mode='bilinear'))
        # print("fusion_module1", x.size())
        # print("x2 size:", x.size())
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        # resnet_feature2 = F.max_pool2d(resnet_feature2, kernel_size=3, stride=1, padding=1)
        # print("resnet_feature2:", resnet_feature2.size())
        x = torch.add(x, x3)  
        x = self.fusion_module2(x,resnet_feature2) 
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        # x_e = self.fusion_module2(x_e,resnet_feature2)
        # x_e = F.relu(F.interpolate(self.decoder_e3(x_e), scale_factor=(2,2), mode='bilinear'))

        # print("x size:", x.size())
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        # resnet_feature3 = F.max_pool2d(resnet_feature3, kernel_size=5, stride=1, padding=2)
        # print("resnet_feature3:", resnet_feature3.size())
        # 继续这个过程直到所有的解码层都处理完毕
        x = torch.add(x, x2,)
        x = self.fusion_module3(x,resnet_feature3)          
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)

        # x_e = self.fusion_module3(x_e,resnet_feature3)
        # x_e = F.relu(F.interpolate(self.decoder_e4(x_e), scale_factor=(2,2), mode='bilinear'))
        
        # print("xsize:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        resnet_feature4 = F.max_pool2d(resnet_feature4, kernel_size=5, stride=1, padding=2)
        # print("resnet_feature4:", resnet_feature4.size())
        x = self.fusion_module4(x,resnet_feature4) 
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

        # x_e = self.fusion_module4(x_e,resnet_feature4)
        # x_e = F.relu(F.interpolate(self.decoder_e5(x_e), scale_factor=(2,2), mode='bilinear'))

        # x_e = torch.add(x, x_e)

        x = self.adjust(F.relu(x)) 
        # x_e = self.adjust(F.relu(x_e)) # 最终调整  
        
        # x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        # print("resnet_feature4:", resnet_feature4.size())
        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()s
        # x = FusionIO(input, x)
        output_2 = FusionIO_1(input, x)
        output_3 = FusionIO_1(input, output_2)
        
        # output_2 = afterFusion( x, input)
        # output_3 = afterFusion( output_2, input)
        # output_e = FusionIO_1(input, x_e)
        # output_4 = FusionIO(output_3, x)w
        
        
        # return x, output_2, output_3, output_e 
        return x, output_2, output_3

    def forward(self, x):
        return self._forward_impl(x)


class casanet_5(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 128,imgchan = 3, extend_scope=1, morph=0, if_offset=True ,device = 'cuda:0' ):
        super(casanet_5, self).__init__()
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
        self.extend_scope = 1
        self.if_offset = True
        # self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv00 = EncoderConv(imgchan, imgchan)
        self.conv0x = DSConv_pro(
            imgchan,
            imgchan,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 0,
            if_offset = self.if_offset,
            device = 'cuda:0'
            
        )
        self.conv0y = DSConv_pro(
            imgchan,
            imgchan,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 1,
            if_offset = self.if_offset,
            device = 'cuda:0'
        )
        self.conv1 = EncoderConv(3*imgchan, imgchan)

        self.conv20 = EncoderConv(imgchan, self.inplanes)
        self.conv2x = DSConv_pro(
            imgchan,
            self.inplanes,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 0,
            if_offset = self.if_offset,
            device = 'cuda:0'
        )
        self.conv2y = DSConv_pro(
            imgchan,
            self.inplanes,
            kernel_size=9,
            extend_scope = self.extend_scope,
            morph = 1,
            if_offset = self.if_offset,
            device = 'cuda:0'
        )
        self.conv3 = EncoderConv(3*self.inplanes, self.inplanes)

        self.maxpooling = nn.MaxPool2d(2)
        
        # self.bn1 = norm_layer(self.inplanes)
        # self.bn2 = norm_layer(128)
        # self.bn3 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(256*s) , num_classes, kernel_size=1, stride=1, padding=0)

        self.decoder_e2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder_e5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

        # self.soft     = nn.Softmax(dim=1)

        self.resnet_branch = resnet50(pretrained=False)   # 不使用预训练权重
        self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

        # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
        # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
        # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
        # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))

        self.fusion_module1 = My_attention(channel_x=512, channel_y=2048)
        self.fusion_module2 = My_attention(channel_x=256, channel_y=1024)
        self.fusion_module3 = My_attention(channel_x=128, channel_y=512)
        self.fusion_module4 = My_attention(channel_x=64, channel_y=256)
        
        
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

        # print(kernel_size)
        # print(img_size)

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x_resnet = x.clone()
        input = Reduceinput(x)
        # input = input.to(device='cuda:1')
        
        
        # AxialAttention Encoder
        # pdb.set_trace()
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # print("x",x.shape)
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # x_0_1 = self.maxpooling(x_0_1)

        # block1
        # x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x_0_1)
        x_2x_0 = self.conv2x(x_0_1)
        x_2y_0 = self.conv2y(x_0_1)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        x1 = self.layer1(x_1_1)
        
        # print("x1",x1.shape)
        x2 = self.layer2(x1)
        # print("x2",x2.shape)
        x3 = self.layer3(x2)
        # print("x3",x3.shape)
        x4 = self.layer4(x3)
        # print("x4",x4.shape)

        # x = torch.add(x, x4)  # 跳跃连接
        # print("x4 size:", x4.size())
        resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
        # print("resnet_feature1:", resnet_feature1.size())
        x = self.fusion_module1(x4,resnet_feature1) 
        # print("fusion_module:", x.size())
        x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        # x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("cat", x.size())
        # x_e = self.fusion_module1(x4,resnet_feature1)
        # x_e = F.relu(F.interpolate(self.decoder_e2(x4), scale_factor=(2,2), mode='bilinear'))
        # print("fusion_module1", x.size())
        # print("x2 size:", x.size())
        resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
        # resnet_feature2 = F.max_pool2d(resnet_feature2, kernel_size=3, stride=1, padding=1)
        # print("resnet_feature2:", resnet_feature2.size())
        x = torch.add(x, x3)  
        x = self.fusion_module2(x,resnet_feature2) 
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        # x_e = self.fusion_module2(x_e,resnet_feature2)
        # x_e = F.relu(F.interpolate(self.decoder_e3(x_e), scale_factor=(2,2), mode='bilinear'))

        # print("x size:", x.size())
        resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
        # resnet_feature3 = F.max_pool2d(resnet_feature3, kernel_size=3, stride=1, padding=2)
        # print("resnet_feature3:", resnet_feature3.size())
        # 继续这个过程直到所有的解码层都处理完毕
        # x = torch.add(x, x2)
        x = self.fusion_module3(x,resnet_feature3)          
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        # x_e = self.fusion_module3(x_e,resnet_feature3)
        # x_e = F.relu(F.interpolate(self.decoder_e4(x_e), scale_factor=(2,2), mode='bilinear'))
        
        # print("xsize3:", x.size())
        resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
        x = self.fusion_module4(x,resnet_feature4) 
        x = self.adjust(F.relu(x)) 
        # resnet_feature4 = F.max_pool2d(resnet_feature4, kernel_size=5, stride=1, padding=2)
        # print("resnet_feature4:", resnet_feature4.size())

        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

        # x_e = self.fusion_module4(x_e,resnet_feature4)
        # x_e = F.relu(F.interpolate(self.decoder_e5(x_e), scale_factor=(2,2), mode='bilinear'))
        # x_e = torch.add(x, x_e)

        # x_e = self.adjust(F.relu(x_e)) # 最终调整  
        
        # x = torch.add(x, x1)
        # print("x, x1 size:", x.size())
        # print("resnet_feature4:", resnet_feature4.size())
        # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x4)
        # print("xx4", x.size())
        # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x3)
        # print("xx3", x.size())
        # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x2)
        # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = torch.add(x, x1)
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        # x = self.adjust(F.relu(x))
        # pdb.set_trace()s
        # x = FusionIO(input, x)
        output_2 = FusionIO(input, x)
        output_3 = FusionIO(input, output_2)
        # output_e = FusionIO_1(input, x_e)
        # output_4 = FusionIO(output_3, x)w
        
        
        return x, output_2, output_3,

    def forward(self, x):
        return self._forward_impl(x)


# class casanet_hr(nn.Module):

#     def __init__(self, block, layers, num_classes=2, zero_init_residual=True,
#                  groups=8, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, s=0.125, img_size = 128,imgchan = 3, extend_scope=1, morph=0, if_offset=True ,device = 'cuda:0' ):
#         super(casanet_hr, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = int(64 * s)
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.extend_scope = 1
#         self.if_offset = True
#         # self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
#         #                        bias=False)
#         # self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

#         self.conv00 = EncoderConv(imgchan, imgchan)
#         self.conv0x = DSConv_pro(
#             imgchan,
#             imgchan,
#             kernel_size=9,
#             extend_scope = self.extend_scope,
#             morph = 0,
#             if_offset = self.if_offset,
#             device = 'cuda:0'
            
#         )
#         self.conv0y = DSConv_pro(
#             imgchan,
#             imgchan,
#             kernel_size=9,
#             extend_scope = self.extend_scope,
#             morph = 1,
#             if_offset = self.if_offset,
#             device = 'cuda:0'
#         )
#         self.conv1 = EncoderConv(3*imgchan, imgchan)

#         self.conv20 = EncoderConv(imgchan, self.inplanes)
#         self.conv2x = DSConv_pro(
#             imgchan,
#             self.inplanes,
#             kernel_size=9,
#             extend_scope = self.extend_scope,
#             morph = 0,
#             if_offset = self.if_offset,
#             device = 'cuda:0'
#         )
#         self.conv2y = DSConv_pro(
#             imgchan,
#             self.inplanes,
#             kernel_size=9,
#             extend_scope = self.extend_scope,
#             morph = 1,
#             if_offset = self.if_offset,
#             device = 'cuda:0'
#         )
#         self.conv3 = EncoderConv(3*self.inplanes, self.inplanes)

#         self.maxpooling = nn.MaxPool2d(2)
        
#         # self.bn1 = norm_layer(self.inplanes)
#         # self.bn2 = norm_layer(128)
#         # self.bn3 = norm_layer(self.inplanes)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size))
#         self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size),
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//2),
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//4),
#                                        dilate=replace_stride_with_dilation[2])
        
#         # Decoder
#         # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
#         self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
#         self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
#         self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
#         self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
#         self.adjust   = nn.Conv2d(int(256*s) , num_classes, kernel_size=1, stride=1, padding=0)

#         self.decoder_e2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
#         self.decoder_e3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
#         self.decoder_e4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
#         self.decoder_e5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

#         # self.soft     = nn.Softmax(dim=1)

#         self.resnet_branch = resnet50(pretrained=False)   # 不使用预训练权重
#         self.resnet_branch = nn.Sequential(*list(self.resnet_branch.children())[:-2])

#         # self.conv_reduce1 = conv1x1(int(1024 * 2 * 3 * s ), int(1024 * 2 * s))
#         # self.conv_reduce2 = conv1x1(int(1024 * 3 * s), int(1024 * s))
#         # self.conv_reduce3 = conv1x1(int(512 *  3 * s), int(512 * s))
#         # self.conv_reduce4 = conv1x1(int(256 *  3 * s), int(256 * s))
#         FusionModuleCA_1 = FusionModuleCA(atn_channels=128, cnn_channels=128)
#         FusionModuleCA_2 = FusionModuleCA(atn_channels=128, cnn_channels=128)
#         FusionModuleCA_3 = FusionModuleCA(atn_channels=128, cnn_channels=128)
#         FusionModuleCA_4 = FusionModuleCA(atn_channels=128, cnn_channels=128)

#         snake1 = SnakeConvBlock(in_channels=128, out_channels=128)
#         snake2 = SnakeConvBlock(in_channels=128, out_channels=128)
#         snake3 = SnakeConvBlock(in_channels=128, out_channels=128)
#         snake4 = SnakeConvBlock(in_channels=128, out_channels=128)
        
#         upsample_block2-1 = UpSampleBlock(in_channels=128, out_channels=128)
#         upsample_block3-1 = UpSampleBlock(in_channels=128, out_channels=128)
#         upsample_block4-1 = UpSampleBlock(in_channels=128, out_channels=128)
#         upsample_block3-2 = UpSampleBlock(in_channels=128, out_channels=128)
#         upsample_block4-2 = UpSampleBlock(in_channels=128, out_channels=128)
#         upsample_block4-3 = UpSampleBlock(in_channels=128, out_channels=128)

#         downsample_block1-2 = DownSampleBlock(in_channels=128, out_channels=128)
#         downsample_block1-3 = DownSampleBlock(in_channels=128, out_channels=128)
#         downsample_block1-4 = DownSampleBlock(in_channels=128, out_channels=128)
#         downsample_block2-3 = DownSampleBlock(in_channels=128, out_channels=128)
#         downsample_block2-4 = DownSampleBlock(in_channels=128, out_channels=128)
#         downsample_block3-4 = DownSampleBlock(in_channels=128, out_channels=128)

#         self.fusion_module1 = My_attention(channel_x=512, channel_y=2048)
#         self.fusion_module2 = My_attention(channel_x=256, channel_y=1024)
#         self.fusion_module3 = My_attention(channel_x=128, channel_y=512)
#         self.fusion_module4 = My_attention(channel_x=64, channel_y=256)

        
        
        
#     def get_resnet_feature(self, x, size):
# #         print(f"Requested size: {size}")
#         found = False
#         layer_index = 0

#         x = self.resnet_branch[0](x)  # conv1
#         x = self.resnet_branch[1](x)  # bn1
#         x = self.resnet_branch[2](x)  # relu
#         x = self.resnet_branch[3](x)  # maxpool

#         for layer in range(4, 8):  # layer1 to layer4
#             x = self.resnet_branch[layer](x)
#             layer_index = layer
#             if x.size(2) == size or x.size(3) == size:
#                 found = True
#                 break

#         if found:
# #             print(f"Matching size found in ResNet layer {layer_index}: {x.size()}")
#             return x
#         else:
# #             print("Matching size not found in any ResNet layer.")
#             raise ValueError("Requested size not found in ResNet features")


#     def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
#                             base_width=self.base_width, dilation=previous_dilation, 
#                             norm_layer=norm_layer, kernel_size=kernel_size))
#         self.inplanes = planes * block.expansion
#         if stride != 1:
#             kernel_size = kernel_size // 2

#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, kernel_size=kernel_size))

#         # print(kernel_size)
#         # print(img_size)

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x):

#         x_resnet = x.clone()
#         input = Reduceinput(x)
#         # input = input.to(device='cuda:1')
        
        
#         # AxialAttention Encoder
#         # pdb.set_trace()
#         # x = self.conv1(x)
#         # x = self.bn1(x)
#         # x = self.relu(x)
#         # x = self.conv2(x)
#         # x = self.bn2(x)
#         # x = self.relu(x)
#         # x = self.conv3(x)
#         # x = self.bn3(x)
#         # x = self.relu(x)
#         # print("x",x.shape)
#         # block0
#         x_00_0 = self.conv00(x)
#         x_0x_0 = self.conv0x(x)
#         x_0y_0 = self.conv0y(x)
#         x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

#         # x_0_1 = self.maxpooling(x_0_1)

#         # block1
#         # x = self.maxpooling(x_0_1)
#         x_20_0 = self.conv20(x_0_1)
#         x_2x_0 = self.conv2x(x_0_1)
#         x_2y_0 = self.conv2y(x_0_1)
#         x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

#         x1 = self.layer1(x_1_1)
        
#         # print("x1",x1.shape)
#         x2 = self.layer2(x1)
#         # print("x2",x2.shape)
#         x3 = self.layer3(x2)
#         # print("x3",x3.shape)
#         x4 = self.layer4(x3)
#         # print("x4",x4.shape)

#         resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])
#         resnet_feature2 = self.get_resnet_feature(x_resnet, x3.size()[2])
#         resnet_feature3 = self.get_resnet_feature(x_resnet, x2.size()[2])
#         resnet_feature4 = self.get_resnet_feature(x_resnet, x1.size()[2])

#         x_f1 = FusionModuleCA_1(x4,resnet_feature1)
#         x_f2 = FusionModuleCA_2(x3,resnet_feature2) 
#         x_f3 = FusionModuleCA_3(x2,resnet_feature3) 
#         x_f4 = FusionModuleCA_4_(x1,resnet_feature4) 

#         x_f1 = self.snake1(x_f1)
#         x_f2 = self.snake2(x_f2)
#         x_f3 = self.snake3(x_f3)
#         x_f4 = self.snake4(x_f4)

#         x_f1_2 = self.downsample_block1_2(x_f1)
#         x_f1_3 = self.downsample_block1_3(x_f1)
#         x_f1_4 = self.downsample_block1_4(x_f1)

#         x_f2_1 = self.upsample_block2-1(x_f2)
#         x_f2_3 = self.downsample_block2_3(x_f2)
#         x_f2_4 = self.downsample_block2_4(x_f2)

#         x_f3_1 = self.downsample_block3_1(x_f3)
#         x_f3_2 = self.downsample_block3_2(x_f3)
#         x_f3_4 = self.down(x_f3)

#         x_f4_1 = self.upsample_block4_1(x_f4)
#         x_f4_2 = self.upsample_block4_2(x_f4)
#         x_f4_3 = self.upsample_block4_3(x_f4)

#         x_f1 = torch.add(x_f1,x_f2_1,x_f3_1,x_f4_1)
#         x_f2 = torch.add(x_f2,x_f1_2,x_f3_2,x_f4_2)
#         x_f3 = torch.add(x_f3,x_f1_3,x_f2_3,x_f4_3)
#         x_f4 = torch.add(x_f4,x_f1_4,x_f2_4,x_f3_4)

#         x_f1 = self.snake1(x_f1)
#         x_f1 = self.snake1(x_f1)
#         x_f2 = self.snake2(x_f2)
#         x_f2 = self.snake2(x_f2)
#         x_f3 = self.snake3(x_f3)
#         x_f3 = self.snake3(x_f3)
#         x_f4 = self.snake4(x_f4)
#         x_f4 = self.snake4(x_f4)

#         x = self.fusion_module1(x4,resnet_feature1) 
#         x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))

        
#         # x = torch.add(x, x4)  # 跳跃连接
#         # print("x4 size:", x4.size())
#         resnet_feature1 = self.get_resnet_feature(x_resnet, x4.size()[2])  # 根据当前特征图的尺寸提取ResNet特征
#         # print("resnet_feature1:", resnet_feature1.size())
#         x = self.fusion_module1(x4,resnet_feature1) 
#         # print("fusion_module:", x.size())
#         x = F.relu(F.interpolate(self.decoder2(x4), scale_factor=(2,2), mode='bilinear'))
#         x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        
#         # x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
#         # print("cat", x.size())
#         # x_e = self.fusion_module1(x4,resnet_feature1)
#         # x_e = F.relu(F.interpolate(self.decoder_e2(x4), scale_factor=(2,2), mode='bilinear'))
#         # print("fusion_module1", x.size())
#         # print("x2 size:", x.size())
#         resnet_feature2 = self.get_resnet_feature(x_resnet, x.size()[2])
#         # resnet_feature2 = F.max_pool2d(resnet_feature2, kernel_size=3, stride=1, padding=1)
#         # print("resnet_feature2:", resnet_feature2.size())
#         x = torch.add(x, x3)  
#         x = self.fusion_module2(x,resnet_feature2) 
#         x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2), mode='bilinear'))
#         x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

#         # x_e = self.fusion_module2(x_e,resnet_feature2)
#         # x_e = F.relu(F.interpolate(self.decoder_e3(x_e), scale_factor=(2,2), mode='bilinear'))

#         # print("x size:", x.size())
#         resnet_feature3 = self.get_resnet_feature(x_resnet, x.size()[2])
#         # resnet_feature3 = F.max_pool2d(resnet_feature3, kernel_size=3, stride=1, padding=2)
#         # print("resnet_feature3:", resnet_feature3.size())
#         # 继续这个过程直到所有的解码层都处理完毕
#         # x = torch.add(x, x2)
#         x = self.fusion_module3(x,resnet_feature3)          
#         x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2), mode='bilinear'))
#         x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

#         # x_e = self.fusion_module3(x_e,resnet_feature3)
#         # x_e = F.relu(F.interpolate(self.decoder_e4(x_e), scale_factor=(2,2), mode='bilinear'))
        
#         # print("xsize3:", x.size())
#         resnet_feature4 = self.get_resnet_feature(x_resnet, x.size()[2])
#         x = self.fusion_module4(x,resnet_feature4) 
#         x = self.adjust(F.relu(x)) 
#         # resnet_feature4 = F.max_pool2d(resnet_feature4, kernel_size=5, stride=1, padding=2)
#         # print("resnet_feature4:", resnet_feature4.size())

#         # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))

#         # x_e = self.fusion_module4(x_e,resnet_feature4)
#         # x_e = F.relu(F.interpolate(self.decoder_e5(x_e), scale_factor=(2,2), mode='bilinear'))
#         # x_e = torch.add(x, x_e)

#         # x_e = self.adjust(F.relu(x_e)) # 最终调整  
        
#         # x = torch.add(x, x1)
#         # print("x, x1 size:", x.size())
#         # print("resnet_feature4:", resnet_feature4.size())
#         # x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
#         # 如果有需要，这里也可以添加额外的ResNet特征拼接
        
        
#         # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x4)
#         # print("xx4", x.size())
#         # x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x3)
#         # print("xx3", x.size())
#         # x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x2)
#         # x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
#         # x = torch.add(x, x1)
#         # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
#         # x = self.adjust(F.relu(x))
#         # pdb.set_trace()s
#         # x = FusionIO(input, x)
#         output_2 = FusionIO(input, x)
#         output_3 = FusionIO(input, output_2)
#         # output_e = FusionIO_1(input, x_e)
#         # output_4 = FusionIO(output_3, x)w
        
        
#         return x, output_2, output_3,

#     def forward(self, x):
#         return self._forward_impl(x)

def ARUNet(pretrained=False, **kwargs):
    model = AxialResUNet_4(AxialBlock_wof, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

def AAFSRnet(pretrained=False, **kwargs):
    model = casanet(AxialBlock_wof, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model
    
def AAFSRnet_2(pretrained=False, **kwargs):
    model = casanet_2(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.25, **kwargs)
    return model

def AAFSRnet_3(pretrained=False, **kwargs):
    model = casanet_3(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.25, **kwargs)
    return model

def AAFSRnet_4(pretrained=False, **kwargs):
    model = casanet_4(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.25, **kwargs)
    return model

def AAFSRnet_5(pretrained=False, **kwargs):
    model = casanet_5(AxialBlock_dynamic_L, [1, 2, 4, 1], s= 0.25, **kwargs)
    return model

