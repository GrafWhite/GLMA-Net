import torch
import torch.nn as nn
import torch.nn.functional as F
# from afterfusion import FusionModel1, FusionModel2, FusionModel3
# from diceloss import dice_loss  # 假设你已经有了dice_loss的实现
# from loss_fd import Focal_Loss, Dice_loss

class ReduceDimModule(nn.Module):
    def __init__(self, device='cuda:1'):
        super(ReduceDimModule, self).__init__()
        # 设定设备
        self.device = device
        # 因为输入通道数已知为4，我们在这里静态创建卷积层
        # 这个卷积层将输入的4通道特征图降维到2通道
        # 直接在指定的CUDA设备上创建卷积层
        self.conv_reduce_dim = nn.Conv2d(4, 2, kernel_size=1).to(self.device)

    def forward(self, x):
        # 确保输入数据也在相同的设备上
        x = x.to(self.device)
        # 应用卷积层降维
        x = self.conv_reduce_dim(x)
        return x


class ReduceDimModule_1(nn.Module):
    def __init__(self):
        super(ReduceDimModule_1, self).__init__()
        # 因为输入通道数已知为4，我们在这里静态创建卷积层
        # 这个卷积层将输入的4通道特征图降维到2通道
        self.conv_reduce_dim = nn.Conv2d(3, 2, kernel_size=1).cuda()

    def forward(self, x):
        # 应用卷积层降维
        x = self.conv_reduce_dim(x)
        return x

class EdgeEnhancementModule(nn.Module):
    def __init__(self, device='cuda'):
        super(EdgeEnhancementModule, self).__init__()
        # Sobel算子初始化
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        # 设置Sobel核心，不需要梯度更新
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=torch.float32)
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        # 1x1卷积，将7通道融合回4通道
        self.conv1x1 = nn.Conv2d(7, 4, 1, bias=True)
        self.conv1x1 =  self.conv1x1.to(device)
        # self.conv1x1.weight.data = self.conv1x1.weight.data.half()
        # if self.conv1x1.bias is not None:
        #     self.conv1x1.bias.data = self.conv1x1.bias.data.half()  # 转换偏置为半精度

        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        edge_channels = []
        for i in range(3):
            gray = rgb[:, i:i+1, :, :]
            edge_x = self.sobel_x(gray)
            edge_y = self.sobel_y(gray)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            edge_channels.append(edge)
        edges = torch.cat(edge_channels, dim=1)
        combined = torch.cat([x, edges], dim=1)
        output = self.conv1x1(combined)
        attention = self.spatial_attention(output)
        output = output * (1 + attention)
        return output



# # 示例用法
# # 假设 input_tensor 是从 DataLoader 获取的，形状为 [batch_size, 4, H, W]
# model = EdgeEnhancementModule()
# output_tensor = model(input_tensor)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class Spatial_attention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(Spatial_attention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class FusionModule(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(FusionModule, self).__init__()
#         # 使用Spacial_attention
#         self.spatial_attention = Spatial_attention(kernel_size=kernel_size)

#     def forward(self, input1, input2):
#         # 直接相加两个输入，不进行降维
#         # input1 = input1.half()
#         # input2 = input2.half()
#         added_inputs = input1 + input2
#         added_inputs = added_inputs.float()
#         # 空间注意力融合
#         attention_output = self.spatial_attention(added_inputs)
#         # 残差连接
#         output = attention_output + added_inputs
#         return output

class FusionModule2(nn.Module):
    def __init__(self, kernel_size=7):
        super(FusionModule2, self).__init__()
        # 使用Spacial_attention
        self.spatial_attention = Spatial_attention(kernel_size=kernel_size)

    def forward(self, input1, input2):
        # 直接相加两个输入，不进行降维
        input1 = input1_tensor.half()
        input2 = input2_tensor.half()
        added_inputs = input1 + input2
        # 空间注意力融合
        attention_output = self.spatial_attention(added_inputs)
        # 残差连接
        output = attention_output + added_inputs
        return output

class Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spacial_attention, self).__init__()
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

class FusionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(FusionModule, self).__init__()
        # 使用Spacial_attention
        self.spatial_attention = Spacial_attention(kernel_size=kernel_size)

    def forward(self, input1, input2):
        # 直接相加两个输入，不进行降维
        added_inputs = input1 + input2
        # 空间注意力融合
        attention_output = self.spatial_attention(added_inputs)
        # 残差连接
        output = attention_output + added_inputs
        return output

# 使用示例
# 假定输入具有特定的通道数和尺寸，这里只是为了示例而已
# input1 和 input2 的尺寸需要匹配
# input1 = torch.randn([batch_size, channel_count, height, width]).cuda()
# input2 = torch.randn([batch_size, channel_count, height, width]).cuda()
# fusion_module = FusionModule().cuda()
# output = fusion_module(input1, input2)
