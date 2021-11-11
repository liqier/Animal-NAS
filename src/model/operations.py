import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys 
sys.path.append("..") 
from args import helper

train_queue, valid_queue, data_shape, num_class, A, parts = helper.get_train_val_loaders()
A = torch.from_numpy(A)
kernel_size=[9,2]
temporal_window_size, max_graph_distance = kernel_size
parts = [torch.tensor([ 3,  8,  9, 10]), torch.tensor([ 3, 11, 12]), torch.tensor([ 5, 13, 14, 15]), torch.tensor([ 5, 16, 17]),torch.tensor([ 1, 0, 2,3,4,5,6,7])]
OPS = {
    'noise': lambda C, stride, affine: NoiseOp(stride, 0., 1.),
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'Part_Att_bottleneck':lambda C, stride, affine: Part_Att_bottleneck(C, C,  A, parts, kernel_size=[9,2]),
    'Part_Share_Att_bottleneck':lambda C, stride, affine: Part_Share_Att_bottleneck(C, C, parts, A ,kernel_size=[9,2]),
    'Part_Conv_Att_bottleneck':lambda C, stride, affine: Part_Conv_Att_bottleneck(C, C, parts, A ,kernel_size=[9,2]),
    'Joint_Att_bottleneck':lambda C, stride, affine: Joint_Att_bottleneck(C,C, parts, A, kernel_size=[9,2], stride=1),
    'Frame_Att_bottleneck':lambda C, stride, affine: Frame_Att_bottleneck(C, C, A, kernel_size=[9,2], stride=1),
    'Channel_Att_bottleneck':lambda C, stride, affine: Channel_Att_bottleneck(C, C, A, kernel_size=[9,2], stride=1),
    'Spatial_Bottleneck_Block':lambda C, stride, affine: Spatial_Bottleneck_Block(C, C, max_graph_distance, True, affine=affine),
    'Temporal_Bottleneck_Block':lambda C, stride, affine: Temporal_Bottleneck_Block(C, temporal_window_size, stride, True, affine=affine),
    'Spatial_Basic_Block':lambda C, stride, affine: Spatial_Basic_Block(C, C, max_graph_distance, False),
    'Temporal_Basic_Block':lambda C, stride, affine: Temporal_Basic_Block(C, temporal_window_size, stride, False, affine=affine),
    'Basic_bottleneck':lambda C, stride, affine: Basic_bottleneck(C, C,  A, kernel_size=[9,2], stride=1),
    'Basic_net':lambda C, stride, affine: Basic_net(C, C,  A, kernel_size=[9,2], stride=1),
    'SpatialGraphConv':lambda C, stride, affine: SpatialGraphConv(C, C, max_graph_distance,  affine=affine),
    'Part_Att': lambda C, stride, affine: Part_Att(C, parts, affine=affine),
    'Part_Share_Att': lambda C, stride, affine: Part_Share_Att(C, parts, affine=affine),
    'Part_Conv_Att': lambda C, stride, affine: Part_Conv_Att(C, parts, affine=affine),
    'Joint_Att': lambda C, stride, affine: Joint_Att(C, parts, affine=affine),
    'Frame_Att': lambda C, stride, affine: Frame_Att(C, affine=affine),
    'Channel_Att': lambda C, stride, affine: Channel_Att(C, affine=affine),

}

class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
          x_new = x[:,:,::self.stride,::self.stride]
        else:
          x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
        return noise


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x,AS):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class Part_Att(nn.Module):
    def __init__(self, channel, parts, affine=True):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel,affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_att = self.softmax(self.fcn(x).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Part_Share_Att(nn.Module):
    def __init__(self, channel, parts, affine=True):
        super(Part_Share_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel,affine=affine),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel,affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [self.part_pool(x[:,:,:,part]) for part in self.parts]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Part_Conv_Att(nn.Module):
    def __init__(self, channel, parts, affine=True):
        super(Part_Conv_Att, self).__init__()

        self.parts = parts
        self.joints = get_corr_joints(parts)

        inter_channel = channel // 4

        self.part_pool = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel,affine=affine),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ) for _ in range(len(self.parts))])

        self.fcn = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel,affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channel*len(self.parts), kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channel,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x

        x_split = [pool(x[:,:,:,part]) for part, pool in zip(self.parts, self.part_pool)]
        x_att = self.softmax(self.fcn(sum(x_split)).view(N, C, len(self.parts)))
        x_att = torch.split(x_att, 1, dim=-1)
        x_att = [x_att[self.joints[i]].expand_as(x[:,:,:,i]) for i in range(V)]
        x_att = torch.stack(x_att, dim=-1)
        return self.relu(self.bn(x * x_att) + res)


class Channel_Att(nn.Module):
    def __init__(self, channel, affine=True):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4,affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(channel,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_att = self.fcn(x).squeeze()
        return self.relu(self.bn(x * x_att[:, :, None, None]) + res)

class Joint_Att(nn.Module):
    def __init__(self, channel, parts, affine=True):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2,affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(channel,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_att = self.fcn(torch.transpose(x, 1, 3)).squeeze()
        return self.relu(self.bn(x * x_att[:, None, None, :]) + res)

class Frame_Att(nn.Module):
    def __init__(self, channel, affine=True):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))
        self.bn = nn.BatchNorm2d(channel,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_avg = torch.transpose(self.avg_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_max = torch.transpose(self.max_pool(torch.transpose(x, 1, 2)), 1, 2)
        x_att = self.conv(torch.cat([x_avg, x_max], dim=1)).squeeze()
        return self.relu(self.bn(x * x_att[:, None, :, None]) + res)

def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()

class Spatial_Bottleneck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance,  residual=False, reduction=4, affine=True, **kwargs):
        super(Spatial_Bottleneck_Block, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels,affine=affine),
            )

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels,affine=affine)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels,affine=affine)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels,affine=affine)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, At):

        res_block = self.residual(x)
        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, At)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        return x


class Temporal_Bottleneck_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4, affine=True):
        super(Temporal_Bottleneck_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels,affine=affine),
            )

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels,affine=affine)
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(inter_channels,affine=affine)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)

        return x


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, affine=True):
        super(Spatial_Basic_Block, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels,affine=affine),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels,  max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels,affine=affine)
        self.relu = nn.ReLU(inplace=True)
        #self.A = A

    def forward(self, x, At):

        res_block = self.residual(x)
        #print('x.type:{}'.format(x.dtype))
        #x = torch.tensor(x, dtype=torch.float).to('cuda')
        x = self.conv(x, At)
        x = x.float().to('cuda')
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x


class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, affine=True):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels,affine=affine),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels,affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels,max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)


    def forward(self, x, At):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        A = torch.tensor(At, dtype=torch.float32).to('cuda')
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()   #使用爱因斯坦求和约定来计算多线性表达式（即乘积之和）的方法。

        return x

class Part_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, A, parts, kernel_size=[9,2], stride=1):
        super(Part_Att_bottleneck, self).__init__()
        self.AH = A
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res = True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.att = Part_Att(out_channels, parts)
        self.edge = nn.Parameter(torch.ones_like(self.AH))
    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.att(self.tcn(self.scn(x, A*self.edge), self.residual(x)))


class Part_Share_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, parts, A, kernel_size=[9,2], stride=1):
        super(Part_Share_Att_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.att = Part_Share_Att(out_channels, parts, affine=True)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.att(self.tcn(self.scn(x, A * self.edge), self.residual(x)))


class Part_Conv_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, parts, A , kernel_size=[9,2], stride=1):
        super(Part_Conv_Att_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.att = Part_Conv_Att(out_channels,parts)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.att(self.tcn(self.scn(x, A * self.edge), self.residual(x)))

class Channel_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=[9,2], stride=1):
        super(Channel_Att_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.att = Channel_Att(out_channels)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.att(self.tcn(self.scn(x, A * self.edge), self.residual(x)))

class Joint_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, parts, A, kernel_size=[9,2], stride=1):
        super(Joint_Att_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.att = Joint_Att(out_channels, parts)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.float32).to('cuda')
        return self.att(self.tcn(self.scn(x, A * self.edge), self.residual(x)))

class Frame_Att_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, A,  kernel_size=[9,2], stride=1):
        super(Frame_Att_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride )
        self.att = Frame_Att(out_channels)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.att(self.tcn(self.scn(x, A * self.edge), self.residual(x)))
#

class Basic_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, A,  kernel_size=[9,2], stride=1):
        super(Basic_bottleneck, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = False, True
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.scn = Spatial_Bottleneck_Block(in_channels, out_channels, max_graph_distance, block_res=True)
        self.tcn = Temporal_Bottleneck_Block(out_channels, temporal_window_size, stride)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = torch.tensor(AS, dtype=torch.double).to('cuda')
        return self.tcn(self.scn(x, A * self.edge), self.residual(x))

class Basic_net(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=[9,2], stride=1):
        super(Basic_net, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        module_res, block_res = True, False
        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        self.scn = Spatial_Basic_Block(in_channels, out_channels, max_graph_distance, block_res)
        self.tcn = Temporal_Basic_Block(out_channels, temporal_window_size, stride)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, AS):
        A = AS.cuda()
        return self.tcn(self.scn(x, A * self.edge), self.residual(x))