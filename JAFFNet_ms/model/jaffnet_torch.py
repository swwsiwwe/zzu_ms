import torch
# from torchvision import models
import torch.nn.functional as F
# from resnet_model import *
import pandas as pd

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDe(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockDe, self).__init__()

        self.convRes = conv3x3(inplanes,planes,stride)
        self.bnRes = nn.BatchNorm2d(planes)
        self.reluRes = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# PyTorch  导入包
# import msadapter.pytorch as torch
# import msadapter.pytorch.nn as nn
# from msadapter.pytorch.utils.data import DataLoader
# from msadapter.torchvision import datasets, transforms
# from msadapter.torchvision.transforms.functional import InterpolationMode as F
#
# import mindspore as ms
# import argparse

class ResnetBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1) -> None:
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.downsample(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResBlock, num_classes=1000) -> None:
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.shape(0), -1)
        out = self.fc(out)
        return out
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

class BasicBlockV1b(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MRF(nn.Module):
    def __init__(self, in_channels, out_channels,rate=[1,2,4,8]):
        super(MRF, self).__init__()

        inner_channels=in_channels//4

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            )


        self.pro = nn.Sequential(
            nn.Conv2d(inner_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)


        out = torch.cat([feat1, feat2, feat3], dim=1)
        return self.pro(out)

class DRF(nn.Module):
    def __init__(self, in_channels):
        super(DRF, self).__init__()
        out_channels =512
        self.mrf1 = MRF(in_channels,out_channels)
        self.mrf2 = MRF(in_channels,out_channels)
        self.mrf3 = MRF(in_channels,out_channels)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.relu= nn.ReLU(inplace=True)
        self.pro = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        size = x.size()[2:]
        feat1 = self.mrf1(x)
        input = x + feat1
        feat2 = self.mrf2(input)

        input = x+feat1+feat2

        feat3 = self.mrf3(input)
        feat4 = F.interpolate(self.conv4(self.gap(x)), size, mode='bilinear', align_corners=True)
        feat = x+feat1+feat2+feat3+feat4
        out = self.pro(feat)
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return avgout + maxout


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out=self.conv2d(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c_out = self.channel_attention(x)
        s_out = self.spatial_attention(x)
        return self.sigmoid(c_out+s_out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        gate_channels = [gate_channel]  # eg 64
        gate_channels += [gate_channel // reduction_ratio] * num_layers  # eg 4
        gate_channels += [gate_channel]  # 64
        # gate_channels: [64, 4, 4]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                'gate_c_fc_%d' % i,
                nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
                                   nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())

        self.gate_c.add_module('gate_c_fc_final',
                               nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)

class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=16,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()

        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,
                      gate_channel // reduction_ratio,
                      kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',
                               nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(gate_channel // reduction_ratio,
                          gate_channel // reduction_ratio,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, x):
        return self.gate_s(x).expand_as(x)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, x):
        att = F.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class JAFFM(nn.Module):
    def __init__(self, in_channels):
        super(JAFFM, self).__init__()
        in_planes = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, in_planes, 1, bias=False),
            nn.ReLU(True)
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels, in_planes, 1, bias=False),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )


        self.spatial_net = nn.Sequential(
            nn.Conv2d(2, 64, 3,dilation=2 ,padding=2,bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, dilation=4, padding=4,bias=False),
            nn.Sigmoid()
        )

        self.pro = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3,dilation=2, padding=2, bias=False, groups=in_planes),

        )
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))
        # self.se=SELayer(in_planes)
        # self.cbam = CBAM(in_planes)

    def forward(self, x, y):

        # channal attetion
        a = self.conv_a(x)
        avg_out = self.conv3(self.conv1(self.avg_pool(a)))
        max_out = self.conv3(self.conv2(self.max_pool(a)))
        c_out = self.sigmoid(avg_out + max_out)


        # spacial attetion
        b = self.conv_b(x)
        avg_out = torch.mean(b, dim=1, keepdim=True)
        max_out, _ = torch.max(b, dim=1, keepdim=True)
        s_in = torch.cat([avg_out, max_out], dim=1)
        s_out = self.spatial_net(s_in)

        # atten_map = c_out
        atten_map = self.pro(c_out*s_out)

        new_y = torch.mul(y, atten_map)

        # atten=self.cbam(x)
        # new_y=atten*y

        return new_y * self.alpha + y

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

class JAFFNet(nn.Module):
    def __init__(self, n_classes=1):
        super(JAFFNet, self).__init__()

        # resnet = models.resnet18(pretrained=True)
        resnet = ResNet18(ResnetBlock)
        ## -------------Encoder--------------

        self.inconv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        # stage 1
        self.encoder1 = resnet.layer1
        # stage 2
        self.encoder2 = resnet.layer2
        # stage 3
        self.encoder3 = resnet.layer3
        # stage 4
        self.encoder4 = resnet.layer4
        # stage 5
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )
        self.pool=nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5g
        self.decoder5_g = nn.Sequential(

            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # stage 4g
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # stage 3g
        self.decoder3_g = nn.Sequential(

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # stage 2g
        self.decoder2_g = nn.Sequential(

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)


        ## -------------Refine Module-------------
        #
        self.jaff1 = JAFFM(512)
        self.jaff2 = JAFFM(512)
        self.jaff3 = JAFFM(256)
        self.jaff4 = JAFFM(128)

        self.bridge = DRF(512)

    def forward(self, x):
        hx = x
        ## -------------Encoder-------------
        hx = self.inconv(hx)

        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)

        h5 = self.encoder5(h4)
        #
        bg=self.bridge(self.pool(h5))

        bgx =  self.upscore2(bg)

        ## -------------Decoder5-------------
        hd5 = self.decoder5_g(torch.cat((bgx, self.jaff1(bgx, h5)), 1))  # 1024-512
        hx5 =  self.upscore2(hd5)

        # -------------Decoder4-------------
        hd4 = self.decoder4_g(torch.cat((hx5, self.jaff2(hx5, h4)), 1))  # 1024->256
        hx4 =  self.upscore2(hd4)

        ## -------------Decoder3-------------
        hd3 = self.decoder3_g(torch.cat((hx4, self.jaff3(hx4, h3)), 1))
        hx3 = self.upscore2(hd3)

        ## -------------Decoder2-------------
        hd2 = self.decoder2_g(torch.cat((hx3, self.jaff4(hx3, h2)), 1))  # 256->64

        out_b = self.outconv6(bgx)
        out_b = self.upscore5(out_b)  # 16->256
        #
        #
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256
        # return F.sigmoid(d2)
        return  F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5),F.sigmoid(out_b)


if __name__ == '__main__':
    # input = torch.randn((1, 3, 224, 224))
    # net = JAFFNet()
    # output = net(input)[0]
    # print(output.shape)
    pytorch_model = JAFFNet()
    pytorch_model.cuda()
    pytorch_weights_dict = pytorch_model.state_dict()
    param_torch = pytorch_weights_dict.keys()
    param_torch_lst = pd.DataFrame(param_torch)
    param_torch_lst.to_csv('param_torch.csv')


