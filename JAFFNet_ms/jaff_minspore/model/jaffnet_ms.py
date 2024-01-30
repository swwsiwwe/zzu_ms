import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Parameter
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,pad_mode='pad', stride=stride,
                     padding=1, has_bias=False)
class BasicBlock(nn.Cell):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def construct(self, x):
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
class ResnetBlock(nn.Cell):
    def __init__(self, inchannel, outchannel, stride=1) -> None:
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1,pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, pad_mode='pad',has_bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.downsample = nn.SequentialCell()
        if stride != 1 or inchannel != outchannel:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.downsample(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Cell):
    def __init__(self, ResBlock, num_classes=1000) -> None:
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad', ceil_mode=False)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Dense(512, num_classes)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
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
        return nn.SequentialCell(*layers)

class MRF(nn.Cell):
    def __init__(self, in_channels, out_channels, rate=None):
        super(MRF, self).__init__()

        if rate is None:
            rate = [1, 2, 4, 8]
        inner_channels=in_channels//4

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1= nn.SequentialCell(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[0], padding=rate[0],pad_mode='pad'),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[1], padding=rate[1],pad_mode='pad'),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
        )


        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[2], padding=rate[2],pad_mode='pad'),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            )

        self.pro = nn.SequentialCell(
            nn.Conv2d(inner_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):

        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)


        out= mindspore.ops.cat([feat1,feat2,feat3],axis=1)

        return self.pro(out)

class DRF(nn.Cell):
    def __init__(self, in_channels):
        super(DRF, self).__init__()
        out_channels =512
        self.mrf1 = MRF(in_channels,out_channels)
        self.mrf2 = MRF(in_channels,out_channels)
        self.mrf3 = MRF(in_channels,out_channels)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels, 1,pad_mode='pad'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.relu= nn.ReLU()
        self.pro = nn.SequentialCell(
            nn.Conv2d(in_channels , out_channels, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )



    def construct(self, x):

        size = x.shape[2:]
        feat1 = self.mrf1(x)
        input = x + feat1
        feat2 = self.mrf2(input)
        input = x+feat1+feat2
        feat3 = self.mrf3(input)

        feat4_pre1 = self.gap(x)
        feat4_pre2=self.conv4(feat4_pre1)
        feat4=ops.interpolate(feat4_pre2, size, mode="bilinear", align_corners=True)

        feat = x+feat1+feat2+feat3+feat4
        out = self.pro(feat)
        return out


class ChannelAttentionModule(nn.Cell):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.SequentialCell(
            nn.Conv2d(channel, channel // ratio, 1, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, has_bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return avgout + maxout


class SpatialAttentionModule(nn.Cell):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avgout = ops.mean(x, axis=1, keep_dims=True)
        maxout, _ = ops.max(x, axis=1, keepdims=True)
        out=ops.cat([avgout,maxout],axis=1)
        out=self.conv2d(out)
        return out


class CBAM(nn.Cell):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        c_out = self.channel_attention(x)
        s_out = self.spatial_attention(x)
        return self.sigmoid(c_out+s_out)

class Flatten(nn.Cell):
    def construct(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Cell):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.SequentialCell()
        self.gate_c.insert_child_to_cell('flatten', Flatten())

        gate_channels = [gate_channel]  # eg 64
        gate_channels += [gate_channel // reduction_ratio] * num_layers  # eg 4
        gate_channels += [gate_channel]  # 64
        # gate_channels: [64, 4, 4]

        for i in range(len(gate_channels) - 2):
            self.gate_c.insert_child_to_cell(
                'gate_c_fc_%d' % i,
                nn.Dense(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.insert_child_to_cell('gate_c_bn_%d' % (i + 1),
                                   nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.insert_child_to_cell('gate_c_relu_%d' % (i + 1), nn.ReLU())

        self.gate_c.insert_child_to_cell('gate_c_fc_final',
                               nn.Dense(gate_channels[-2], gate_channels[-1]))

    def construct(self, x):
        avg_pool = ops.avg_pool2d(x, x.shape(2), stride=x.shape(2))

        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)

class SpatialGate(nn.Cell):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=16,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.SequentialCell()

        self.gate_s.insert_child_to_cell(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,
                      gate_channel // reduction_ratio,
                      kernel_size=1))
        self.gate_s.insert_child_to_cell('gate_s_bn_reduce0',
                               nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.insert_child_to_cell('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.insert_child_to_cell(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(gate_channel // reduction_ratio,
                          gate_channel // reduction_ratio,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.insert_child_to_cell(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.insert_child_to_cell('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.insert_child_to_cell(
            'gate_s_conv_final',
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def construct(self, x):
        return self.gate_s(x).expand_as(x)
class BAM(nn.Cell):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def construct(self, x):
        att =ops.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att
class SELayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class JAFFM(nn.Cell):
    def __init__(self, in_channels):
        super(JAFFM, self).__init__()
        in_planes = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_a = nn.SequentialCell(
            nn.Conv2d(in_channels, in_planes, 1, has_bias=False),
            nn.ReLU()
        )
        self.conv_b = nn.SequentialCell(
            nn.Conv2d(in_channels, in_planes, 1, has_bias=False),
            nn.ReLU()
        )

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_planes, in_planes // 16, 1, has_bias=False),
            nn.ReLU(),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_planes, in_planes // 16, 1, has_bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_planes // 16, in_planes, 1, has_bias=False),
        )


        self.spatial_net = nn.SequentialCell(
            nn.Conv2d(2, 64, 3,dilation=2 ,padding=2,has_bias=False,pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, dilation=4, padding=4,has_bias=False,pad_mode='pad'),
            nn.Sigmoid()
        )

        self.pro = nn.SequentialCell(
            nn.Conv2d(in_planes, in_planes, 3,dilation=2, padding=2, has_bias=False, group=in_planes,pad_mode='pad'),

        )
        self.sigmoid = nn.Sigmoid()

        self.alpha = Parameter(ops.zeros(1, mindspore.float32), requires_grad=True)
        #x = Tensor(0, mindspore.float32)
        #self.alpha = Parameter(x, requires_grad=True)

    def construct(self, x, y):

        # channal attetion
        a = self.conv_a(x)
        avg_out = self.conv3(self.conv1(self.avg_pool(a)))
        max_out = self.conv3(self.conv2(self.max_pool(a)))
        c_out = self.sigmoid(avg_out + max_out)


        # spacial attetion
        b = self.conv_b(x)
        avg_out = ops.mean(b, axis=1, keep_dims=True)
        max_out, _ = ops.max(b, axis=1, keepdims=True)
        s_in = ops.cat([avg_out, max_out], axis=1)
        s_out = self.spatial_net(s_in)

        # atten_map = c_out
        atten_map = self.pro(c_out*s_out)

        new_y = ops.mul(y, atten_map)

        # atten=self.cbam(x)
        # new_y=atten*y

        return new_y * self.alpha + y

class BasicConv(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class convbnrelu(nn.Cell):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, padding=p, dilation=d, groups=g, has_bias=bias, pad_mode='pad')]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.SequentialCell(*conv)

    def construct(self, x):
        return self.conv(x)

interpolate = lambda x, size: ops.interpolate(x, size=size, mode='bilinear', align_corners=True)
class PyramidPooling(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def construct(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(ops.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(ops.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(ops.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(ops.adaptive_avg_pool2d(x, 6)), size)
        x = ops.cat([x, feat1, feat2, feat3, feat4], axis=1)
        x = self.out(x)

        return x

class JAFFNet(nn.Cell):
    def __init__(self, n_classes=1):
        super(JAFFNet, self).__init__()
        resnet = ResNet18(ResnetBlock)
        ## -------------Encoder--------------

        self.inconv = nn.SequentialCell(
            nn.Conv2d(3, 64, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
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
        self.encoder5 = nn.SequentialCell(
            nn.MaxPool2d(2, 2, ceil_mode=True,pad_mode='pad'),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )
        self.pool=nn.MaxPool2d(2, 2, ceil_mode=True,pad_mode='pad')

        # stage 5g
        self.decoder5_g = nn.SequentialCell(

            nn.Conv2d(1024, 512, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # stage 4g
        self.decoder4_g = nn.SequentialCell(
            nn.Conv2d(1024, 512, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # stage 3g
        self.decoder3_g = nn.SequentialCell(

            nn.Conv2d(512, 256, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # stage 2g
        self.decoder2_g = nn.SequentialCell(
            nn.Conv2d(256, 128, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        ## -------------Bilinear Upsampling--------------

        self.upscore6 = nn.ResizeBilinear()
        self.upscore5 = nn.ResizeBilinear()
        self.upscore4 = nn.ResizeBilinear()
        self.upscore3 = nn.ResizeBilinear()
        self.upscore2 = nn.ResizeBilinear()
        self.upscore1 = nn.ResizeBilinear()

        '''

        '''
        ## -------------Side Output--------------
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1,pad_mode='pad')
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1,pad_mode='pad')
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1,pad_mode='pad')
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1,pad_mode='pad')
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1,pad_mode='pad')


        ## -------------Refine Module-------------
        #
        self.jaff1 = JAFFM(512)
        self.jaff2 = JAFFM(512)
        self.jaff3 = JAFFM(256)
        self.jaff4 = JAFFM(128)

        self.bridge = DRF(512)
        self.sigmoid=ops.Sigmoid()

    def construct(self, x):
        hx = x
        ## -------------Encoder-------------
        hx = self.inconv(hx)

        h1 = self.encoder1(hx)

        h2 = self.encoder2(h1)

        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)

        h5 = self.encoder5(h4)
        #
        bg= self.bridge(self.pool(h5))

        bgx =  self.upscore2(bg,scale_factor=2)

        ## -------------Decoder5-------------
        hd5 = self.decoder5_g(ops.cat((bgx, self.jaff1(bgx, h5)),1))  # 1024-512
        hx5 =  self.upscore2(hd5,scale_factor=2)

        # -------------Decoder4-------------
        hd4 = self.decoder4_g(ops.cat((hx5, self.jaff2(hx5, h4)), 1))  # 1024->256
        hx4 =  self.upscore2(hd4,scale_factor=2)

        ## -------------Decoder3-------------
        hd3 = self.decoder3_g(ops.cat((hx4, self.jaff3(hx4, h3)), 1))
        hx3 = self.upscore2(hd3,scale_factor=2)

        ## -------------Decoder2-------------
        hd2 = self.decoder2_g(ops.cat((hx3, self.jaff4(hx3, h2)), 1))  # 256->64

        out_b = self.outconv6(bgx)
        out_b = self.upscore5(out_b,scale_factor=16)  # 16->256
        #
        #
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5,scale_factor=16)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4,scale_factor=8)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3,scale_factor=4)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2,scale_factor=2)  # 128->256
        # return F.sigmoid(d2)

        # out=self.sigmoid(d2)
        # return  ops.sigmoid(d2)
        return ops.sigmoid(d2), ops.sigmoid(d3), ops.sigmoid(d4), ops.sigmoid(d5),ops.sigmoid(out_b)







if __name__ == '__main__':
    import mindspore.numpy as np
    input = np.randn((1, 3, 224, 224))
    net = JAFFNet()
    output = net(input)[0]
    print(output.shape)