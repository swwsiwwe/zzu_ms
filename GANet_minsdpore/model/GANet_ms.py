import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter


class convbnrelu(nn.Cell):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, padding=p,pad_mode='pad', dilation=d, group=g, has_bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.SequentialCell(*conv)

    def construct(self, x):
        return self.conv(x)
class DSSA(nn.Cell):
    # attention
    def __init__(self, in_channels):
        super().__init__()

        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, in_channels, 1)

        self.softmax = nn.Softmax(axis=1)  # 初始化一个Softmax操作
        self.L21=ops.L2Normalize(axis=-1)
        self.L22=ops.L2Normalize(axis=-1)

        self.scale = Parameter(ops.ones(1,mindspore.float32), requires_grad=True)

    def construct(self, x):
        b, c, h, w = x.shape
        n = h * w

        key = self.conv_b(x).view(b * c, 1, n)
        query = self.conv_c(x).view(b * c, 1, n)
        value = self.conv_d(x)

        key=self.L21(key)

        query=self.L22(query)


        key = key.permute(0, 2, 1)
        kq = ops.bmm(query, key).view(b, c, 1, 1)

        atten = self.softmax(kq * self.scale)
        feat_e = atten * value

        feat_e = self.conv_e(feat_e)

        return feat_e


class MSA(nn.Cell):
    # attention
    def __init__(self, in_channels, num_heads=8):
        super().__init__()

        head_dim = in_channels // num_heads
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, in_channels, 1)

        self.softmax = nn.Softmax(axis=-1)  # 初始化一个Softmax操作

        self.scale = head_dim ** -0.5

    def construct(self, x):
        b, c, h, w = x.shape
        n = h * w

        key = self.conv_b(x).view(b * 8, c // 8, n)
        query = self.conv_c(x).view(b * 8, c // 8, n)
        value = self.conv_d(x).view(b * 8, c // 8, n)

        key = key.permute(0, 2, 1)
        kq = ops.bmm(key, query)

        atten = self.softmax(kq * self.scale)
        feat_e = ops.bmm(value, atten.permute(0, 2, 1)).view(b, c, h, w)
        feat_e = self.conv_e(feat_e)

        return feat_e


class Hydra(nn.Cell):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.to_q = nn.Dense(dim, dim)
        self.to_k = nn.Dense(dim, dim)
        self.to_v = nn.Dense(dim, dim)
        self.project = nn.Dense(dim, dim)

    def construct(self, x):
        b, c, h, w = x.shape
        n = h * w
        x = x.view(b, c, h * w).permute(0, 2, 1)
        Q = self.to_q(x).view(b, n, self.dim)
        K = self.to_k(x).view(b, n, self.dim)
        V = self.to_v(x).view(b, n, self.dim)

        K=ops.L2Normalize(K, axis=1)
        Q=ops.L2Normalize(Q, axis=1)


        weights = ops.mul(K, V).sum(axis=1, keepdims=True)
        # Q_sig = torch.sigmoid(Q)
        Yt = ops.mul(Q, weights)

        Yt = Yt.view(b, n, self.dim)
        Yt = self.project(Yt)
        Yt = Yt.permute(0, 2, 1)

        return Yt.view(b, c, h, w)


class PolyNL(nn.Cell):
    # attention
    def __init__(self, in_channels):
        super().__init__()

        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)


        self.alpha = Parameter(ops.ones(1, mindspore.float32), requires_grad=True)

        self.beta = Parameter(ops.ones(1, mindspore.float32), requires_grad=True)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def construct(self, x):
        b, c, h, w = x.shape
        n = h * w

        # key = self.conv_b(x).view(b*c,1,n)
        # query = self.conv_c(x).view(b*c,1,n)
        # value = self.conv_d(x)

        key = self.conv_b(x).view(b, c, n)
        query = self.conv_c(x).view(b, c, n)
        value = x.view(b, c, n)

        atten = self.gap(key * query) * value

        y = self.conv_d(atten.view(b, c, h, w))

        out = self.alpha * y + self.beta * x

        return out

class LayerNorm(nn.Cell):
    def __init__(self, dim, eps=1e-8):
        super(LayerNorm, self).__init__()

        self.gamma = Parameter(ops.ones([1, dim, 1, 1], mindspore.float32), requires_grad=True)
        self.beta = Parameter(ops.zeros([1, dim, 1, 1], mindspore.float32), requires_grad=True)
        self.eps = eps

    def construct(self, x):
        mean = x.mean(axis=1, keep_dims=True)
        var = x.var(axis=1, keepdims=True)
        x = (x - mean) / (ops.sqrt(var + self.eps))
        return x * self.gamma + self.beta
import math
def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = ops.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out
class TransformerBlock(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = nn.SequentialCell(
            nn.Conv2d(dim, dim * 4, 1, has_bias=False),
            nn.GELU(approximate=False),
            nn.Conv2d(dim * 4, dim * 4, 3, padding=1, stride=1, pad_mode='pad',group=dim * 4, has_bias=False),
            nn.GELU(approximate=False),
            nn.Conv2d(dim * 4, dim, 1, has_bias=False),
        )

        self.mhsa = DSSA(dim)

    def construct(self, x):

        x = x + self.mhsa(self.norm1(x))

        x = self.ffn(self.norm2(x)) + x

        return x


class RAM(nn.Cell):
    def __init__(self, channel1, channel2,ead_num=4):
        super(RAM, self).__init__()


        self.k = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.q1 = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.q2 = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.pro = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=False,bn=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(axis=-1)

        self.scale = Parameter(ops.ones(1),requires_grad=True)

        self.head_num = 4


    def construct(self, f_h, f_l, f_g):
        b, c, h, w = f_l.shape

        k = self.k(self.avg_pool(f_l)).view(b * self.head_num, c // self.head_num, 1)
        q1 = self.q1(self.avg_pool(f_g)).view(b * self.head_num, c // self.head_num, 1)
        q2 = self.q2(self.avg_pool(f_h)).view(b * self.head_num, c // self.head_num, 1)
        v = f_l.view(b * self.head_num, c // self.head_num, h * w)

        k = k.permute(0, 2, 1)
        atten1 = ops.bmm(q1, k)
        atten2 = ops.bmm(q2, k)
        atten = atten1 + atten2
        atten = self.softmax(atten * self.scale)
        out = ops.bmm(atten, v).view(b, c, h, w)
        out = self.pro(out)
        w_f_l = f_l + out

        return w_f_l


class PyramidPoolAgg(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, inputs):
        B, C, H, W = inputs[-1].shape

        return ops.cat([ops.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], axis=1)


class FAM(nn.Cell):
    def __init__(self, channel1, channel2):
        super(FAM, self).__init__()

        self.conv1 = convbnrelu(channel1, channel1, k=1, s=1, p=0, bn=False, relu=False)
        self.conv2 = convbnrelu(channel1, channel1, k=1, s=1, p=0, bn=False, relu=False)
        self.conv3 = convbnrelu(channel2, channel1, k=1, s=1, p=0, bn=False,relu=False)
        self.ram = RAM(channel1,channel2)
        self.conv4 = convbnrelu(channel1,channel1, k=1, s=1, p=0,bn=False,relu=False)
        self.conv5 = convbnrelu(channel2,channel1, k=1, s=1, p=0,bn=False,relu=False)

    def construct(self, f_h, f_l, f_g):
        b, c, h, w = f_l.shape
        #
        f_h = self.conv1(f_h)
        f_l = self.conv2(f_l)
        w_f_l = self.ram(f_h, f_l,self.conv3(f_g))

        f_h = ops.interpolate(f_h, size=(h, w), mode='bilinear', align_corners=True)
        f_h = self.conv4(f_h)

        f_g = self.conv5(f_g)
        f_g = ops.interpolate(f_g, size=(h, w), mode='bilinear', align_corners=True)


        fused = w_f_l + f_h + f_g

        return fused


class GANet(nn.Cell):
    def __init__(self):
        super(GANet, self).__init__()

        self.context_path = VAMM_backbone()
        #mindspore这里没有预训练的权重的backbone了，所以把预训练模型加载代码删去
        #self.context_path.load_state_dict(torch.load("D:\yanfeng\BASNet-master\models\models\SAMNet_backbone_pretrain.pth"))

        self.transformer = nn.SequentialCell(
            DSConv3x3(128, 128, stride=1),
            TransformerBlock(128),
            TransformerBlock(128),
        )

        self.prepare = convbnrelu(128,128, k=1, s=1, p=0, relu=False,bn=False)

        self.fam2 = FAM(96, 128)
        self.fam3 = FAM(64, 128)
        self.fam4 = FAM(32, 128)
        self.fam5 = FAM(16, 128)

        self.fuse = nn.CellList([
            DSConv3x3(128, 96, dilation=1),
            DSConv3x3(96, 64, dilation=2),
            DSConv3x3(64, 32, dilation=2),
            DSConv3x3(32, 16, dilation=2),
            DSConv3x3(16, 16, dilation=2)
        ])

        self.heads = nn.CellList([
            SalHead(in_channel=96),
            SalHead(in_channel=64),
            SalHead(in_channel=32),
            SalHead(in_channel=16),
            SalHead(in_channel=16),
            SalHead(in_channel=128)
        ])

    def construct(self, x):  # (3, 1)
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5 = self.context_path(x)
        # #
        ct_stage6 = self.transformer(ct_stage5)  # (128, 1/32)

        fused_stage1 = self.fuse[0](self.prepare(ct_stage5))  # (96, 1/32)


        fused_stage2 = self.fuse[1](self.fam2(fused_stage1, ct_stage4, ct_stage6))  # (64, 1/16)


        fused_stage3 = self.fuse[2](self.fam3(fused_stage2, ct_stage3, ct_stage6))  # (32, 1/8)


        fused_stage4 = self.fuse[3](self.fam4(fused_stage3, ct_stage2, ct_stage6))  # (16, 1/4)

        #
        fused_stage5 = self.fuse[4](self.fam5(fused_stage4, ct_stage1, ct_stage6))  # (16, 1/2)



        # fused_stage5 = self.fuse[4](ct_stage1 + refined4)  # (16, 1/2)

        output_side0 = interpolate(self.heads[-1](ct_stage6), x.shape[2:])
        output_side1 = interpolate(self.heads[0](fused_stage1), x.shape[2:])
        output_side2 = interpolate(self.heads[1](fused_stage2), x.shape[2:])
        output_side3 = interpolate(self.heads[2](fused_stage3), x.shape[2:])
        output_side4 = interpolate(self.heads[3](fused_stage4), x.shape[2:])
        output_main = self.heads[4](fused_stage5)
        #return output_main
        return output_main, output_side1, output_side2, output_side3, output_side4, output_side0

interpolate = lambda x, size: ops.interpolate(x, size=size, mode='bilinear', align_corners=True)


class DSConv3x3(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.SequentialCell(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def construct(self, x):
        return self.conv(x)




class SalHead(nn.Cell):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def construct(self, x):
        return self.conv(x)


class VAMM_backbone(nn.Cell):
    def __init__(self):
        super(VAMM_backbone, self).__init__()
        self.layer1 = nn.SequentialCell(
            convbnrelu(3, 16, k=3, s=1, p=1),
            VAMM(16, dilation_level=[1, 2, 3])
        )
        self.layer2 = nn.SequentialCell(
            DSConv3x3(16, 32, stride=2),
            VAMM(32, dilation_level=[1, 2, 3])
        )
        self.layer3 = nn.SequentialCell(
            DSConv3x3(32, 64, stride=2),
            VAMM(64, dilation_level=[1, 2, 3]),
            VAMM(64, dilation_level=[1, 2, 3]),
            VAMM(64, dilation_level=[1, 2, 3])
        )
        self.layer4 = nn.SequentialCell(
            DSConv3x3(64, 96, stride=2),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3])
        )
        self.layer5 = nn.SequentialCell(
            DSConv3x3(96, 128, stride=2),
            VAMM(128, dilation_level=[1, 2]),
            VAMM(128, dilation_level=[1, 2]),
            VAMM(128, dilation_level=[1, 2])
        )

    def construct(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out1, out2, out3, out4, out5


class VAMM(nn.Cell):
    def __init__(self, channel, dilation_level=None, reduce_factor=4):
        super(VAMM, self).__init__()
        if dilation_level is None:
            dilation_level = [1, 2, 4, 8]
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.CellList([
            DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
        ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, padding=0, has_bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.SequentialCell(
            convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
            DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
            DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
            nn.Conv2d(channel // reduce_factor, 1, 1, 1, padding=0, has_bias=False)
        )

    def construct(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = ops.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = ops.softmax(f, axis=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)])) + x


class PyramidPooling(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel * 2, out_channel, k=1, s=1, p=0)

    def construct(self, x):
        size = x.shape[2:]
        feat1 = interpolate(self.conv1(ops.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(ops.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(ops.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(ops.adaptive_avg_pool2d(x, 6)), size)
        x = ops.cat([x, feat1, feat2, feat3, feat4], axis=1)
        x = self.out(x)

        return x

if __name__ == '__main__':
    import mindspore.numpy as np
    input = np.randn((1, 3, 224, 224))
    net = GANet()
    output = net(input)
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, output[4].shape,output[5].shape)
