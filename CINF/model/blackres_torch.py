import torch
import torch.nn as nn
from einops import rearrange
from resnet import computeNet
from myattention1 import BlackholeBlock, SwinTransformerBlock, PatchMerging
import torch.nn.functional as F
import pandas as pd


def init_weight(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight,gain=1)
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight,gain=1)


class Segformer(nn.Module):
    def __init__(
            self,
            *,
            dims=(96, 192, 384, 768),
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=2,
            channels=3,
            decoder_dim=256,
            num_classes=2
    ):
        super().__init__()

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_dim // 2, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        fused = [to_fused(output) for output, to_fused in zip(x, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        out = self.to_segmentation(fused)
        return out

class FuseBlock(nn.Module):
    def __init__(self, channel, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=dim, kernel_size=1, stride=1)
        self.layernorm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim*2, dim)

    def forward(self, cnnx, transformerx):
        x = self.conv1(cnnx)
        x = rearrange(x, 'b c w h -> b (w h) c')
        x = torch.concat((x, transformerx), dim=-1)
        x = self.linear(x)

        return x


class BlackholeT5Net(nn.Module):
    def __init__(self, backbone: str, num_class=11):
        super().__init__()
        self.resnet = computeNet(backbone)

        #  stage1
        self.pre = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1, stride=1)
        self.stage11 = SwinTransformerBlock(dim=96, input_resolution=[56, 56], num_heads=3, window_size=7, shift_size=0)
        self.stage12 = SwinTransformerBlock(dim=96, input_resolution=[56, 56], num_heads=3, window_size=7, shift_size=4)
        self.per_mer1 = PatchMerging(dim=96, input_resolution=[56, 56])

        #  stage2
        self.fuse2 = FuseBlock(channel=128, dim=192)
        self.stage21 = SwinTransformerBlock(dim=192, input_resolution=[28, 28], num_heads=6, window_size=7,
                                            shift_size=0)
        self.stage22 = SwinTransformerBlock(dim=192, input_resolution=[28, 28], num_heads=6, window_size=7,
                                            shift_size=4)
        self.per_mer2 = PatchMerging(dim=192, input_resolution=[28, 28])

        #  stage3
        self.fuse3 = FuseBlock(channel=256, dim=384)
        self.stage31 = BlackholeBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7, shift_size=0,
                                      top_channel=128, top_sptial=28)
        self.stage32 = SwinTransformerBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7,
                                            shift_size=4)
        self.stage33 = BlackholeBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7, shift_size=0,
                                      top_channel=128, top_sptial=28)
        self.stage34 = SwinTransformerBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7,
                                            shift_size=4)
        self.stage35 = BlackholeBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7, shift_size=0,
                                      top_channel=128, top_sptial=28)
        self.stage36 = SwinTransformerBlock(dim=384, input_resolution=[14, 14], num_heads=12, window_size=7,
                                            shift_size=4)
        self.per_mer3 = PatchMerging(dim=384, input_resolution=[14, 14])

        #  stage4
        self.fuse4 = FuseBlock(channel=512, dim=768)
        self.stage41 = BlackholeBlock(dim=768, input_resolution=[7, 7], num_heads=24, window_size=7, shift_size=0,
                                      top_channel=144, top_sptial=28)
        self.stage42 = SwinTransformerBlock(dim=768, input_resolution=[7, 7], num_heads=24, window_size=7, shift_size=4)

        #  last head
        # self.lasthead = nn.Sequential(
        #     nn.LayerNorm(7 * 7),
        #     nn.Linear(7 * 7, num_class),
        #     nn.Softmax(dim=1)
        # )
        #
        # #  seg
        # self.seg_head = Segformer()
        self.post1 = nn.LayerNorm(96)
        self.post2 = nn.LayerNorm(192)
        self.post3 = nn.LayerNorm(384)
        self.post4 = nn.LayerNorm(768)

        self.stage11.apply(init_weight)
        self.stage12.apply(init_weight)
        self.stage21.apply(init_weight)
        self.stage22.apply(init_weight)
        self.stage31.apply(init_weight)
        self.stage32.apply(init_weight)
        self.stage33.apply(init_weight)
        self.stage34.apply(init_weight)
        self.stage35.apply(init_weight)
        self.stage36.apply(init_weight)
        self.stage41.apply(init_weight)
        self.stage42.apply(init_weight)

        self.pre.apply(init_weight)
        self.per_mer1.apply(init_weight)
        self.fuse2.apply(init_weight)
        self.per_mer2.apply(init_weight)
        self.fuse3.apply(init_weight)
        self.per_mer3.apply(init_weight)
        self.fuse4.apply(init_weight)

        # self.lasthead.apply(init_weight)
        #
        # self.seg_head.apply(init_weight)

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)

        x1 = self.pre(x1)
        x1 = rearrange(x1, 'b c w h -> b (w h) c')
        x1 = self.stage11(x1)
        x1 = self.stage12(x1)
        xx1 = self.post1(x1)
        x1 = self.per_mer1(x1)

        x2 = self.fuse2(x2, x1)
        x2 = self.stage21(x2)
        x2 = self.stage22(x2)
        xx2 = self.post2(x2)
        x2 = self.per_mer2(x2)

        x3 = self.fuse3(x3, x2)
        x3 = self.stage31(x3)
        x3 = self.stage32(x3)
        x3 = self.stage33(x3)
        x3 = self.stage34(x3)
        x3 = self.stage35(x3)
        x3 = self.stage36(x3)
        xx3 = self.post3(x3)
        x3 = self.per_mer3(x3)

        x4 = self.fuse4(x4, x3)
        x4 = self.stage41(x4)
        x4 = self.stage42(x4)
        xx4 = self.post4(x4)

        # x4 = x4.mean(dim=-1)
        # x4 = self.lasthead(x4)

        xx4 = rearrange(xx4, 'b (w h) c -> b c w h', w=7, h=7)
        xx3 = rearrange(xx3, 'b (w h) c -> b c w h', w=14, h=14)
        xx2 = rearrange(xx2, 'b (w h) c -> b c w h', w=28, h=28)
        xx1 = rearrange(xx1, 'b (w h) c -> b c w h', w=56, h=56)
        #
        # output_se = self.seg_head()

        return [xx1, xx2, xx3, xx4]

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        # x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
        #        * self.conv_upsample3(self.upsample(x2)) * x3
        x2_1 = x2
        x3_1 = x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

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

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class swin_polyp(nn.Module):
    def __init__(self, num_classes, channel=32):
        super().__init__()
        # self.backbone = BlackholeT4Net('resnet18')
        # embed_dim = config.MODEL.SWIN.EMBED_DIM
        # dim = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]

        self.Translayer2_0 = BasicConv2d(96, channel, 1)
        self.Translayer2_1 = BasicConv2d(192, channel, 1)
        self.Translayer3_1 = BasicConv2d(384, channel, 1)
        self.Translayer4_1 = BasicConv2d(768, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(96)
        self.sa = SpatialAttention()
        self.SAM = SAM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, num_classes, 1)
        self.out_CFM = nn.Conv2d(channel, num_classes, 1)

    def forward(self, x):
        # feature = self.backbone(x)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        # CIM
        x1 = self.ca(x1) * x1  # channel attention
        cim_feature = self.sa(x1) * x1  # spatial attention

        # CFM
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear', align_corners=True)
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear', align_corners=True)

        outputs = []
        outputs.append(prediction1_8)
        outputs.append(prediction2_8)

        return outputs

class Bt4pytlypup(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = BlackholeT5Net('resnet18')
        self.sehead = swin_polyp(num_classes=6)
    def forward(self, x):
        x = self.net(x)
        out = self.sehead(x)
        return out


if __name__ =='__main__':
    bt = Bt4pytlypup()
    inputs = torch.rand((1, 3, 224, 224))
    outputs = bt(inputs)
    print(outputs[0].size())
    print(outputs[1].size())

    pytorch_model = Bt4pytlypup()
    pytorch_model.cuda()
    # pytorch_model.load_state_dict(torch.load("E:\yanfeng\save\dagm\jaffnet.pth"))
    pytorch_weights_dict = pytorch_model.state_dict()
    param_torch = pytorch_weights_dict.keys()
    param_torch_lst = pd.DataFrame(param_torch)
    param_torch_lst.to_csv('cinf_param_torch.csv')
