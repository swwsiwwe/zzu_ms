import math
import pandas as pd
import numpy as np

from mindspore import Parameter
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Initializer
from mindspore.ops import composite as C


mindspore.set_context(device_target='Ascend', device_id=0)


BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


class Patten(nn.Cell):
    def __init__(self, in_dim, n):
        super(Patten, self).__init__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // n, kernel_size=1, pad_mode='pad', has_bias=True)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // n, kernel_size=1, pad_mode='pad', has_bias=True)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, pad_mode='pad', has_bias=True)
        self.gamma = Parameter(ops.zeros(1))

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        m_batchsize, C, h, w = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, w * h)
        energy = ops.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, w * h)

        out = ops.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, h, w)

        out = self.gamma * out + x
        return out


class Catten(nn.Cell):
    def __init__(self, in_dim):
        super(Catten, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(ops.zeros(1))
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = ops.bmm(proj_query, proj_key)
        energy_new = ops.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = ops.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

# class BNorm_init(nn.BatchNorm2d):
#     def reset_parameters(self):
#         ops.uniform(self.weight, 0, 1)
#         ops.zeros_(self.bias)
def _calculate_fan_in_and_fan_out(arr):
    # 计算fan_in和fan_out。fan_in是 `arr` 中输入单元的数量，fan_out是 `arr` 中输出单元的数量。
    shape = arr.shape
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("'fan_in' and 'fan_out' can not be computed for arr with fewer than"
                         " 2 dimensions, but got dimensions {}.".format(dimensions))
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for i in range(2, dimensions):
            receptive_field_size *= shape[i]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class XavierNormal(Initializer):
    def __init__(self, gain=1):
        super().__init__()
        # 配置初始化所需要的参数
        self.gain = gain

    def _initialize(self, arr): # arr为需要初始化的Tensor
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr) # 计算fan_in, fan_out值

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out)) # 根据公式计算std值
        data = np.random.normal(0, std, arr.shape) # 使用numpy构造初始化好的ndarray

        arr[:] = data[:] # 将初始化好的ndarray赋值到arr

# class Conv2d_init(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, has_bias=True,
#                  padding_mode="pad"):
#         super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                           has_bias, padding_mode)
#
#     def reset_parameters(self):
#         init.xavier_normal_(self.weight)
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)


def _conv_block(in_chanels, out_chanels, kernel_size, padding, dilation=1):
    return nn.SequentialCell(nn.Conv2d(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, dilation=dilation, padding=padding, pad_mode="pad", has_bias=False
                                       , weight_init=XavierNormal()),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())


class FeatureNorm(nn.Cell):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = Parameter(ops.ones(self.shape, dtype=mindspore.float32))
        self.bias = Parameter(
            ops.zeros(self.shape, dtype=mindspore.float32)) if include_bias else Parameter(
            ops.zeros(self.shape, dtype=mindspore.float32))

        self.eps = eps

    def construct(self, features):
        f_std = ops.std(features, axis=self.reduce_dims, keepdims=True)
        f_mean = ops.mean(features, axis=self.reduce_dims, keep_dims=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


class SegDecNet_tf(nn.Cell):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNet_tf, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        # self.volume = Dilation_vgg(self.input_channels)

        self.patten = Patten(1024, 4)
        self.catten = Catten(1024)
        self.beta = mindspore.Tensor([0.5])
        self.gamma = mindspore.Tensor([0.5])

        self.volume = nn.SequentialCell(_conv_block(self.input_channels, 32, 5, 2),
                                    # _conv_block(32, 32, 5, 2), # Has been accidentally left out and remained the same since then
                                    nn.MaxPool2d(2,2,pad_mode='pad'),
                                    _conv_block(32, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2,2,pad_mode='pad'),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2, 2,pad_mode='pad'),
                                    _conv_block(64, 1024, 15, 7))
        # self.RFB = BasicRFB_a(1024,1024)
        self.seg_mask_1 = nn.SequentialCell(
            # Conv2d_init(in_channels=1152, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0, pad_mode='pad',has_bias=False, weight_init=XavierNormal()),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        self.seg_mask_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=1152, out_channels=1, kernel_size=1, padding=0, pad_mode='pad', has_bias=False, weight_init=XavierNormal()),
            # Conv2d_init(in_channels=1024, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))

        self.extractor = nn.SequentialCell(nn.MaxPool2d(2,2,pad_mode='pad'),
                                       _conv_block(in_chanels=1152, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(2,2,pad_mode='pad'),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(2,2,pad_mode='pad'),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))
        self.get_unreverse = nn.SequentialCell(
            _conv_block(in_chanels=256, out_chanels=64, kernel_size=3, dilation=1, padding=1),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=1, padding=1),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=2, padding=2),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=5, padding=5))
        self.get_mask_unreverse = _conv_block(in_chanels=64, out_chanels=1, kernel_size=1, padding=0)
        self.get_mask_reverse = _conv_block(in_chanels=64, out_chanels=1, kernel_size=1, padding=0)

        self.get_reverse_e = _conv_block(in_chanels=1024, out_chanels=256, kernel_size=3, padding=1)
        self.get_unreverse_e = _conv_block(in_chanels=1024, out_chanels=256, kernel_size=3, padding=1)

        self.get_reverse = nn.SequentialCell(
            _conv_block(in_chanels=256, out_chanels=64, kernel_size=3, dilation=1, padding=1),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=1, padding=1),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=2, padding=2),
            _conv_block(in_chanels=64, out_chanels=64, kernel_size=3, dilation=5, padding=5))

        self.global_max_pool_feat = nn.MaxPool2d(32,32,pad_mode='pad')
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32,stride=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(int(self.input_height / 8), int(self.input_width / 8)),stride=(int(self.input_height / 8), int(self.input_width / 8)),pad_mode='pad')
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(int(self.input_height / 8), int(self.input_width / 8)),stride=(int(self.input_height / 8), int(self.input_width / 8)))

        self.fc = nn.Dense(in_channels=64, out_channels=1)

        self.volume_lr_multiplier_layer = GradientMultiplyLayer()
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer()
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer()

        self.extractor_logit_pixel = nn.SequentialCell(
            _conv_block(in_chanels=3, out_chanels=8, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'),
            _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'),
            _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'))

        self.extractor_Feature = nn.SequentialCell(
            _conv_block(in_chanels=1152, out_chanels=8, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'),
            _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'),
            _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2,stride=2,pad_mode='pad'))

        self.device = device

    def set_gradient_multipliers_debug(self, multiplier):
        self.volume_lr_multiplier_mask = ops.ones((1,)) * multiplier
        self.glob_max_lr_multiplier_mask = ops.ones((1,)) * multiplier
        self.glob_avg_lr_multiplier_mask = ops.ones((1,)) * multiplier

    def construct(self, input):
        # grad_all = C.GradOperation(get_all=True)
        self.set_gradient_multipliers_debug(0)

        volume0 = self.volume(input)
        # volume1 = self.RFB(volume0)
        volume = volume0
        # volume1 = self.patten(volume)
        # volume2 = self.catten(volume)
        # volume = volume1+volume2
        # volume = self.volume(input)
        atten = self.seg_mask_1(volume)
        atten_map = nn.Sigmoid()(atten)

        net_reverse = self.get_reverse_e(volume)
        net_reverse = ops.mul(net_reverse, atten_map)
        net_reverse = self.get_reverse(net_reverse)
        net_reverse_mask = self.get_mask_reverse(net_reverse)

        net_unreverse = self.get_unreverse_e(volume)
        net_unreverse = ops.mul(net_unreverse, 1 - atten_map)
        net_unreverse = self.get_unreverse(net_unreverse)
        net_unreverse_mask = self.get_mask_unreverse(net_unreverse)

        # seg_mask = net_unreverse_mask + net_reverse_mask + seg_mask
        seg_mask_feature = ops.cat((volume, net_unreverse, net_reverse), axis=1)
        seg_mask_2 = self.seg_mask_2(seg_mask_feature)
        seg_mask_1 = atten

        seg_mask_continue = ops.cat((atten, net_unreverse_mask, net_reverse_mask), axis=1)

        # test = self.volume_lr_multiplier_layer(seg_mask_feature, self.volume_lr_multiplier_mask)
        # cat = grad_all(self.volume_lr_multiplier_layer)(seg_mask_feature, self.volume_lr_multiplier_mask)[0]
        # print(cat.shape,cat.sum())
        cat = seg_mask_feature
        # cat = self.volume_lr_multiplier_layer(seg_mask_feature, self.volume_lr_multiplier_mask)

        features = self.extractor_Feature(cat)
        global_max_feat = ops.max(ops.max(features, axis=-1, keepdims=True)[0], axis=-2, keepdims=True)[0]
        global_avg_feat = ops.mean(features, axis=(-1, -2), keep_dims=True)

        dec_seg_mask = self.extractor_logit_pixel(seg_mask_continue)
        global_max_seg = ops.max(ops.max(dec_seg_mask, axis=-1, keepdims=True)[0], axis=-2, keepdims=True)[0]
        # test = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        # global_max_seg = grad_all(self.glob_max_lr_multiplier_layer)(global_max_seg, self.glob_max_lr_multiplier_mask)[0]
        # global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = ops.mean(dec_seg_mask, axis=(-1, -2), keep_dims=True)
        # test = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)
        # global_avg_seg = grad_all(self.glob_avg_lr_multiplier_layer)(global_avg_seg, self.glob_avg_lr_multiplier_mask)[0]
        # global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        self.beta = self.beta
        self.gamma = self.gamma
        global_max_feat = global_max_feat + self.beta * global_max_seg
        global_avg_feat = global_avg_feat + self.gamma * global_avg_seg
        global_max_feat = global_max_feat + 0.5 * global_max_seg
        global_avg_feat = global_avg_feat + 0.5 * global_avg_seg

        fc_in = ops.cat([global_max_feat, global_avg_feat], axis=1)
        # print(fc_in.shape)
        # print(fc_in.size(0))
        sp = fc_in.shape
        fc_in = fc_in.reshape(sp[0], -1)
        prediction = self.fc(fc_in)

        return prediction, seg_mask_1, seg_mask_2

# ??
# class GradientMultiplyLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, mask_bw):
#         ctx.save_for_backward(mask_bw)
#         return input
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         mask_bw, = ctx.saved_tensors
#         return grad_output.mul(mask_bw), None

class GradientMultiplyLayer(nn.Cell):
    def __init__(self):
        super(GradientMultiplyLayer, self).__init__()

    def construct(self, input, mask_bw):
        """ construct """
        return input

    def bprop(self, input, mask_bw, output, grad_output):

        return (grad_output.mul(mask_bw.astype(grad_output.dtype)), grad_output.mul(mask_bw.astype(grad_output.dtype)))


class BasicConv(nn.Cell):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, group=groups, has_bias=bias, pad_mode='pad')
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Cell):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, group=in_planes, has_bias=bias, pad_mode='pad')
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Cell):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride,
                      padding=(1, 0)),
            BasicSepConv((inter_planes // 2) * 3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=3, stride=stride, padding=1),
            BasicSepConv((inter_planes // 2) * 3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(3 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU()

    def construct(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = ops.cat((x1, x2), 1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out * self.scale + x
        else:
            short = self.shortcut(x)
            out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Cell):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.SequentialCell(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU()

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


if __name__ == '__main__':
    import numpy as np
    import cv2 as cv
    # import torchvision.transforms as transforms
    from mindspore import Tensor
    img = cv.imread('./20000.png')
    print(img.shape)
    img.resize((232, 640, 3))
    print(img.shape)
    img_tensor = np.array(img, dtype=np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, axes=(2, 0, 1))
    img_tensor = Tensor.from_numpy(img_tensor)
    img_tensor = img_tensor[None, :]
    print(img_tensor.shape)

    net = SegDecNet_tf('cpu', 232, 640, 3)

    # output = net(img)
    # print(output[0].shape, output[1].shape, output[2].shape)
    from c2net.context import prepare

    #初始化导入数据集和预训练模型到容器内
    c2net_context = prepare()

    #获取数据集路径
    kolektorsdd2_path = c2net_context.dataset_path+"/"+"KolektorSDD2"

    #获取预训练模型路径
    _3_loss_model_ch5s_path = c2net_context.pretrain_model_path+"/"+"3_loss_model_ch5s"

    #输出结果必须保存在该目录
    you_should_save_here = c2net_context.output_path
    param_dict = mindspore.load_checkpoint(_3_loss_model_ch5s_path+'/best_state_dict.ckpt')
    mindspore.load_param_into_net(net, param_dict)
    prediction, pred_seg_1, pred_seg = net(img_tensor)
    # print(output[0].shape, output[1].shape, output[2].shape)
    print(pred_seg)
    print("...")
    print(prediction)
    print(nn.Sigmoid()(prediction))