import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import pandas as pd
from mindspore import Parameter
from mindspore.common.initializer import initializer, XavierNormal


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


def get_backbone(name:str) -> nn.Module:
    if name == "resnet18":
        return ResNet18(ResnetBlock)
    else:
        print("no Model")


def init_weight(m):
    if type(m) == nn.Conv2d:
        m.weight = Parameter(initializer('xavier_uniform', m.weight, mindspore.float32))
    if type(m) == nn.Dense:
        m.weight = Parameter(initializer('xavier_uniform', m.weight, mindspore.float32))


class computeNet(nn.Cell):
    def __init__(self, backbone: str, num_class=7, f_resnet=True):
        super().__init__()
        self.backbone = get_backbone(backbone)

        self.start_layer = nn.SequentialCell(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )

        self.r2 = self.backbone.layer1
        self.r3 = self.backbone.layer2
        self.r4 = self.backbone.layer3
        self.r5 = self.backbone.layer4

        if f_resnet is True:
            for p in self.parameters():
                p.requires_grad = False

        self.upr5 = nn.Sequential(
            nn.Conv2dTranspose(in_channels=512, out_channels=512, kernel_size=2, stride=2, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1,has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fuser5r4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,stride=1,has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upr4 = nn.Sequential(
            nn.Conv2dTranspose(in_channels=256, out_channels=256, kernel_size=2, stride=2, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fuser4r3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.upr3 = nn.Sequential(
            nn.Conv2dTranspose(in_channels=128, out_channels=128, kernel_size=2, stride=2, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fuser3r2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, has_bias=True, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )


    def construct(self,x):
        x = self.start_layer(x)
        x = self.r2(x)
        x2 = x
        print(x2.size())

        x = self.r3(x)
        x3 = x
        x = self.r4(x)
        x4 = x
        x5 = self.r5(x)
        xx5 = x5

        x5 = self.upr5(x5)
        x4 = ops.concat((x4,x5), axis=1)
        x4 = self.fuser5r4(x4)
        xx4 = x4

        x4 = self.upr4(x4)
        x3 = ops.concat((x3, x4), axis=1)
        x3 = self.fuser4r3(x3)
        xx3 = x3

        x3 = self.upr3(x3)
        x2 = ops.concat((x2, x3), axis=1)
        x2 = self.fuser3r2(x2)
        xx2 = x2

        return xx2, xx3, xx4, xx5
