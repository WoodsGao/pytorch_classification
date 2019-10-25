import torch
import torch.nn as nn
import math
from utils.blocks import bn, relu, ResBlock, BLD, EmptyLayer


class SENet(nn.Module):
    def __init__(self, num_classes):
        super(SENet, self).__init__()

        # full pre-activation
        conv_block = BLD
        res_block = ResBlock
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
        self.additional_module = nn.Sequential(
            bn(512), relu
        )
        self.backbone = nn.Sequential(
            conv_block(32, 32, stride=2),
            conv_block(32, 32, dilation=3),
            conv_block(32, 64, stride=2),
            res_block(64, 64),
            res_block(64, 64, dilation=3),
            res_block(64, 128, stride=2),
            res_block(128, 128),
            res_block(128, 128, dilation=3),
            res_block(128, 128),
            res_block(128, 128, dilation=7),
            res_block(128, 128),
            res_block(128, 128, dilation=17),
            res_block(128, 128),
            res_block(128, 256, stride=2),
            res_block(256, 256),
            res_block(256, 256, dilation=3),
            res_block(256, 256),
            res_block(256, 256, dilation=7),
            res_block(256, 256),
            res_block(256, 256, dilation=17),
            res_block(256, 256),
            res_block(256, 512, stride=2),
            res_block(512, 512),
            res_block(512, 512, dilation=7),
            res_block(512, 512),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, num_classes, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.additional_module(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = SENet(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
