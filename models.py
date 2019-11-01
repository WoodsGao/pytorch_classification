import torch
import torch.nn as nn
import math
from utils.blocks import bn, lrelu, ResBlock, BLD, EmptyLayer


class SENet(nn.Module):
    def __init__(self, num_classes):
        super(SENet, self).__init__()
        # full pre-activation
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
        self.block1 = nn.Sequential(ResBlock(32, 64, stride=2))
        self.block2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128, dilation=6),
            ResBlock(128, 128),
        )
        self.block3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=6),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=12),
            ResBlock(256, 256),
            ResBlock(256, 256, dilation=18),
            ResBlock(256, 256),
        )
        self.block4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=6),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=12),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=18),
            ResBlock(512, 512),
            ResBlock(512, 512, dilation=30),
            ResBlock(512, 512),
        )
        self.block5 = nn.Sequential(
            ResBlock(512, 1024, stride=2),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024, dilation=6),
            ResBlock(1024, 1024, dilation=12),
            ResBlock(1024, 1024, dilation=18),
            ResBlock(1024, 1024),
        )
        self.fc = nn.Sequential(
            bn(1024),
            lrelu,
            nn.Conv2d(1024, num_classes, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = SENet(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
