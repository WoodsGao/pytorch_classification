import torch
import torch.nn as nn
import math
from utils.blocks import bn, relu, ResBlock


class SENet(nn.Module):
    def __init__(self,
                 num_classes,
                 filters=[64, 128, 256, 512, 1024],
                 res_n=[1, 2, 8, 8, 4]):
        super(SENet, self).__init__()
        assert (len(filters) == 5 and len(res_n) == 5)
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3, bias=False)
        layers = [ResBlock(32, filters[0], 2)
                  ] + [ResBlock(filters[0], filters[0], se_block=True)] * res_n[0]
        self.res1 = nn.Sequential(*layers)
        layers = [ResBlock(filters[0], filters[1], 2)
                  ] + [ResBlock(filters[1], filters[1], se_block=True)] * res_n[1]
        self.res2 = nn.Sequential(*layers)
        layers = [ResBlock(filters[1], filters[2], 2)
                  ] + [ResBlock(filters[2], filters[2], se_block=True)] * res_n[2]
        self.res3 = nn.Sequential(*layers)
        layers = [ResBlock(filters[2], filters[3], 2)
                  ] + [ResBlock(filters[3], filters[3])] * res_n[3]
        self.res4 = nn.Sequential(*layers)
        layers = [ResBlock(filters[3], filters[4], 2)
                  ] + [ResBlock(filters[4], filters[4])] * res_n[4]
        self.res5 = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            bn(filters[4]),
            relu,
            nn.Conv2d(filters[-1], num_classes, 1),
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
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = SENet(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
