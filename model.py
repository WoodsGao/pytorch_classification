import torch
import torch.nn as nn

relu = nn.LeakyReLU(0.1)
bn = nn.BatchNorm2d


class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            bn(filters),
            nn.Conv2d(filters, filters // 16, 1, bias=False),
            relu,
            nn.Conv2d(filters // 16, filters, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        weight = self.weight(gap)
        return x * weight


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            bn(in_features),
            relu,
            SELayer(in_features),
            nn.Conv2d(in_features, out_features // 2, 1, 1, 0, bias=False),
            bn(out_features // 2),
            relu,
            nn.Conv2d(out_features // 2, out_features, 3, stride, 2, bias=False, groups=32, dilation=2),
            # SELayer(out_features),
        )
        self.downsample = None
        if stride > 1 or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, 3, stride, 1),
            )

    def forward(self, x):
        if self.downsample is not None:
            downsample = self.downsample(x)
        else:
            downsample = x
        return downsample + self.block(x)


class SENet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 128, 256, 512, 1024],
                 res_n=[1, 2, 8, 8, 4]):
        super(SENet, self).__init__()
        last_features = 32
        layers = [
            nn.Conv2d(in_features, last_features, 7, padding=3, bias=False),
            # bn(last_features),
            # relu,
        ]
        for fi, f in enumerate(filters):
            layers += [ResBlock(last_features, f, 2)
                       ] + [ResBlock(f, f)] * res_n[fi]
            last_features = f
        layers += [
            bn(last_features),
            relu,
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filters[-1], out_features, 1),
            nn.Sigmoid(),
        ]
        self.seq = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.flatten(self.seq(x), 1)


if __name__ == "__main__":
    model = SENet(3, 8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
