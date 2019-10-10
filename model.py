import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, filters):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            *[
                # bias???
                nn.Conv2d(filters,
                          filters // 16,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                # nn.Dropout(0.5),
                nn.BatchNorm2d(filters // 16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(filters // 16,
                          filters,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                # nn.Dropout(0.5),
                nn.BatchNorm2d(filters),
                nn.Sigmoid()
            ])

    def forward(self, x):
        weights = self.gap(x)
        weights = self.fc(weights)
        return x * weights


def DBL(in_features, out_features, ksize, stride=1):
    padding = (ksize - 1) // 2
    layers = [
        nn.BatchNorm2d(in_features),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(in_features,
                  out_features,
                  ksize,
                  stride=stride,
                  padding=padding,
                  bias=False)
    ]
    return nn.Sequential(*layers)


class ResUnit(nn.Module):
    def __init__(self, filters):
        super(ResUnit, self).__init__()
        self.dbl1 = DBL(filters, filters // 2, 1)
        self.dbl2 = DBL(filters // 2, filters, 3)
        self.se = SELayer(filters)

    def forward(self, x):
        origin = x.clone()
        x = self.dbl1(x)
        x = self.dbl2(x)
        x = self.se(x)
        x += origin
        return x


class SENet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 128, 256, 512, 1024],
                 res_n=[1, 2, 8, 8, 4]):
        super(SENet, self).__init__()
        last_features = 32
        self.conv1 = nn.Conv2d(in_features,
                               last_features,
                               7,
                               padding=3,
                               bias=False)
        res_blocks = []
        for fi, f in enumerate(filters):
            layers = [DBL(last_features, f, 3, 2)] + [ResUnit(f)] * res_n[fi]
            res_blocks.append(nn.Sequential(*layers))
            last_features = f
        self.res_blocks = nn.Sequential(*res_blocks)
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.BatchNorm2d(filters[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(filters[-1], out_features, 7, padding=3),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.fc(x).view(x.shape[0], -1)
        x = F.softmax(x, 1)
        return x


if __name__ == "__main__":
    model = SENet(3, 8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
