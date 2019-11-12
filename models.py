import torch
import torch.nn as nn
import math
from utils.modules.nn import Swish
from utils.modules.backbones import DenseNet, Mini, ResNet, Xception


class SENet(nn.Module):
    def __init__(self, num_classes):
        super(SENet, self).__init__()
        # full pre-activation
        self.backbone = DenseNet()
        self.fc = nn.Sequential(
            nn.BatchNorm2d(1024), Swish(),
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
        x = self.backbone(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = SENet(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
