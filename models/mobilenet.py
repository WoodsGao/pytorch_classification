import torch.nn as nn

from pytorch_modules.backbones import mobilenet_v2


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(1280, num_classes, 1), nn.Flatten(1))

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.fc(x)
        return x
