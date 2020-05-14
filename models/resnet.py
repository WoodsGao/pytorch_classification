import torch.nn as nn

from pytorch_modules.backbones import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(512, num_classes, 1), nn.Flatten(1))

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.fc(x)
        return x
