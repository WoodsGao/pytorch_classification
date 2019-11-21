import torch
import torch.nn as nn
import math
from utils.modules.nn import Swish
from utils.modules.backbones import DenseNet, BasicModel, EfficientNetB4


# gap classification model
class GCM(BasicModel):
    def __init__(self, num_classes):
        super(GCM, self).__init__()
        # full pre-activation
        self.backbone = EfficientNetB4()
        self.fc = nn.Sequential(
            nn.GroupNorm(8, 448), Swish(),
            nn.Conv2d(448, num_classes, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.init()
        self.weight_standard()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = GCM(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    print(b.shape)
