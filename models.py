import torch
import torch.nn as nn
from utils.modules.backbones import BasicModel, EfficientNetB4


# gap classification model
class GCM(BasicModel):
    def __init__(self, num_classes):
        super(GCM, self).__init__()
        # full pre-activation
        self.backbone = EfficientNetB4()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(448, num_classes, 1),
        )
        self.init()

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
