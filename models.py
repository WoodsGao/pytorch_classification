import torch
import torch.nn as nn
from pytorch_modules.nn import CNS
from pytorch_modules.backbones import BasicModel, EfficientNet, ResNet


# gap classification model
class EfficientNetGCM(BasicModel):
    def __init__(self, num_classes, model_id=0):
        super(EfficientNetGCM, self).__init__()
        # full pre-activation
        self.backbone = EfficientNet(model_id)
        self.fc = nn.Sequential(
            CNS(self.backbone.out_channels[-1], self.backbone.out_channels[-1] * 4, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.backbone.out_channels[-1] * 4, num_classes, 1),
        )
        self.init()
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone.block1(x)
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        x = self.backbone.block4(x)
        x = self.backbone.block5(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = EfficientNetGCM(8)
    a = torch.ones([2, 3, 224, 224])
    b = model(a)
    b.mean().backward()
    print(b.shape)
