import torch
import torch.nn as nn


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        a = self.alpha
        g = self.gamma
        loss = - a * (1 - y_pred) ** g * y_true * torch.log(y_pred) - \
            (1 - a) * y_pred ** g * (1 - y_true) * torch.log(1 - y_pred)
        return loss.sum()
