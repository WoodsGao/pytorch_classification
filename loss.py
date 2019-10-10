import torch
import torch.nn as nn


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        loss = -y_true*self.alpha*torch.pow((1-y_pred), self.gamma)*torch.log(y_pred)-(1-y_true)*(1-self.alpha)*torch.pow(y_pred, self.gamma)*torch.log(1-y_pred)
        return loss.sum()