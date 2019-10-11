import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if weight is not None:
            self.weight = weight.unsqueeze(0)
        else:
            self.weight = None

    def forward(self, y_pred, y_true):
        a = self.alpha
        g = self.gamma
        loss = - a * torch.pow((1 - y_pred), g) * y_true * torch.log(y_pred + 1e-10) - \
            (1 - a) * torch.pow(y_pred, g) * (1 - y_true) * torch.log(1 - y_pred + 1e-10)
        if self.weight is not None:
            loss *= self.weight
        return loss.sum(1)
