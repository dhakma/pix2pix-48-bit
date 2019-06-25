from torch import nn
import torch


class TorchVecLoss(nn.Module):

    def __init__(self):
        super(TorchVecLoss, self).__init__()

    # def forward(self, x, y):
    #     return torch.sum((x - y).pow(2).sum(1))

    def forward(self, x, y):
        return torch.sum(torch.sqrt((x - y).pow(2).sum(1)))
