import torch.nn.functional as F
import torch.nn as nn
import torch

Tensor = torch.Tensor


class Sum_ReLU(nn.Module):


    def __init__(self) :
        super(Sum_ReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, abs(x)/torch.sum(abs(x)) * torch.sin(x)  * x)