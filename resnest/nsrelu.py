import torch.nn.functional as F
import torch.nn as nn
import torch

Tensor = torch.Tensor


class NSReLU(nn.Module):


    def __init__(self) :
        super(NSReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, abs(x)/torch.sum(abs(x)) * x)
