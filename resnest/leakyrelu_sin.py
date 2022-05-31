import torch.nn.functional as F
import torch.nn as nn
import torch

Tensor = torch.Tensor


class LeakyReLU_Sin(nn.Module):


    def __init__(self) :
        super(LeakyReLU_Sin, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, 0.02*torch.sin(x) * x)