import torch.nn.functional as F
import torch.nn as nn
import torch

Tensor = torch.Tensor


class Sum_ReLU_2(nn.Module):


    def __init__(self) :
        super(Sum_ReLU_2, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x+abs(x)/torch.sum(abs(x)) * x, x  * 0)