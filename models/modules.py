import torch
from torch import nn
import torch.nn.functional as F


class SumPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, X):
        return (
            F.avg_pool2d(X, self.kernel_size, self.stride)
            * self.kernel_size
            * self.kernel_size
        )
