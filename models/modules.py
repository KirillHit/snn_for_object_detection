import torch
from torch import nn
import torch.nn.functional as F


class SumPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, X):
        return (
            F.avg_pool2d(X, self.kernel_size, self.stride, self.padding)
            * self.kernel_size
            * self.kernel_size
        )
