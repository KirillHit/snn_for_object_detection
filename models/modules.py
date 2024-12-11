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


class BatchNorm2dNoBias(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.bias = None
