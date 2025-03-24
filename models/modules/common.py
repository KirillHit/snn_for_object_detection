"""
Simple custom modules
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Any, NamedTuple
from norse.torch.module.snn import SNNCell, _merge_states

common_list = (
    "SumPool2d",
    "Storage",
    "StateStorage"
)


class SumPool2d(nn.Module):
    """Applies a 2D average pooling over an input signal composed of several input planes

    Summarizes the values of the cells of a kernel. To do this, it calls
    :external:func:`torch.nn.functional.avg_pool2d` and multiplies the result by the kernel area.
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        :param kernel_size: The size of the window.
        :type kernel_size: int
        :param stride: The stride of the window. Defaults to 1
        :type stride: int, optional
        :param padding: Implicit zero padding to be added on both sides. Defaults to 0
        :type padding: int, optional
        """
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Direct Module pass

        :param X: Input tensor.
        :type X: torch.Tensor
        :return: Result of summing pool.
        :rtype: torch.Tensor
        """
        return (
            F.avg_pool2d(X, self.kernel_size, self.stride, self.padding)
            * self.kernel_size
            * self.kernel_size
        )


class Storage(nn.Module):
    """
    Stores the forward pass values

    It is intended for use in feature pyramids, where you need to get multiple
    matrices from different places in the network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Store the input tensor and returns it back

        :param X: Input tensor.
        :type X: torch.Tensor
        :return: Input tensor.
        :rtype: torch.Tensor
        """
        self.storage = X
        return X

    def get_storage(self) -> torch.Tensor:
        """Returns the stored tensor

        :return: Stored tensor.
        :rtype: torch.Tensor
        """
        temp = self.storage
        self.storage = None
        return temp


class StateStorage(torch.nn.Module):
    """
    Class wrapper for neurons with a state of Norse. 
    Saves the intermediate states of the stored neurons for further analysis.
    """
    
    def __init__(self, m: SNNCell):
        """
        :param m: An initialized module with a state from the Norse library, 
            which will be called with a direct passage
        :type m: SNNCell
        """
        super().__init__()
        self.module = m
        self.state_list: List[torch.Tensor] = []
        self.spike_list: List[torch.Tensor] = []

    def get_state(self) -> NamedTuple:
        """Returns intermediate states of neurons"""
        return _merge_states(self.state_list)
    
    def get_spikes(self) -> torch.Tensor:
        """Returns spikes at the output of neurons for all time steps"""
        return torch.stack(self.spike_list)

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        """
        Conveys the value to his module directly.
        If the network is in non -training mode, retains intermediate states
        """
        if state is None:
            self.state_list.clear()
            self.spike_list.clear()
        out, new_state = self.module(input_tensor, state)
        if not self.training:
            self.state_list.append(new_state)
            self.spike_list.append(out)
        return out, new_state
