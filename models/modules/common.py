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
    "StorageGetter",
    "ResidualModule",
    "StateStorage",
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

    def __init__(self):
        super().__init__()
        self.storage = []
        self.channels = []
        self.requests_threshold = 0
        self.requests_idx = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Store the input tensor and returns it back"""
        self.storage.append(X)
        return X

    def add_requests(self) -> None:
        """Adds one to the receive counter threshold

        A receive is considered a call to the get method. If the threshold is not zero,
        then the storage will be automatically released when the threshold is reached.
        """
        self.number_requests += 1

    def add_input(self, channels: int) -> None:
        """Adds information about the input tensor that the storage will expect

        This is only necessary so that the generator can calculate the parameters
        for subsequent layers.

        :param channels: Number of input tensor channels
        :type channels: int
        """
        self.channels.append(channels)

    def shape(self) -> List[int]:
        """Returns a list of channel values for the expected data

        The method does not analyze the current data, but relies on the data received via the add_input method.
        """
        return self.channels

    def reset(self) -> None:
        """Resets storage"""
        self.storage = []

    def get(self) -> List[torch.Tensor]:
        """Returns a list of saved tensors"""
        temp = self.storage
        if self.number_requests:
            self.requests_idx += 1
            if self.requests_idx == self.number_requests:
                self.requests_idx = 0
                self.reset()
        return temp


class StorageGetter(nn.Module):
    """Returns the specified stored tensor"""

    def __init__(self, storage: Storage, idx: int = 0):
        super().__init__()
        self.storage, self.idx = storage, idx

    def forward(self) -> torch.Tensor:
        """Store the input tensor and returns it back"""
        return self.storage.get()[self.idx]


class ResidualModule(nn.Module):
    def __init__(self, type: str, storage: Storage):
        super().__init__()
        self.storage = storage
        types = {"residual": self._residual, "dense": self._dense}
        self.func = types[type]

    def forward(self) -> torch.Tensor:
        return self.func(self.storage.get())

    def _residual(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(input).sum(dim=0)

    def _dense(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(input, dim=1)


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
