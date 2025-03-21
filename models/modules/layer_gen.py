"""
Layer Generators
"""

from torch import nn
from typing import Tuple, Optional
import norse.torch as snn
from norse.torch.module.snn import SNNCell
from models.modules.synapse import SynapseCell
from models.modules.conv_lstm import ConvLSTM
from models.modules.sli import SLICell
from models.modules.common import *

layers_list = (
    "Residual",
    "Dense",
    "LayerGen",
    "Pass",
    "Conv",
    "Norm",
    "LIF",
    "LI",
    "ReLU",
    "SiLU",
    "Tanh",
    "LSTM",
    "Pool",
    "Up",
    "Return",
    "Synapse",
    "SLI",
)


class Residual(list):
    """Class inherited from :external:class:`list` type without changes

    Needed to mark a network in the configuration as residual.

    .. code-block::
        :caption: Example

        Residual(
            [
                [*conv(out_channels, kernel)],
                [Conv(out_channels, 1)],
            ]
        )
    """

    pass


class Dense(list):
    """Class inherited from :external:class:`list` type without changes

    Needed to mark the network in the configuration as densely connected.

    .. code-block::
        :caption: Example

        Dense(
            [
                [*conv(out_channels, kernel)],
                [Conv(out_channels, 1)],
            ]
        )
    """

    pass


class LayerGen:
    """Base class for model layer generators

    The ``get`` method must initialize the network module and pass it to the generator
    (See :class:`BlockGen`).

    .. warning::

        This class can only be used as a base class for inheritance.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        """Initializes and returns the network layer

        :param in_channels: Number of input channels.
        :type in_channels: int
        :return: The generated module and the number of channels that will be after applying
            this layer to a tensor with ``in_channels`` channels.
        :rtype: Tuple[nn.Module, int]
        """
        raise NotImplementedError


class Pass(LayerGen):
    """A placeholder layer generator that does nothing

    Uses :external:class:`torch.nn.Identity` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Identity(), in_channels


class Conv(LayerGen):
    """Generator of standard 2d convolution

    Uses :external:class:`torch.nn.Conv2d` module.
    Bias defaults to ``False``, padding is calculated automatically.
    """

    def __init__(self, out_channels: int = None, kernel_size: int = 3, stride: int = 1):
        """
        :param out_channels: Number of channels produced by the convolution.
            Defaults to None.
        :type out_channels: int, optional
        :param kernel_size:  Size of the convolving kernel. Defaults to 3.
        :type kernel_size: int, optional
        :param stride: Stride of the convolution. Defaults to 1.
        :type stride: int, optional
        """
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        out = in_channels if self.out_channels is None else self.out_channels
        return nn.Conv2d(
            in_channels,
            out,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            stride=self.stride,
            bias=False,
        ), out


class Pool(LayerGen):
    """Pooling layer generator

    Uses modules :external:class:`torch.nn.AvgPool2d`,
    :external:class:`torch.nn.MaxPool2d`, :class:`SumPool2d`.
    """

    def __init__(self, type: str, kernel_size: int = 2, stride: Optional[int] = None):
        """
        :param type: Pooling type.

            - ``A`` - :external:class:`torch.nn.AvgPool2d`.
            - ``M`` - :external:class:`torch.nn.MaxPool2d`.
            - ``S`` - :class:`SumPool2d`.
        :type type: str
        :param kernel_size: The size of the window. Defaults to 2.
        :type kernel_size: int, optional
        :param stride: The stride of the window. Default value is kernel_size.
        :type stride: Optional[int], optional
        :raises ValueError: Non-existent pool type.
        """
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        match type:
            case "A":
                self.pool = nn.AvgPool2d
            case "M":
                self.pool = nn.MaxPool2d
            case "S":
                self.pool = SumPool2d
            case _:
                raise ValueError(f'[ERROR]: Non-existent pool type "{type}"!')

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return self.pool(self.kernel_size, self.stride), in_channels


class Up(LayerGen):
    """Upsample layer generator

    Uses :external:class:`torch.nn.Upsample` module.
    """

    def __init__(self, scale: int = 2, mode: str = "nearest"):
        """
        :param scale:  Multiplier for spatial size. Defaults to 2.
        :type scale: int, optional
        :param mode: The upsampling algorithm: one of 'nearest', 'linear', 'bilinear',
            'bicubic' and 'trilinear'. Defaults to "nearest".
        :type mode: str, optional
        """
        self.scale = scale
        self.mode = mode

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Upsample(scale_factor=self.scale, mode=self.mode), in_channels


class Norm(LayerGen):
    """Batch Normalization layer generator

    Uses :external:class:`torch.nn.BatchNorm2d` module.
    """

    def __init__(self, bias: bool = False):
        """
        :param bias: If True, adds a learnable bias. Defaults to False.
        :type bias: bool, optional
        """
        self.bias = bias

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        norm_layer = nn.BatchNorm2d(in_channels)
        if not self.bias:
            norm_layer.bias = None
        return norm_layer, in_channels


class LIF(LayerGen):
    """Generator of the layer of LIF neurons

    Uses :external:class:`norse.torch.module.lif.LIFCell` module.
    """

    def __init__(self, state_storage: bool = False):
        """
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        module = snn.LIFCell() if not self.state_storage else StateStorage(snn.LIFCell())
        return module, in_channels


class LI(LayerGen):
    """Generator of the layer of LI neurons

    Uses :external:class:`norse.torch.module.leaky_integrator.LICell` module.
    """

    def __init__(self, state_storage: bool = False):
        """
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        module = snn.LICell() if not self.state_storage else StateStorage(snn.LICell())
        return module, in_channels


class ReLU(LayerGen):
    """ReLU layer generator

    Uses :external:class:`torch.nn.ReLU` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.ReLU(), in_channels


class SiLU(LayerGen):
    """SiLU layer generator

    Uses :external:class:`torch.nn.SiLU` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.SiLU(), in_channels


class Tanh(LayerGen):
    """SiLU layer generator

    Uses :external:class:`torch.nn.Tanh` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Tanh(), in_channels


class LSTM(LayerGen):
    """LSTM layer generator

    Uses :class:`ConvLSTM <models.module.conv_lstm.ConvLSTM>` module.
    """

    def __init__(self, hidden_size: Optional[int] = None):
        """
        :param hidden_size: Number of hidden channels. Defaults to None.
        :type hidden_size: Optional[int], optional
        """
        self.hidden_size = hidden_size

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        h_size = in_channels if self.hidden_size is None else self.hidden_size
        return ConvLSTM(in_channels, h_size), h_size


class Return(LayerGen):
    """Generator of layers for storing forward pass values

    It is intended for use in feature pyramids, where you need to get multiple
    matrices from different places in the network.

    Uses :class:`Storage` module.
    """

    out_channels: int

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        self.out_channels = in_channels
        return Storage(), in_channels


class Synapse(LayerGen):
    """Generator of the layer of synapse

    Uses :class:`SynapseCell <models.module.synapse.SynapseCell>` module.
    """

    def get(self, in_channels: int) -> Tuple[snn.LICell, int]:
        return SynapseCell(), in_channels


class SLI(LayerGen):
    """Generator of the layer of Saturable LI neurons

    Uses :class:`SLICell <models.module.sli.SLICell>` module.
    """

    def __init__(self, state_storage: bool = False):
        """
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        module = SLICell() if not self.state_storage else StateStorage(SLICell())
        return module, in_channels
