"""
Model generation tools

Custom types
------------

List for recursive model generation

.. data:: ListGen
    :type: List[LayerGen | ListGen]

List for storing the state of the model

.. data:: ListState
    :type: List[torch.Tensor | None | ListState]
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from norse.torch.utils.state import _is_module_stateful
import norse.torch as snn

__all__ = (
    "SumPool2d",
    "Storage",
    "LayerGen",
    "ListGen",
    "ListState",
    "Pass",
    "Conv",
    "Norm",
    "LIF",
    "LI",
    "ReLU",
    "SiLU",
    "LSTM",
    "Pool",
    "Up",
    "Return",
    "BlockGen",
    "ModelGen",
)


#####################################################################
#                          Custom modules                           #
#####################################################################
nn.AvgPool2d


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

    _storage: torch.Tensor

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Store the input tensor and returns it back

        :param X: Input tensor.
        :type X: torch.Tensor
        :return: Input tensor.
        :rtype: torch.Tensor
        """
        self._storage = X
        return X

    def get_storage(self) -> torch.Tensor:
        """Returns the stored tensor

        :return: Stored tensor.
        :rtype: torch.Tensor
        """
        temp = self._storage
        self._storage = None
        return temp


class ConvLSTM(nn.Module):
    """Convolutional LSTM

    For more details, see https://github.com/ndrplz/ConvLSTM_pytorch/tree/master.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
    ):
        """
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param hidden_channels: Number of hidden channels.
        :type hidden_channels: int
        :param kernel_size: Size of the convolving kernel. Defaults to 1.
        :type kernel_size: int, optional
        :param bias: If ``True``, adds a learnable bias to the output. Defaults to False.
        :type bias: bool, optional
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def _init_hidden(self, target: torch.Tensor):
        batch, _, h, w = target.shape
        return (
            torch.zeros((batch, self.hidden_channels, h, w), device=target.device),
            torch.zeros((batch, self.hidden_channels, h, w), device=target.device),
        )

    def forward(
        self, X: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param X: Input tensor.  Shape [batch, channel, h, w].
        :type X: torch.Tensor
        :param state: Past state of the cell. Defaults to None.
            It is a list of the form: (hidden state, cell state).
        :type state: Optional[Tuple[torch.Tensor, torch.Tensor]], optional
        :return: List of form: (next hidden state, (next hidden state, next cell state)).
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        hidden_state, cell_state = self._init_hidden(X) if state is None else state
        combined = torch.cat([X, hidden_state], dim=1)
        combined = self.conv(combined)
        input_gate, forget_gate, out_gate, in_node = torch.split(
            combined, self.hidden_channels, dim=1
        )
        I = torch.sigmoid(input_gate)
        F = torch.sigmoid(forget_gate)
        O = torch.sigmoid(out_gate)
        C = torch.tanh(in_node)

        cell_next = F * cell_state + I * C
        hidden_next = O * torch.tanh(cell_next)

        # This form is needed for the model generator to work
        return hidden_next, (hidden_next, cell_next)


#####################################################################
#                         Layer Generators                          #
#####################################################################


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

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return snn.LIFCell(), in_channels


class LI(LayerGen):
    """Generator of the layer of LI neurons

    Uses :external:class:`norse.torch.module.leaky_integrator.LICell` module.
    """

    def get(self, in_channels: int) -> Tuple[snn.LICell, int]:
        return snn.LICell(), in_channels


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


class LSTM(LayerGen):
    """LSTM layer generator

    Uses :class:`ConvLSTM` module.
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


#####################################################################
#                         Block Generators                          #
#####################################################################


type ListGen = List[LayerGen | ListGen]
type ListState = List[torch.Tensor | None | ListState]


class BlockGen(nn.Module):
    """Block generator for the network

    Takes as input two-dimensional arrays of :class:`LayerGen`.
    The inner dimensions are sequential, the outer ones are added together.
    Lists can include blocks from other two-dimensional lists.
    They will be considered recursively.

    .. code-block::
        :caption: Simple configuration list example

        def vgg_block(out_channels: int, kernel: int = 3):
            return Conv(out_channels, kernel), Norm(), LIF()

        cfgs: ListGen = [
            *vgg_block(8), Pool("S"), *vgg_block(32), Pool("S"), *vgg_block(64), Pool("S")
        ]

    .. code-block::
        :caption: Example of a configuration list with residual links

        def conv(out_channels: int, kernel: int = 3, stride: int = 1):
            return (
                Conv(out_channels, stride=stride, kernel_size=kernel),
                Norm(),
                LIF(),
            )

        def res_block(out_channels: int, kernel: int = 3):
            return (
                Conv(out_channels, 1),
                # Residual block. The values from all branches are added together
                [
                    [*conv(out_channels, kernel)],
                    [Conv(out_channels, 1)],
                ],
                Conv(out_channels, 1),
            )

        cfgs: ListGen = [
            *conv(64, 7, 2), *res_block(64, 5), *conv(128, 5, 2), *res_block(128)
        ]

    """

    out_channels: int
    """The number of channels that will be after applying this block to 
    a tensor with ``in_channels`` channels."""

    def __init__(self, in_channels: int, cfgs: ListGen):
        """
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param cfgs: Two-dimensional lists of layer generators.
        :type cfgs: ListGen
        """
        super().__init__()
        branch_list: List[nn.ModuleList] = []
        self.branch_state: List[List[bool]] = []
        for branch_cfg in cfgs:
            layer_list, state_layers, self.out_channels = self._make_branch(
                in_channels, branch_cfg
            )
            branch_list.append(layer_list)
            self.branch_state.append(state_layers)
        self.net = nn.ModuleList(branch_list)

    def _make_branch(
        self, in_channels: int, cfg: ListGen
    ) -> Tuple[nn.ModuleList, List[bool], int]:
        """Recursively generates a network branch from a configuration list

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param cfg: Lists of layer generators.
        :type cfg: ListGen
        :return:
            1. List of modules;
            2. Mask of layers requiring states;
            2. Number of output channels;
        :rtype: Tuple[nn.ModuleList, List[bool], int]
        """
        state_layers: List[bool] = []
        layer_list: List[nn.Module] = []
        channels = in_channels
        for layer_gen in cfg:
            if isinstance(layer_gen, list):
                layer = BlockGen(channels, layer_gen)
                channels = layer.out_channels
            else:
                layer, channels = layer_gen.get(channels)
            layer_list.append(layer)
            state_layers.append(_is_module_stateful(layer))
        return nn.ModuleList(layer_list), state_layers, channels

    def forward(
        self, X: torch.Tensor, state: ListState | None = None
    ) -> Tuple[torch.Tensor, ListState]:
        """Direct block pass

        :param X: Input tensor. Shape is Shape [batch, channel, h, w].
        :type X: torch.Tensor
        :param state: List of block layer states. Defaults to None.
        :type state: ListState | None, optional
        :return: The resulting tensor and the list of new states.
        :rtype: Tuple[torch.Tensor, ListState]
        """
        out = []
        out_state = []
        state = [None] * len(self.net) if state is None else state
        for branch, state_flags, branch_state in zip(
            self.net, self.branch_state, state
        ):
            branch_state = (
                [None] * len(branch) if branch_state is None else branch_state
            )
            Y = X
            for idx, (layer, is_state) in enumerate(zip(branch, state_flags)):
                if is_state:
                    Y, branch_state[idx] = layer(Y, branch_state[idx])
                else:
                    Y = layer(Y)
            out.append(Y)
            out_state.append(branch_state)
        return torch.stack(out).sum(dim=0), out_state


#####################################################################
#                        Backbone Generator                         #
#####################################################################


class ModelGen(nn.Module):
    """Base class for model generators

    Class :class:`BlockGen` is used as a generation tool.

    Child classes must define ways to process input and output data.
    Different variants are presented in classes :class:`models.backbone.BackboneGen`,
    :class:`models.neck.NeckGen`, :class:`models.head.HeadGen`.

    .. warning::

        This class can only be used as a base class for inheritance.
    """

    out_channels: int = 0
    """The number of channels that will be after applying this block to 
    a tensor with ``in_channels`` channels."""

    default_cfgs: Dict[str, ListGen] = {}
    """List of configurations provided by default."""

    def __init__(
        self,
        cfg: str | ListGen,
        in_channels: int = 2,
        init_weights: bool = True,
    ) -> None:
        """
        :param cfg: Lists of layer generators.
        :type cfg: str | ListGen
        :param in_channels: Number of input channels. Defaults to 2.
        :type in_channels: int, optional
        :param init_weights: If ``true`` apply weight initialization function. 
            Defaults to True.
        :type init_weights: bool, optional
        """
        super().__init__()

        self._load_cfg()
        self.net_cfg = cfg if isinstance(cfg, list) else self.default_cfgs[cfg]

        self._net_generator(in_channels)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _net_generator(self, in_channels: int) -> None:
        self.net = BlockGen(in_channels, [self.net_cfg])
        self.out_channels = self.net.out_channels

    def _load_cfg(self):
        raise NotImplementedError

    def forward_impl(self, X: torch.Tensor, state: ListState | None = None):
        """State-based network pass

        :param X: Input tensor. Shape is Shape [batch, channel, h, w].
        :type X: torch.Tensor
        :param state: List of block layer states. Defaults to None.
        :type state: ListState | None, optional
        :return: The resulting tensor and the list of new states.
        :rtype: Tuple[torch.Tensor, ListState]
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor):
        """Network pass for data containing time resolution

        :param X: Input tensor. Shape is Shape [ts, batch, channel, h, w].
        :type X: torch.Tensor
        :return: The resulting tensor and the list of new states.
        :rtype: torch.Tensor,
        """
        raise NotImplementedError
