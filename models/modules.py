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


class Storage(nn.Module):
    """
    Stores the forward pass values. The "get" method returns the stored tensor.
    It is intended for use in feature pyramids, where you need to get multiple
    matrices from different places in the network.
    """

    storage: torch.Tensor

    def forward(self, X):
        self.storage = X
        return X

    def get_storage(self):
        temp = self.storage
        self.storage = None
        return temp


class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=1,
            bias=False,
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
        Args:
            X (torch.Tensor): Shape [batch, in_channels, h, w]
            state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
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

        return hidden_next, (hidden_next, cell_next)


#####################################################################
#                         Layer Generators                          #
#####################################################################


class LayerGen:
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        raise NotImplementedError


type ListGen = List[LayerGen | ListGen]
type ListState = List[torch.Tensor | None | ListState]


class Pass(LayerGen):
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Identity(), in_channels


class Conv(LayerGen):
    def __init__(self, out_channels: int = None, kernel_size: int = 3, stride: int = 1):
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
    def __init__(self, type: str, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
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
    def __init__(self, scale: int = 2, mode: str = "bilinear"):
        self.scale = scale
        self.mode = mode

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Upsample(scale_factor=self.scale, mode=self.mode), in_channels


class Norm(LayerGen):
    def __init__(self, bias: bool = False):
        self.bias = bias

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        norm_layer = nn.BatchNorm2d(in_channels)
        if not self.bias:
            norm_layer.bias = None
        return norm_layer, in_channels


class LIF(LayerGen):
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return snn.LIFCell(), in_channels


class LI(LayerGen):
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return snn.LICell(), in_channels


class ReLU(LayerGen):
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.ReLU(), in_channels


class SiLU(LayerGen):
    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.SiLU(), in_channels


class LSTM(LayerGen):
    def __init__(self, hidden_size: Optional[int] = None):
        self.hidden_size = hidden_size

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        h_size = in_channels if self.hidden_size is None else self.hidden_size
        return ConvLSTM(in_channels, h_size), h_size


class Return(LayerGen):
    out_channels: int

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        self.out_channels = in_channels
        return Storage(), in_channels


#####################################################################
#                         Block Generators                          #
#####################################################################


class BlockGen(nn.Module):
    """Network Structural Elements Generator"""

    out_channels: int

    def __init__(self, in_channels: int, cfgs: ListGen):
        """Takes as input two-dimensional arrays of layer generators.
        The inner dimensions are sequential, the outer ones are added together.
        Lists can include blocks from other two-dimensional lists.
        They will be considered recursively.

        Args:
            in_channels (int): Number of input channels
            cfgs (ListGen): Two-dimensional lists of layer generators
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
    ) -> Tuple[nn.ModuleList, ListState, int]:
        """Recursively generates a network branch from a configuration list.
        Args:
            in_channels (int): Number of input channels
            cfg (ListGen): Lists of layer generators
        Returns:
            Tuple[nn.ModuleList, ListState, int]:
                0 - List of modules;
                1 - Mask of layers requiring states;
                2 - Number of output channels;
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
        """
        Args:
            X (torch.Tensor): Input tensor. Shape is [batch, p, h, w].
            state (ListState, optional). Defaults to None.
        Returns:
            Tuple[torch.Tensor, ListState].
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
    out_channels: int = 0
    default_cfgs: Dict[str, ListGen] = {}

    def __init__(
        self,
        cfg: str | ListGen,
        in_channels: int = 2,
        init_weights: bool = False,
    ) -> None:
        super().__init__()

        self._load_cfg()
        self.net_cfg = cfg if isinstance(cfg, list) else self.default_cfgs[cfg]

        self._net_generator(in_channels)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.9, std=0.1)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)

    def _net_generator(self, in_channels: int) -> None:
        self.net = BlockGen(in_channels, [self.net_cfg])
        self.out_channels = self.net.out_channels

    def _load_cfg(self):
        raise NotImplementedError

    def forward_impl(self, X, state):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError
