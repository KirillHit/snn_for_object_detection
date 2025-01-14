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
from typing import List, Tuple, Dict
from norse.torch.utils.state import _is_module_stateful
from utils.anchors import AnchorGenerator
from .modules import *


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
                Residual(
                    [
                        [*conv(out_channels, kernel)],
                        [Conv(out_channels, 1)],
                    ]
                ),
                Conv(out_channels, 1),
            )

        cfgs: ListGen = [
            *conv(64, 7, 2), *res_block(64, 5), *conv(128, 5, 2), *res_block(128)
        ]

    """

    out_channels: int = 0
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

        if isinstance(cfgs, Residual):
            self.out_type = self._residual_out
            self.channels_calc = self._residual_channels
        elif isinstance(cfgs, Dense):
            self.out_type = self._dense_out
            self.channels_calc = self._dense_channels
        else:
            self.out_type = self._forward_out
            self.channels_calc = self._forward_channels
            cfgs = [cfgs]

        for branch_cfg in cfgs:
            layer_list, state_layers, out_channels = self._make_branch(
                in_channels, branch_cfg
            )
            branch_list.append(layer_list)
            self.branch_state.append(state_layers)
            self.channels_calc(out_channels)
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

    def _residual_out(self, out: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(out).sum(dim=0)

    def _residual_channels(self, channels: int) -> torch.Tensor:
        if not self.out_channels:
            self.out_channels = channels
        elif self.out_channels != channels:
            raise RuntimeError(
                "[ERROR]: The number of channels in the residual "
                "network does not match! Check the configuration settings."
            )

    def _dense_out(self, out: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(out, dim=1)

    def _dense_channels(self, channels: int) -> torch.Tensor:
        self.out_channels += channels

    def _forward_out(self, out: List[torch.Tensor]) -> torch.Tensor:
        return out[0]

    def _forward_channels(self, channels: int) -> torch.Tensor:
        self.out_channels = channels

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
        return self.out_type(out), out_state


#####################################################################
#                        Backbone Generator                         #
#####################################################################


class BaseConfig:
    """Base class for model configuration generators"""

    def backbone_cfgs(self) -> ListGen:
        """Generates and returns a network backbone configuration

        :return: Backbone configuration.
        :rtype: ListGen
        """
        raise NotImplementedError

    def neck_cfgs(self) -> ListGen:
        """Generates and returns a network neck configuration

        :return: Neck configuration.
        :rtype: ListGen
        """
        raise NotImplementedError

    def head_cfgs(self, box_out: int, cls_out: int) -> ListGen:
        """Generates and returns a network head configuration

        :param box_out: Number of output channels for box predictions.
        :type box_out: int
        :param cls_out: Number of output channels for class predictions.
        :type cls_out: int
        :return: Head configuration.
        :rtype: ListGen
        """
        raise NotImplementedError


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
        cfg: BaseConfig,
        in_channels: int = 2,
        init_weights: bool = True,
    ) -> None:
        """
        :param cfg: Network Configuration Generator.
        :type cfg: BaseConfig
        :param in_channels: Number of input channels. Defaults to 2.
        :type in_channels: int, optional
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        """
        super().__init__()

        self.net_cfg = self._load_cfg(cfg)

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
        self.net = BlockGen(in_channels, self.net_cfg)
        self.out_channels = self.net.out_channels

    def _load_cfg(self, cfg: BaseConfig) -> ListGen:
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


#####################################################################
#                        Backbone Generator                         #
#####################################################################


class BackboneGen(ModelGen):
    """Model base generator

    Returns the tensor from the last layer of the network.
    """

    def _load_cfg(self, cfg: BaseConfig) -> ListGen:
        return cfg.backbone_cfgs()

    def forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        return self.net(X, state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = []
        state = None
        for time_step_x in X:
            Y, state = self.forward_impl(time_step_x, state)
            out.append(Y)
        return torch.stack(out)


#####################################################################
#                          Neck Generator                           #
#####################################################################


class NeckGen(ModelGen):
    """Network neck generator

    Returns a list of tensors that were stored in the
    :class:`models.modules.Return` layers.
    """

    out_shape: List[int]
    """Stores the format of the output data 
    
    - The number of elements is equal to the number of tensors in the output list.
    - The numeric value is equal to the number of channels of the corresponding tensor.
    
    This data is required to initialize :class:`models.head.Head`.
    """

    def __init__(
        self, cfg: str | ListGen, in_channels: int = 2, init_weights: bool = False
    ):
        super().__init__(cfg, in_channels, init_weights)
        self.out_shape = self._search_out(self.net_cfg)

    def _search_out(self, cfg: str | ListGen) -> List[int]:
        """Finds the indices of the layers from which it is necessary to obtain tensors

        :param cfg: Lists of layer generators.
        :type cfg: str | ListGen
        :return: List of layer indices from which values will be returned.
        :rtype: List[int]
        """
        out: List[int] = []
        for module in cfg:
            if isinstance(module, Return):
                out.append(module.out_channels)
            elif isinstance(module, list):
                out += self._search_out(module)
        return out

    def _load_cfg(self, cfg: BaseConfig) -> ListGen:
        return cfg.neck_cfgs()

    def forward_impl(
        self, X: List[torch.Tensor], state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        out = []
        _, state = self.net(X, state)
        for module in self.net.modules():
            if isinstance(module, Storage):
                out.append(module.get_storage())
        return out, state

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        storage: List[List[torch.Tensor]] = [[] for _ in self.out_shape]
        state = None
        for time_step_x in X:
            Y, state = self.forward_impl(time_step_x, state)
            for idx, res in enumerate(Y):
                storage[idx].append(res)
        out: List[torch.Tensor] = []
        for ret_layer in storage:
            out.append(torch.stack(ret_layer))
        return out


#####################################################################
#                          Head descriptor                          #
#####################################################################


class Head(nn.Module):
    """Head model holder

    Applies a head model to multiple maps and merges them.
    For each input map, its own :class:`HeadGen` model is generated.
    Predictions obtained for different feature maps are combined.
    """

    def __init__(
        self,
        cfg: str | ListGen,
        num_classes: int,
        in_shape: List[int],
        init_weights: bool = True,
    ) -> None:
        """
        :param cfg: Lists of layer generators.
        :type cfg: str | ListGen
        :param num_classes: Number of classes.
        :type num_classes: int
        :param in_shape: Input data format.
            Expecting to receive a list of :attr:`models.neck.NeckGen.out_shape`
        :type in_shape: List[int]
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        """
        super().__init__()
        self.num_classes = num_classes

        # TODO Automatic calculation
        max = 0.75
        min = 0.08
        size_per_pix = 3
        sizes = torch.arange(
            min, max, (max - min) / (len(in_shape) * size_per_pix), dtype=torch.float32
        )
        sizes = sizes.reshape((-1, size_per_pix))
        ratios = torch.tensor((0.5, 1.0, 2), dtype=torch.float32)

        num_anchors = size_per_pix * len(ratios)
        num_class_out = num_anchors * (self.num_classes + 1)
        num_box_out = num_anchors * 4

        for idx, channels in enumerate(in_shape):
            setattr(
                self,
                f"anchor_gen_{idx}",
                AnchorGenerator(sizes=sizes[idx], ratios=ratios),
            )
            setattr(
                self,
                f"model_{idx}",
                HeadGen(cfg, num_box_out, num_class_out, channels, init_weights),
            )

    def forward(
        self, X: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct network pass

        :param X: Feature map list. One map shape [ts, batch, channel, h, w].
        :type X: List[torch.Tensor]
        :return: Predictions made by a neural network.
            Contains three tensors:

            1. anchors: Shape [anchor, 4]
            2. cls_preds: Shape [ts, batch, anchor, num_classes + 1]
            3. bbox_preds: Shape [ts, batch, anchor, 4]
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        anchors, cls_preds, bbox_preds = [], [], []

        for idx, map in enumerate(X):
            anchors.append(getattr(self, f"anchor_gen_{idx}")(map))
            boxes, classes = getattr(self, f"model_{idx}")(map)
            bbox_preds.append(boxes)
            cls_preds.append(classes)

        anchors = torch.cat(anchors, dim=0)
        cls_preds = self._concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], cls_preds.shape[1], -1, self.num_classes + 1
        )
        bbox_preds = self._concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], bbox_preds.shape[1], -1, 4)
        return anchors, cls_preds, bbox_preds

    def _flatten_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor so that each pixel retains channels values and smooths each batch"""
        return torch.flatten(torch.permute(pred, (0, 1, 3, 4, 2)), start_dim=2)

    def _concat_preds(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """Concatenating Predictions for Multiple Scales"""
        return torch.cat([self._flatten_pred(p) for p in preds], dim=2)


#####################################################################
#                          Head Generator                           #
#####################################################################


class HeadGen(ModelGen):
    """Model head generator

    The configuration lists for this module look different.

    .. code-block::
        :caption: Configuration list example

        cfgs: ListGen = [
            [
                Conv(kernel_size=1),
                Norm(),
                LSTM(),
            ],
            [
                Conv(box_out, 1),
            ],
            [
                Conv(cls_out, 1),
            ],
        ],

    The configuration includes three lists:

    - The first one is for data preparation
    - The second one is for box prediction
    - The third one is for class prediction

    Box and class prediction models use the output of the preparation
    network as input.
    """

    def __init__(
        self,
        cfg: str | ListGen,
        box_out: int,
        cls_out: int,
        in_channels: int = 2,
        init_weights=False,
    ):
        """
        :param cfg: Lists of layer generators.
        :type cfg: str | ListGen
        :param box_out: The number of channels obtained as a result of the  class prediction network.
        :type box_out: int
        :param cls_out: The number of channels obtained as a result of the box prediction network.
        :type cls_out: int
        :param in_channels: Number of input channels. Defaults to 2.
        :type in_channels: int, optional
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        """
        self.box_out = box_out
        self.cls_out = cls_out
        super().__init__(cfg, in_channels, init_weights)

    def _net_generator(self, in_channels: int) -> None:
        self.base_net = BlockGen(in_channels, [self.net_cfg[0]])
        self.box_net = BlockGen(self.base_net.out_channels, [self.net_cfg[1]])
        self.cls_net = BlockGen(self.base_net.out_channels, [self.net_cfg[2]])

    def _load_cfg(self, cfg: BaseConfig) -> ListGen:
        return cfg.head_cfgs(self.box_out, self.cls_out)

    def forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        state = [None] * 3 if state is None else state
        Y, state[0] = self.base_net(X, state[0])
        box, state[1] = self.box_net(Y, state[1])
        cls, state[2] = self.cls_net(Y, state[2])

        return box, cls, state

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        boxes, clses = [], []
        state = None
        for time_step_x in X:
            box, cls, state = self.forward_impl(time_step_x, state)
            boxes.append(box)
            clses.append(cls)
        return torch.stack(boxes), torch.stack(clses)