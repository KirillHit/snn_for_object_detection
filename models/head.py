"""
Model head generator
"""

import torch
from torch import nn
from typing import List, Tuple, Dict
from models.modules import *

from utils.anchors import AnchorGenerator


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
                Conv(kernel_size=1),
                Conv(kernel_size=1),
                Conv(box_out, 1),
            ],
            [
                Conv(kernel_size=1),
                Conv(kernel_size=1),
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

    def _load_cfg(self):
        self.default_cfgs.update(main_cfg(self.box_out, self.cls_out))

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


#####################################################################
#                       Model configurations                        #
#####################################################################


def main_cfg(box_out: int, cls_out: int) -> Dict[str, ListGen]:
    """Default configuration generator

    See source code.

    :return: Lists of layer generators.
    :rtype: Dict[str, ListGen]
    """
    cfgs: Dict[str, ListGen] = {
        "main": [
            [
                Conv(kernel_size=1),
                Norm(),
                LSTM(),
            ],
            [
                Conv(kernel_size=1),
                Conv(kernel_size=1),
                Conv(box_out, 1),
            ],
            [
                Conv(kernel_size=1),
                Conv(kernel_size=1),
                Conv(cls_out, 1),
            ],
        ],
    }
    return cfgs
