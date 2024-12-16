import torch
from torch import nn
from typing import List, Tuple, Dict
from models.modules import *

from utils.anchors import AnchorGenerator


#####################################################################
#                          Head descriptor                          #
#       Applies a head model to multiple maps and merges them       #
#####################################################################


class Head(nn.Module):
    def __init__(
        self,
        cfg: str | ListGen,
        num_classes: int,
        in_shape: List[int],
        init_weights=False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

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
        """
        Args:
            X (List[torch.Tensor]): feature map list. Map shape - [ts, batch, in_channels, h, w]
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                anchors (torch.Tensor): [num_anchors, 4]
                cls_preds (torch.Tensor): [ts, num_batch, all_anchors, (num_classes + 1)]
                bbox_preds (torch.Tensor): [ts, num_batch, all_anchors, 4]
        """

        anchors, cls_preds, bbox_preds = [], [], []

        for idx, map in enumerate(X):
            anchors.append(getattr(self, f"anchor_gen_{idx}")(map))
            boxes, classes = getattr(self, f"model_{idx}")(map)
            bbox_preds.append(boxes)
            cls_preds.append(classes)

        anchors = torch.cat(anchors, dim=0)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], cls_preds.shape[1], -1, self.num_classes + 1
        )
        bbox_preds = self.concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], bbox_preds.shape[1], -1, 4)
        return anchors, cls_preds, bbox_preds

    def flatten_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor so that each pixel retains channels values and smooths each batch"""
        return torch.flatten(torch.permute(pred, (0, 1, 3, 4, 2)), start_dim=2)

    def concat_preds(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """Concatenating Predictions for Multiple Scales"""
        return torch.cat([self.flatten_pred(p) for p in preds], dim=2)


#####################################################################
#                          Head Generator                           #
#####################################################################


class HeadGen(ModelGen):
    def __init__(
        self,
        cfg: str | ListGen,
        box_out: int,
        cls_out: int,
        in_channels: int = 2,
        init_weights=False,
    ):
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
        self, X: List[torch.Tensor], state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        state = [None] * 3 if state is None else state
        Y, state[0] = self.base_net(X, state[0])
        box, state[1] = self.box_net(Y, state[1])
        cls, state[2] = self.cls_net(Y, state[2])

        return box, cls, state

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): img [ts, batch, in_channels, h, w]
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                boxes (torch.Tensor): predicted boxes [ts, batch, box_out, h, w]
                classes (torch.Tensor): predicted classes [ts, batch, cls_out, h, w]
        """
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
    fun = SELU
    cfgs: Dict[str, ListGen] = {
        "main": [
            [Conv(None, 1), Norm(), LI()],
            [Conv(None, 1), Norm(True), fun(), Conv(box_out, 1)],
            [Conv(None, 1), Norm(True), fun(), Conv(cls_out, 1)],
        ],
    }
    return cfgs
