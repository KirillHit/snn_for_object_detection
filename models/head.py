import torch
from torch import nn
import norse.torch as snn
from norse.torch.functional.leaky_integrator import LIParameters
from typing import List

from utils.anchors import AnchorGenerator


class Head(nn.Module):
    def __init__(self, num_classes: int, in_shape: List[int]) -> None:
        super().__init__()
        self.num_classes = num_classes

        sizes = (
            [0.062, 0.078, 0.094],
            [0.125, 0.156, 0.188],
            [0.250, 0.312, 0.375],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
            [0.500, 0.625, 0.750],
        )
        ratios = (0.7, 1, 1.3)

        num_anchors = len(sizes[0]) + len(ratios) - 1
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
                f"decoder_{idx}",
                Decoder(channels, num_box_out, num_class_out, 3, True),
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
            boxes, classes = getattr(self, f"decoder_{idx}")(map)
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


class Decoder(nn.Module):
    def __init__(
        self, in_channels: int, box_out: int, cls_out: int, kernel_size: int, train_li_layer = False
    ) -> None:
        super().__init__()

        self.decoder = self.make_decoder(in_channels, kernel_size, train_li_layer)

        self.box_preds = snn.Lift(nn.Conv2d(in_channels, box_out, kernel_size=1))
        self.cls_preds = snn.Lift(nn.Conv2d(in_channels, cls_out, kernel_size=1))

    def make_decoder(
        self, in_channels: int, kernel_size: int, train_li_layer: bool
    ) -> snn.SequentialState:
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        layers: List[nn.Module] = []
        conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        li_params = LIParameters()
        if train_li_layer:
            self.li_grad_params = torch.nn.Parameter(li_params.tau_mem_inv)
        layers += [snn.Lift(conv2d), snn.LI(p=li_params)]
        
        return snn.SequentialState(*layers)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X (torch.Tensor): img [ts, batch, in_channels, h, w]
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                boxes (torch.Tensor): predicted boxes [ts, batch, box_out, h, w]
                classes (torch.Tensor): predicted classes [ts, batch, cls_out, h, w]
        """
        Y = self.decoder(X)
        return self.box_preds(Y), self.cls_preds(Y)
