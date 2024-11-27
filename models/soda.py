import torch
from torch import nn
from torch.nn import functional as F
from typing import Union

from engine.model import Module
from utils.roi import RoI
import utils.box as box

from .backbone.vgg import VGGBackbone
from .neck.ssd import SSDNeck
from .head import Head


class SODa(Module):
    """
    Spike Object Detector.
    """

    start_time = 0

    def __init__(
        self,
        backbone: Union[VGGBackbone],
        neck: Union[SSDNeck],
        num_classes: int,
        loss_ratio: int,
        time_window: int = 0,
    ):
        super().__init__()
        self.loss_ratio = loss_ratio
        self.base_net = backbone
        self.neck_net = neck
        self.head_net = Head(num_classes, neck.out_shape)
        self.roi_blk = RoI(iou_threshold=0.3)
        self.time_window = time_window

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.parameters(), lr=0.001)
        # return torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=5e-4)

    def loss(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            preds (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                anchors (torch.Tensor): [all_anchors, 4]
                cls_preds (torch.Tensor): [ts, num_batch, all_anchors,(num_classes + 1)]
                bbox_preds (torch.Tensor): [ts, num_batch, all_anchors * 4]
            labels [torch.Tensor]:
                Tensor shape [num_labels, 5]
                    One label contains: [class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        Returns:
            torch.Tensor: loss
        """
        anchors, ts_cls_preds, ts_bbox_preds = preds
        bbox_offset, bbox_mask, class_labels = self.roi_blk(anchors, labels)
        _, _, _, num_classes = ts_cls_preds.shape

        cls = self.cls_loss.forward(
            ts_cls_preds[-1].reshape(-1, num_classes), class_labels.reshape(-1)
        )
        bbox = self.box_loss.forward(
            ts_bbox_preds[-1] * bbox_mask, bbox_offset * bbox_mask
        )

        mask = class_labels.reshape(-1) > 0
        gt_loss = cls[mask].mean()
        background_loss = cls[~mask].mean()

        return (
            gt_loss * self.loss_ratio
            + background_loss * (1 - self.loss_ratio)
            + bbox.mean()
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.time_window:
            preds = self.forward(batch[0][self.start_time :])
            self.start_time = torch.randint(
                0, self.time_window, (1,), requires_grad=False, dtype=torch.uint32
            )
        else:
            preds = self.forward(batch[0])
        loss = self.loss(preds, batch[1])
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.training_step(batch)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.training_step(batch)

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X (torch.Tensor): Real img
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                anchors (torch.Tensor): [all_anchors, 4]
                cls_preds (torch.Tensor): [ts, num_batch, all_anchors,(num_classes + 1)]
                bbox_preds (torch.Tensor): [ts, num_batch, all_anchors * 4]
        """
        Y = self.base_net.forward(X)
        fratures_maps = self.neck_net.forward(Y)
        return self.head_net.forward(fratures_maps)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): img batch
        Returns:
            torch.Tensor: Shape [ts, batch, anchors, 6].
                One label contains [class, iou, luw, luh, rdw, rdh]
        """
        self.eval()
        anchors, cls_preds, bbox_preds = self.forward(X)
        time_stamps = cls_preds.shape[0]
        output = []
        for ts in range(time_stamps):
            cls_probs_ts = F.softmax(cls_preds[ts], dim=2)
            output.append(box.multibox_detection(cls_probs_ts, bbox_preds[ts], anchors))
        return torch.stack(output)
