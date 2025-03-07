"""
Basic object detector class
"""

import torch
from torch import nn
from torch.nn import functional as F

from engine.model import Model
from utils.roi import RoI
import utils.box as box

from .generator import BackboneGen, NeckGen, Head


class SODa(Model):
    """
    Basic object detector class

    Implements the basic functions for calculating losses,
    training the network and generating predictions. The network model is passed
    as a parameter when initializing.
    """

    _start_time = 0

    def __init__(
        self,
        backbone: BackboneGen,
        neck: NeckGen,
        head: Head,
        loss_ratio: int,
        time_window: int = 0,
    ):
        """
        :param backbone: Main network.
        :type backbone: BackboneGen
        :param neck: Feature Map Extraction Network.
        :type neck: NeckGen
        :param head: Network for transforming feature maps into predictions.
        :type head: Head
        :param loss_ratio: The ratio of the loss for non-detection to the loss for false positives.
            The higher this parameter, the more guesses the network generates.
            This is necessary to keep the network active.
        :type loss_ratio: int
        :param time_window: The size of the time window at the beginning of the sequence,
            which can be truncated to a random length. This ensures randomization of the length of
            training sequences and the ability of the network to work with streaming information.
            Defaults to 0.
        :type time_window: int, optional
        """
        super().__init__()
        self.loss_ratio = loss_ratio
        self.base_net = backbone
        self.neck_net = neck
        self.head_net = head
        self.roi_blk = RoI(iou_threshold=0.4)
        self.time_window = time_window

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.parameters(), lr=0.001)

    def loss(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Loss calculation function.

        :param preds: Predictions made by a neural network.
            Contains three tensors:

            1. anchors: Shape [anchor, 4]
            2. cls_preds: Shape [ts, batch, anchor, num_classes + 1]
            3. bbox_preds: Shape [ts, batch, anchor, 4]
        :type preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :param labels: Tensor shape [num_labels, 5].

            One label contains: class id, xlu, ylu, xrd, yrd.
        :type labels: torch.Tensor
        :return: Value of losses
        :rtype: torch.Tensor
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
            preds = self.forward(batch[0][self._start_time :])
            self._start_time = torch.randint(
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
        """Direct network pass

        :param X: Input data.
        :type X: torch.Tensor
        :return: List of three tensors:

            1. Anchors. Shape [anchor, 4].
            2. Class predictions. Shape [ts, batch, anchor, num_classes + 1].
            3. Box predictions. Shape [ts, batch, anchor, 4].
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Y = self.base_net.forward(X)
        fratures_maps = self.neck_net.forward(Y)
        return self.head_net.forward(fratures_maps)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Returns the network's predictions based on the input data

        :param X: Input data.
        :type X: torch.Tensor
        :return: Network Predictions. 
            
            Shape [ts, batch, anchors, 6]. 
            
            One label contains (class, iou, luw, luh, rdw, rdh)
        :rtype: torch.Tensor
        """
        self.eval()
        anchors, cls_preds, bbox_preds = self.forward(X)
        time_stamps = cls_preds.shape[0]
        output = []
        for ts in range(time_stamps):
            cls_probs_ts = F.softmax(cls_preds[ts], dim=2)
            output.append(box.multibox_detection(cls_probs_ts, bbox_preds[ts], anchors))
        return torch.stack(output)
