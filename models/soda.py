"""
Basic object detector class
"""

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import torchmetrics.detection

from utils.roi import RoI
import utils.box as box
from models.generator import ListGen, BackboneGen, NeckGen, Head


class SODa(L.LightningModule):
    """
    Basic object detector class

    Implements the basic functions for calculating losses,
    training the network and generating predictions. The network model is passed
    as a parameter when initializing.

    .. warning::

        This class can only be used as a base class for inheritance.
    """

    def __init__(
        self,
        num_classes: int,
        loss_ratio: float = 0.04,
        time_window: int = 0,
        iou_threshold: float = 0.4,
        learning_rate: float = 0.001,
        state_storage: bool = False,
        init_weights: bool = True,
    ):
        """
        :param num_classes: Number of classes.
        :type num_classes: int
        :param loss_ratio: The ratio of the loss for non-detection to the loss for false positives.
            The higher this parameter, the more guesses the network generates.
            This is necessary to keep the network active. Defaults to 0.04.
        :type loss_ratio: int, optional
        :param time_window: The size of the time window at the beginning of the sequence,
            which can be truncated to a random length. This ensures randomization of the length of
            training sequences and the ability of the network to work with streaming information.
            Defaults to 0.
        :type time_window: int, optional
        :param iou_threshold: #TODO
        :type iou_threshold: int, optional
        :param learning_rate: #TODO
        :type learning_rate: int, optional
        :param state_storage: If true preserves preserves all intermediate states of spiking neurons.
            Necessary for analyzing the network operation. Defaults to False.
        :type state_storage: bool, optional
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        """
        super().__init__()
        self.save_hyperparameters()

        self.base_net = BackboneGen(
            self.backbone_cfgs,
            in_channels=2,
            init_weights=self.hparams.init_weights,
        )
        self.neck_net = NeckGen(
            self.neck_cfgs,
            self.base_net.out_channels,
            init_weights=self.hparams.init_weights,
        )
        self.head_net = Head(
            self.head_cfgs,
            self.hparams.num_classes,
            self.neck_net.out_shape,
            init_weights=self.hparams.init_weights,
        )
        self.roi_blk = RoI(self.hparams.iou_threshold)
        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
        )

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.parameters(), lr=self.hparams.learning_rate)

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
        :param labels: Tensor shape [batch, num_labels, 5].

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
            gt_loss * self.hparams.loss_ratio
            + background_loss * (1 - self.hparams.loss_ratio)
            + bbox.mean()
        )

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

    def _rand_start_time(self) -> int:
        return (
            torch.randint(
                0,
                self.hparams.time_window,
                (1,),
                requires_grad=False,
                dtype=torch.uint32,
            )
            if self.hparams.time_window
            else 0
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self.loss(preds, batch[1])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self.loss(preds, batch[1])
        self.log("test_loss", loss, prog_bar=True)
        self._map_estimate(preds, batch[1])
        return loss

    def on_test_epoch_end(self):
        self._map_compute()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self.loss(preds, batch[1])
        self.log("val_loss", loss, prog_bar=True)
        self._map_estimate(preds, batch[1])
        return loss

    def on_validation_epoch_end(self):
        self._map_compute()

    def _map_compute(self):
        result = self.map_metric.compute()
        self.log_dict({k: result[k] for k in result.keys() if k in ["map", "map_50"]})
        self.map_metric.reset()

    def _map_estimate(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ):
        """#TODO
        :param preds: Predictions made by a neural network.
            Contains three tensors:

            1. anchors: Shape [anchor, 4]
            2. cls_preds: Shape [ts, batch, anchor, num_classes + 1]
            3. bbox_preds: Shape [ts, batch, anchor, 4]
        :type preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :param labels: Tensor shape [batch, num_labels, 5].

            One label contains: class id, xlu, ylu, xrd, yrd.
        :type labels: torch.Tensor
        """
        anchors, cls_preds, bbox_preds = preds
        prep_pred = box.multibox_detection(
            F.softmax(cls_preds[-1], dim=2), bbox_preds[-1], anchors
        )
        map_preds = []
        map_target = []
        for batch, label in zip(prep_pred, labels):
            masked_preds = batch[batch[:, 0] >= 0]
            map_preds.append(
                {
                    "boxes": masked_preds[:, 2:],
                    "scores": masked_preds[:, 1],
                    "labels": masked_preds[:, 0].type(torch.IntTensor),
                }
            )
            masked_label = label[label[:, 0] >= 0]
            map_target.append(
                {
                    "boxes": masked_label[:, 1:],
                    "labels": masked_label[:, 0].type(torch.IntTensor),
                }
            )
        self.map_metric.update(map_preds, map_target)

    def predict(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns the network's predictions based on the input data

        :param batch: Input data.
        :type batch: torch.Tensor
        :return: Network Predictions.

            Shape [ts, batch, anchors, 6].

            One label contains (class, iou, luw, luh, rdw, rdh)
        :rtype: torch.Tensor
        """
        anchors, cls_preds, bbox_preds = self.forward(batch)
        time_stamps = cls_preds.shape[0]
        output = []
        for ts in range(time_stamps):
            cls_probs_ts = F.softmax(cls_preds[ts], dim=2)
            output.append(box.multibox_detection(cls_probs_ts, bbox_preds[ts], anchors))
        return torch.stack(output)
    
    def predict_stated(self, batch: tuple[torch.Tensor, torch.Tensor], state) :
        #TODO
        pass

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        #TODO
        pass
