"""
Basic object detector class
"""

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import torchmetrics.detection
from typing import Tuple, Optional

from utils.roi import RoI
from utils.plotter import Plotter
import utils.box as box
from models.generator import ListGen, BackboneGen, NeckGen, Head, ListState


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
        time_window: int = 16,
        iou_threshold: float = 0.4,
        learning_rate: float = 0.001,
        state_storage: bool = False,
        init_weights: bool = True,
        plotter: Optional[Plotter] = None,
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
        :param iou_threshold: Minimum acceptable iou. Defaults to 0.4.
        :type iou_threshold: float, optional
        :param learning_rate: Learning rate. Defaults to 0.001.
        :type learning_rate: float, optional
        :param state_storage: If true preserves preserves all intermediate states of spiking neurons.
            Necessary for analyzing the network operation. Defaults to False.
        :type state_storage: bool, optional
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        :param plotter: #TODO
        :type plotter: Plotter, optional
        """
        super().__init__()
        self.save_hyperparameters(ignore=["plotter"])
        self.plotter = plotter

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

        .. note::
            This method must be overridden in a child class.

        :return: Backbone configuration.
        :rtype: ListGen
        """
        raise NotImplementedError

    def neck_cfgs(self) -> ListGen:
        """Generates and returns a network neck configuration

        .. note::
            This method must be overridden in a child class.

        :return: Neck configuration.
        :rtype: ListGen
        """
        raise NotImplementedError

    def head_cfgs(self, box_out: int, cls_out: int) -> ListGen:
        """Generates and returns a network head configuration

        .. note::
            This method must be overridden in a child class.

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

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = None
        for ts in X:
            preds, state = self._forward_impl(ts, state)
        return preds

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log("train_loss", loss, prog_bar=True, batch_size=batch[0].shape[1])
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log("test_loss", loss, batch_size=batch[0].shape[1])
        self._map_estimate(preds, batch[1])
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log("val_loss", loss, batch_size=batch[0].shape[1])
        self._map_estimate(preds, batch[1])
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if self.plotter is None:
            return
        else:
            self.plotter.labels = self.trainer.datamodule.get_labels()
        self.plotter.labels = self.trainer.datamodule.get_labels()
        state = None
        video = []
        input = batch[0][:, 0].squeeze(1)
        for idx, ts in enumerate(input):
            preds, state = self.predict(ts, state)
            preds = None if idx < self.hparams.time_window else preds
            video.append(self.plotter.apply(ts, preds, None))
        video.append(self.plotter.apply(ts, preds, batch[1][0]))
        self.plotter(video, self.trainer.datamodule.hparams.time_step)

    def on_test_epoch_end(self):
        self._map_compute()

    def on_validation_epoch_end(self):
        self._map_compute()

    def predict(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        """
        Shape X [c, h, w]
        #TODO
        """
        preds, state = self._forward_impl(X.unsqueeze(0), state)
        anchors, cls, bbox = preds
        prep_pred = box.multibox_detection(
            F.softmax(cls, dim=2), bbox, anchors
        ).squeeze(0)
        prep_pred = prep_pred[prep_pred[:, 0] >= 0]
        prep_pred[:, 2:] = torch.clamp(prep_pred[:, 2:], min=0.0, max=1.0)
        return prep_pred, state

    def _forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ListState]:
        state = [None] * 3 if state is None else state
        base_out, state[0] = self.base_net.forward(X, state[0])
        neck_out, state[1] = self.neck_net.forward(base_out, state[1])
        anchors, cls_preds, bbox_preds, state[2] = self.head_net.forward(
            neck_out, state[2]
        )
        return (anchors, cls_preds, bbox_preds), state

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

    def _loss(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        anchors, cls_preds, bbox_preds = preds
        bbox_offset, bbox_mask, class_labels = self.roi_blk(anchors, labels)
        _, _, num_classes = cls_preds.shape

        cls = self.cls_loss.forward(
            cls_preds.reshape(-1, num_classes), class_labels.reshape(-1)
        )
        bbox = self.box_loss.forward(bbox_preds * bbox_mask, bbox_offset * bbox_mask)

        mask = class_labels.reshape(-1) > 0
        gt_loss = cls[mask].mean()
        background_loss = cls[~mask].mean()

        return (
            gt_loss * self.hparams.loss_ratio
            + background_loss * (1 - self.hparams.loss_ratio)
            + bbox.mean()
        )

    def _map_compute(self):
        result = self.map_metric.compute()
        self.log_dict({k: result[k] for k in result.keys() if k in ["map", "map_50"]})
        self.map_metric.reset()

    def _map_estimate(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ):
        anchors, cls_preds, bbox_preds = preds
        prep_pred = box.multibox_detection(
            F.softmax(cls_preds, dim=2), bbox_preds, anchors
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
