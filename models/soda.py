"""
Basic object detector class
"""

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import torchmetrics.detection
from typing import Tuple, Optional, List

from utils.roi import RoI
from utils.plotter import Plotter
import utils.box as box
from models.generator import *


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
        in_channels: int = 2,
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
        :param init_weights: If true, apply the weight initialization function. Defaults to True.
        :type init_weights: bool, optional
        :param in_channels: Number of input channels
        :type in_channels: int, optional
        :param plotter: Class for displaying results. Needed for the prediction step.
            Expects to receive a utils.Plotter object. Defaults to None.
        :type plotter: Plotter, optional
        """
        super().__init__()
        self.save_hyperparameters(ignore=["plotter"])
        self.plotter = plotter
        self.roi_blk = RoI(self.hparams.iou_threshold)
        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")
        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            backend="faster_coco_eval",
            compute_on_cpu=False,
            sync_on_compute=False,
            dist_sync_on_step=True,
        )

        """
        The class expects the storage below to store the network results. 
        This should be defined in the child class configuration.
        """
        self.storage_box = Storage()
        self.storage_cls = Storage()
        self.storage_anchor = Storage()

    def _prepare_net(self):
        self.net = ModelGenerator(
            self.get_cfgs(), self.hparams.in_channels, self.hparams.init_weights
        )

    def get_cfgs(self) -> List[LayerGen]:
        """Generates and returns a network configuration

        .. note::
            This method must be overridden in a child class.

        :return: Net configuration.
        :rtype: List[LayerGen]
        """
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = None
        for ts in X:
            preds, state = self._forward_impl(ts, state)
        return preds

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch[0].shape[1],
            sync_dist=True,
        )
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log("test_loss", loss, batch_size=batch[0].shape[1], sync_dist=True)
        self._map_estimate(preds, batch[1])
        return loss

    def on_test_epoch_end(self):
        self._map_compute()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch[0][self._rand_start_time() :])
        loss = self._loss(preds, batch[1])
        self.log("val_loss", loss, batch_size=batch[0].shape[1], sync_dist=True)
        self._map_estimate(preds, batch[1])
        return loss

    def on_validation_epoch_end(self):
        self._map_compute()

    def on_predict_epoch_start(self):
        if self.plotter is None:
            raise RuntimeError(
                "To display predictions, you must initialize the plotter for the model"
            )
        self.plotter.labels = self.trainer.datamodule.get_labels()

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        state = None
        video = []
        input = batch[0][:, 0].squeeze(1)
        for idx, ts in enumerate(input):
            preds, state = self.predict(ts, state)
            preds = None if idx < self.hparams.time_window else preds
            video.append(self.plotter.apply(ts, preds, None))
        video.append(self.plotter.apply(ts, preds, batch[1][0]))
        self.plotter(video, self.trainer.datamodule.hparams.time_step, str(batch_idx))

    def predict(self, X: torch.Tensor, state: ListState | None) -> Tuple[torch.Tensor, ListState]:
        """Prediction method for inference

        .. code-block:: python
            :caption: Example

            state = None
            for ts in cap:
                preds, state = self.predict(ts, state)
                # Does something with predictions

        :param X: Input image from event camera. Expected shape [channel, height, width].
        :type X: torch.Tensor
        :param state: Previous state of the detector or None
        :type state: ListState | None
        :return: Returns a list of two elements:

            1. **Predictions**: A tensor of size [number of predictions, 6],
                where each prediction contains a vector (class id, confidence, lux, luy, rdx, rdy).
            2. **State**: New state of the detector
        :rtype: Tuple[torch.Tensor, ListState]
        """
        preds, state = self._forward_impl(X.unsqueeze(0), state)
        anchors, cls, bbox = preds
        prep_pred = box.multibox_detection(F.softmax(cls, dim=2), bbox, anchors).squeeze(0)
        prep_pred = prep_pred[prep_pred[:, 0] >= 0]
        prep_pred[:, 2:] = torch.clamp(prep_pred[:, 2:], min=0.0, max=1.0)
        return prep_pred, state

    def _forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ListState]:
        _, state = self.net.forward(X, state)

        anchors = self.storage_anchor.get()
        cls_preds = self.storage_cls.get()
        bbox_preds = self.storage_box.get()
        self.storage_box.reset()
        self.storage_cls.reset()

        anchors = torch.cat(anchors)
        cls_preds = self._concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.hparams.num_classes + 1)
        bbox_preds = self._concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], -1, 4)

        return (anchors, cls_preds, bbox_preds), state

    def _flatten_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor so that each pixel retains channels values and smooths each batch"""
        return torch.flatten(torch.permute(pred, (0, 2, 3, 1)), start_dim=1)

    def _concat_preds(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """Concatenating Predictions for Multiple Scales"""
        return torch.cat([self._flatten_pred(p) for p in preds], dim=1)

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

        cls = self.cls_loss.forward(cls_preds.reshape(-1, num_classes), class_labels.reshape(-1))
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
        self.log_dict(
            {
                k: result[k]
                for k in result.keys()
                if k
                in [
                    "map",
                    "map_50",
                    "mar_1",
                    "mar_10",
                    "mar_100",
                ]
            },
        )
        self.map_metric.reset()

    def _map_estimate(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ):
        anchors, cls_preds, bbox_preds = preds
        prep_pred = box.multibox_detection(F.softmax(cls_preds, dim=2), bbox_preds, anchors)
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
