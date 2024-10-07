import torch
from torch import nn
import norse.torch as norse
from torch.nn import functional as F
from tqdm import tqdm

from engine.model import Module
from utils.anchors import AnchorGenerator
from utils.roi import RoI
import utils.box as box


class SpikeYOLO(Module):
    """Simple Single Shot Multibox Detection"""

    def __init__(self, num_classes):
        super().__init__()
        self.base_net = SpikeCNN()
        self.fpn_blk = SpikeFPN(num_classes)
        self.roi_blk = RoI(iou_threshold=0.3)

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.parameters(), lr=0.002)
        # return torch.optim.SGD(self.parameters(), lr=0.2, weight_decay=5e-4)

    def loss(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels_batch: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            preds (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                anchors (torch.Tensor): [all_anchors, 4]
                cls_preds (torch.Tensor): [ts, num_batch, all_anchors,(num_classes + 1)]
                bbox_preds (torch.Tensor): [ts, num_batch, all_anchors * 4]
            labels_batch (list[torch.Tensor]):
                The length of the list is equal to the number of butch
                One label contains (ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        Returns:
            torch.Tensor: loss
        """
        anchors, ts_cls_preds, ts_bbox_preds = preds
        _, batch_size, _, _ = ts_cls_preds.shape
        loss = torch.zeros(
            (batch_size), dtype=ts_cls_preds.dtype, device=ts_cls_preds.device
        )
        for batch_idx, labels in enumerate(labels_batch):
            ts_list: torch.Tensor = torch.unique(labels[..., 0])
            loss_ts = torch.zeros(
                (ts_list.shape[0]), dtype=ts_cls_preds.dtype, device=ts_cls_preds.device
            )
            for ts_idx, ts in enumerate(ts_list):
                masked_labels = labels[..., 1:]
                masked_labels = masked_labels[labels[..., 0] == ts]
                bbox_offset, bbox_mask, class_labels = self.roi_blk.target(
                    anchors, masked_labels
                )
                cls = self.cls_loss.forward(
                    ts_cls_preds[ts.type(torch.uint32), batch_idx],
                    class_labels,
                ).mean()
                bbox = self.box_loss.forward(
                    ts_bbox_preds[ts.type(torch.uint32), batch_idx] * bbox_mask,
                    bbox_offset * bbox_mask,
                ).mean()
                loss_ts[ts_idx] += cls + bbox
            loss[batch_idx] = loss_ts.mean()
        return loss.mean()

    def training_step(
        self, batch: tuple[torch.Tensor, list[torch.Tensor]]
    ) -> torch.Tensor:
        preds = self.forward(batch[0])
        loss = self.loss(preds, batch[1])
        return loss

    def test_step(self, batch: tuple[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        return self.training_step(batch)

    def validation_step(
        self, batch: tuple[torch.Tensor, list[torch.Tensor]]
    ) -> torch.Tensor:
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
        Y, _ = self.base_net.forward(X)
        return self.fpn_blk.forward(Y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): img batch
        Returns:
            torch.Tensor: [class, roi, luw, luh, rdw, rdh]
        """
        self.eval()
        anchors, cls_preds, bbox_preds = self.forward(X)
        cls_probs = F.softmax(cls_preds, dim=2)
        output = box.multibox_detection(cls_probs, bbox_preds, anchors)
        return output


class SpikeCNN(nn.Module):
    """Convolutional neural network for extracting features from images"""

    def __init__(self) -> None:
        super().__init__()
        num_filters = [2, 4, 16, 32]
        blk = [
            SpikeDownSampleBlk(num_filters[i], num_filters[i + 1])
            for i in range(len(num_filters) - 1)
        ]
        self.cnn_net = norse.SequentialState(*blk)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cnn_net(X)


class SpikeDownSampleBlk(nn.Module):
    """Reduces the height and width of input feature maps by half"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lif1 = norse.LIFCell()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        s1 = None
        zs = []
        for ts in range(X.shape[0]):
            z = self.conv1(X[ts])
            z, s1 = self.lif1(z, s1)
            z = nn.functional.max_pool2d(z, kernel_size=2, stride=2)
            zs.append(z)
        Y = torch.stack(zs)
        return Y


class SpikeFPN(nn.Module):
    """Feature Pyramid Networks for identifying features of different scales"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.low_layer = SpikeDownSampleBlk(32, 64)
        self.mid_layer = SpikeDownSampleBlk(64, 128)
        self.high_layer = SpikeDownSampleBlk(128, 128)

        """ sizes = (
            [0.010, 0.024, 0.044],
            [0.010, 0.034, 0.068],
            [0.044, 0.068, 0.154],
            [0.068, 0.154, 0.274],
        )
        ratios = ([0.38, 0.3, 0.5], [1.4, 1.0, 2.1], [0.38, 0.3, 0.5], [1.4, 1.0, 2.1]) """

        sizes = (
            [0.062, 0.078, 0.094],
            [0.125, 0.156, 0.188],
            [0.250, 0.312, 0.375],
            [0.500, 0.625, 0.750],
        )
        ratios = (0.7, 1, 1.3)
        self.base_anchors = AnchorGenerator(sizes=sizes[0], ratios=ratios)
        self.low_anchors = AnchorGenerator(sizes=sizes[1], ratios=ratios)
        self.mid_anchors = AnchorGenerator(sizes=sizes[2], ratios=ratios)
        self.high_anchors = AnchorGenerator(sizes=sizes[3], ratios=ratios)

        num_anchors = len(sizes[0]) + len(ratios) - 1

        num_class_out = num_anchors * (self.num_classes + 1)
        num_box_out = num_anchors * 4
        self.base_pred = DetectorDirectDecoder(32, num_box_out, num_class_out, 3)
        self.low_pred = DetectorDirectDecoder(64, num_box_out, num_class_out, 3)
        self.mid_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)
        self.high_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X (torch.Tensor): feature map [ts, batch, in_channels, h, w]
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                anchors (torch.Tensor): [num_anchors, 4]
                cls_preds (torch.Tensor): [ts, num_batch, all_anchors, (num_classes + 1)]
                bbox_preds (torch.Tensor): [ts, num_batch, all_anchors, 4]
        """
        low_feature_map = self.low_layer(X)
        mid_feature_map = self.mid_layer(low_feature_map)
        high_feature_map = self.high_layer(mid_feature_map)

        anchors, cls_preds, bbox_preds = [], [], []

        anchors.append(self.base_anchors(X))
        boxes, classes = self.base_pred(X)
        bbox_preds.append(boxes)
        cls_preds.append(classes)

        anchors.append(self.low_anchors(low_feature_map))
        boxes, classes = self.low_pred(low_feature_map)
        bbox_preds.append(boxes)
        cls_preds.append(classes)

        anchors.append(self.mid_anchors(mid_feature_map))
        boxes, classes = self.mid_pred(mid_feature_map)
        bbox_preds.append(boxes)
        cls_preds.append(classes)

        anchors.append(self.high_anchors(high_feature_map))
        boxes, classes = self.high_pred(high_feature_map)
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


class DetectorDirectDecoder(nn.Module):
    def __init__(
        self, in_channels: int, box_out: int, cls_out: int, kernel_size: int
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.li = norse.LICell()
        self.box_preds = nn.Conv2d(in_channels, box_out, kernel_size=1)
        self.cls_preds = nn.Conv2d(in_channels, cls_out, kernel_size=1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X (torch.Tensor): img [ts, batch, in_channels, h, w]
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                boxes (torch.Tensor): predicted boxes [ts, batch, box_out, h, w]
                classes (torch.Tensor): predicted classes [ts, batch, cls_out, h, w]
        """
        s1 = None
        boxes = []
        classes = []
        for ts in range(X.shape[0]):
            z = self.conv(X[ts])
            z, s1 = self.li(z, s1)
            boxes.append(self.box_preds(z))
            classes.append(self.cls_preds(z))
        return torch.stack(boxes), torch.stack(classes)
