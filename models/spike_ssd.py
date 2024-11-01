import torch
from torch import nn
from torch.nn import functional as F
import snntorch as snn

from engine.model import Module
from utils.anchors import AnchorGenerator
from utils.roi import RoI
import utils.box as box


class SpikeSSD(Module):
    """Simple Single Shot Multibox Detection"""

    def __init__(self, num_classes):
        super().__init__()
        self.base_net = SpikeCNN()
        self.fpn_blk = SpikeFPN(num_classes)
        self.roi_blk = RoI(iou_threshold=0.3)

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        #return torch.optim.Adamax(self.parameters(), lr=0.001)
        return torch.optim.SGD(self.parameters(), lr=0.002)

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
                (max(ts_list.shape[0], 1)),
                dtype=ts_cls_preds.dtype,
                device=ts_cls_preds.device,
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
                loss_ts[ts_idx] = cls + bbox
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
        Y = self.base_net.forward(X)
        return self.fpn_blk.forward(Y)

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


class SpikeCNN(nn.Module):
    """Convolutional neural network for extracting features from images"""

    def __init__(self) -> None:
        super().__init__()
        num_filters = [2, 8, 32, 64]
        blk = [
            SpikeDownSampleBlk(num_filters[i], num_filters[i + 1])
            for i in range(len(num_filters) - 1)
        ]
        self.cnn_net = nn.Sequential(*blk)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.cnn_net(X)


class SpikeDownSampleBlk(nn.Module):
    """Reduces the height and width of input feature maps by half"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lif = snn.Leaky(beta=0.85)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = self.lif.init_leaky()

        spk_rec = []
        # mem_rec = []

        for step in range(X.shape[0]):
            x_conv = self.conv(X[step])
            spk, mem = self.lif(x_conv, mem)
            spk_rec.append(spk)
            # mem_rec.append(mem)

        return torch.stack(spk_rec)


class SpikeFPN(nn.Module):
    """Feature Pyramid Networks for identifying features of different scales"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.low_layer = SpikeDownSampleBlk(64, 128)
        self.mid_layer = SpikeDownSampleBlk(128, 128)
        self.high_layer = SpikeDownSampleBlk(128, 128)

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
        self.base_pred = DetectorDirectDecoder(64, num_box_out, num_class_out, 3)
        self.low_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)
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
        self.li = snn.Leaky(beta=0.85, reset_mechanism="none")
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
        mem = self.li.init_leaky()
        boxes = []
        classes = []
        for ts in range(X.shape[0]):
            z = self.conv(X[ts])
            z, mem = self.li(z, mem)
            boxes.append(self.box_preds(mem))
            classes.append(self.cls_preds(mem))
        return torch.stack(boxes), torch.stack(classes)
