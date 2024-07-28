import torch
from torch import nn
import norse.torch as norse
from torch.nn import functional as F

from engine.model import Module
from utils.anchors import AnchorGenerator
from utils.roi import RoI
import utils.box as box


class SpikeYOLO(Module):
    """Simple Single Shot Multibox Detection"""

    def __init__(self, num_classes, seq_length=16):
        super().__init__()
        self.encoder = DirectEncoder(seq_length)
        self.base_net = SpikeCNN()
        self.fpn_blk = SpikeFPN(num_classes)
        self.roi_blk = RoI(iou_threshold=0.5)

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=0.002)
        #return torch.optim.SGD(self.parameters(), lr=0.2, weight_decay=5e-4)

    def loss(self, y_hat, y):
        """
        Args:
            y_hat: preds
            y: true
        """
        cls_preds, bbox_preds = y_hat
        bbox_offset, bbox_mask, class_labels = y

        batch_size, _, num_classes = cls_preds.shape
        cls = torch.reshape(
            self.cls_loss(cls_preds.reshape(-1, num_classes), class_labels.reshape(-1)),
            (batch_size, -1),
        ).mean(dim=1)
        bbox = torch.reshape(
            self.box_loss(bbox_preds * bbox_mask, bbox_offset * bbox_mask),
            (batch_size, -1),
        ).mean(dim=1)
        return cls + bbox

    def training_step(self, batch):
        anchors, cls_preds, bbox_preds = self(batch[0])
        y = self.roi_blk(anchors, batch[1])
        loss = self.loss((cls_preds, bbox_preds), y)
        return loss.mean()

    def validation_step(self, batch):
        return self.training_step(batch)

    def forward(self, X):
        """
        Args:
            X: Real img

        Returns:
            anchors: [all_anchors, 4]
            cls_preds: [num_batch, all_anchors,(num_classes + 1)]
            bbox_preds: [num_batch, all_anchors * 4]
        """
        Y = self.encoder(X)
        Y, state = self.base_net(Y)
        return self.fpn_blk(Y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): img batch
            threshold (_type_): TODO

        Returns:
            torch.Tensor: [class, roi, luw, luh, rdw, rdh]
        """
        self.eval()
        anchors, cls_preds, bbox_preds = self(X)
        cls_probs = F.softmax(cls_preds, dim=2)
        output = box.multibox_detection(cls_probs, bbox_preds, anchors)
        return output


class SpikeDownSampleBlk(nn.Module):
    """Reduces the height and width of input feature maps by half"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lif1 = norse.LIFCell()
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.lif2 = norse.LIFCell()

    def forward(self, X):
        s1 = s2 = None
        zs = []
        for ts in range(X.shape[0]):
            z = self.conv1(X[ts])
            z, s1 = self.lif1(z, s1)
            #z = self.conv2(z)
            #z, s2 = self.lif2(z, s2)
            z = nn.functional.max_pool2d(z, kernel_size=2, stride=2)
            zs.append(z)
        Y = torch.stack(zs)
        return Y


class SpikeFPN(nn.Module):
    """Feature Pyramid Networks for identifying features of different scales"""

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.low_layer = SpikeDownSampleBlk(64, 128)
        #self.mid_layer = SpikeDownSampleBlk(128, 128)
        #self.high_layer = SpikeDownSampleBlk(128, 128)

        sizes = (
            [0.2, 0.272],
            [0.37, 0.447],
            [0.54, 0.619],
            [0.71, 0.79],
        )
        ratios = (0.5, 1, 1.5)
        self.base_anchors = AnchorGenerator(sizes=sizes[0], ratios=ratios)
        self.low_anchors = AnchorGenerator(sizes=sizes[1], ratios=ratios)
        #self.mid_anchors = AnchorGenerator(sizes=sizes[2], ratios=ratios)
        #self.high_anchors = AnchorGenerator(sizes=sizes[3], ratios=ratios)

        num_anchors = len(sizes[0]) + len(ratios) - 1

        num_class_out = num_anchors * (self.num_classes + 1)
        num_box_out = num_anchors * 4
        self.base_pred = DetectorDirectDecoder(64, num_box_out, num_class_out, 3)
        self.low_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)
        #self.mid_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)
        #self.high_pred = DetectorDirectDecoder(128, num_box_out, num_class_out, 3)

    def forward(self, X):
        """
        Args:
            X: Feature map

        Returns:
            anchors: [num_anchors, 4]
            cls_preds: [num_batch, all_anchors, (num_classes + 1)]
            bbox_preds: [num_batch, all_anchors, 4]
        """
        low_feature_map = self.low_layer(X)
        #mid_feature_map = self.mid_layer(low_feature_map)
        #high_feature_map = self.high_layer(mid_feature_map)

        anchors, cls_preds, bbox_preds = [], [], []

        anchors.append(self.base_anchors(X))
        boxes, classes = self.base_pred(X)
        bbox_preds.append(boxes)
        cls_preds.append(classes)

        anchors.append(self.low_anchors(low_feature_map))
        boxes, classes = self.low_pred(low_feature_map)
        bbox_preds.append(boxes)
        cls_preds.append(classes)

        anchors = torch.cat(anchors, dim=0)
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], -1, 4)
        return anchors, cls_preds, bbox_preds

    def flatten_pred(self, pred: torch.Tensor):
        """Transforms the tensor so that each pixel retains channels values and smooths each batch"""
        return torch.flatten(torch.permute(pred, (0, 2, 3, 1)), start_dim=1)

    def concat_preds(self, preds):
        """Concatenating Predictions for Multiple Scales"""
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)


class SpikeCNN(nn.Module):
    """Convolutional neural network for extracting features from images"""

    def __init__(self):
        super().__init__()
        blk = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            blk.append(SpikeDownSampleBlk(num_filters[i], num_filters[i + 1]))
        self.cnn_net = norse.SequentialState(*blk)

    def forward(self, X):
        return self.cnn_net(X)


class DirectEncoder(nn.Module):
    def __init__(self, seq_length: int) -> None:
        super().__init__()
        self.encode = norse.ConstantCurrentLIFEncoder(seq_length=seq_length)

    def forward(self, X: torch.Tensor):
        Y = self.encode(X)

        return Y


class DetectorDirectDecoder(nn.Module):
    def __init__(
        self, in_channels: int, box_out: int, cls_out, kernel_size: int
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.li = norse.LICell()

        self.box_preds = nn.Conv2d(in_channels, box_out, kernel_size=1)
        self.cls_preds = nn.Conv2d(in_channels, cls_out, kernel_size=1)

    def forward(self, X: torch.Tensor):
        """
        Returns:
            boxes: predicted boxes
            classes: predicted classes
        """
        s1 = None
        for ts in range(X.shape[0]):
            z = self.conv(X[ts])
            z, s1 = self.li(z, s1)

        box = self.box_preds(z)
        cls = self.cls_preds(z)

        return box, cls
