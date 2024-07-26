import torch
from torch import nn
from torch.nn import functional as F

from engine.model import Module
from utils.anchors import AnchorGenerator
from utils.roi import RoI
import utils.box as box


class YOLO(Module):
    """Simple Single Shot Multibox Detection"""

    def __init__(self, num_classes):
        super().__init__()
        self.base_net = CNN()
        self.fpn_blk = FPN(num_classes)
        self.roi_blk = RoI(iou_threshold=0.5)

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.2, weight_decay=5e-4)

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
        Y = self.base_net(X)
        return self.fpn_blk(Y)

    def predict(self, X, threshold):
        self.eval()
        anchors, cls_preds, bbox_preds = self(X)
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = box.multibox_detection(cls_probs, bbox_preds, anchors)
        return output


class DownSampleBlk(nn.Module):
    """Reduces the height and width of input feature maps by half"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*blk)

    def forward(self, X):
        return self.net(X)


class FPN(nn.Module):
    """Feature Pyramid Networks for identifying features of different scales"""

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.low_layer = DownSampleBlk(64, 128)
        self.mid_layer = DownSampleBlk(128, 128)
        self.high_layer = DownSampleBlk(128, 128)

        self.low_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.mid_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.high_conv = nn.Conv2d(128, 64, kernel_size=1)
        
        sizes = ((0.038, 0.039, 0.40)), (0.041, 0.042, 0.43), (0.044, 0.045, 0.46)
        ratios = (0.7, 1, 1.5)
        self.low_anchors = AnchorGenerator(sizes=sizes[0], ratios=ratios)
        self.mid_anchors = AnchorGenerator(sizes=sizes[1], ratios=ratios)
        self.high_anchors = AnchorGenerator(sizes=sizes[2], ratios=ratios)

        num_anchors = len(sizes[0]) + len(ratios) - 1
        
        num_class_out = num_anchors * (self.num_classes + 1)
        self.low_class_pred = nn.Conv2d(64, num_class_out, kernel_size=3, padding=1)
        self.mid_class_pred = nn.Conv2d(64, num_class_out, kernel_size=3, padding=1)
        self.high_class_pred = nn.Conv2d(64, num_class_out, kernel_size=3, padding=1)

        num_box_out = num_anchors * 4
        self.low_box_pred = nn.Conv2d(64, num_box_out, kernel_size=3, padding=1)
        self.mid_box_pred = nn.Conv2d(64, num_box_out, kernel_size=3, padding=1)
        self.high_box_pred = nn.Conv2d(64, num_box_out, kernel_size=3, padding=1)

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
        mid_feature_map = self.mid_layer(low_feature_map)
        high_feature_map = self.high_layer(mid_feature_map)

        low_feature_map = self.low_conv(low_feature_map)
        mid_feature_map = self.mid_conv(mid_feature_map)
        high_feature_map = self.high_conv(high_feature_map)

        anchors, cls_preds, bbox_preds = [], [], []

        anchors.append(self.low_anchors(low_feature_map))
        cls_preds.append(self.low_class_pred(low_feature_map))
        bbox_preds.append(self.low_box_pred(low_feature_map))

        anchors.append(self.mid_anchors(mid_feature_map))
        cls_preds.append(self.mid_class_pred(mid_feature_map))
        bbox_preds.append(self.mid_box_pred(mid_feature_map))

        anchors.append(self.high_anchors(high_feature_map))
        cls_preds.append(self.high_class_pred(high_feature_map))
        bbox_preds.append(self.high_box_pred(high_feature_map))

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


class CNN(nn.Module):
    """Convolutional neural network for extracting features from images"""

    def __init__(self):
        super().__init__()
        blk = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            blk.append(DownSampleBlk(num_filters[i], num_filters[i + 1]))
        self.net = nn.Sequential(*blk)

    def forward(self, X):
        return self.net(X)
