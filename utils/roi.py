import torch
from tqdm import tqdm
import utils.box as box


class RoI:
    """Label anchor boxes using ground-truth bounding boxes."""

    def __init__(self, iou_threshold=0.5) -> None:
        """
        Args:
            iou_threshold (float): Minimum acceptable iou. TODO
        """
        self.iou_threshold = iou_threshold

    def __call__(
        self, anchors: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Label anchor boxes using ground-truth bounding boxes
        Args:
            anchors (torch.Tensor): [num_anchors, 4]
            labels (torch.Tensor): [num_batch, num_gt_boxes, 5 (class, luw, luh, rdw, rdh)]
        Returns:
            bbox_offset: [num_batch, num_anchors, 4]. Ground truth offsets for each box
            bbox_mask: [num_batch, num_anchors, 4]. (0)*4 for background, (1)*4 for object
            class_labels: [num_batch, num_anchors]. Class of each box (0 - background)
        """
        batch_size = labels.shape[0]
        device, num_anchors = anchors.device, anchors.shape[0]
        batch_offset, batch_mask, batch_class_labels = [], [], []
        for i in range(batch_size):
            label = labels[i, :, :]
            anchors_bbox_map = self.assign_anchor_to_box(label[:, 1:], anchors)
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
            # Initialize class labels and assigned bounding box coordinates with zeros
            class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            assigned_bb = torch.zeros(
                (num_anchors, 4), dtype=torch.float32, device=device
            )
            # Label classes of anchor boxes using their assigned ground-truth
            # bounding boxes. If an anchor box is not assigned any, we label its
            # class as background (the value remains zero)
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            # Offset transformation
            offset = box.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset)
            batch_mask.append(bbox_mask)
            batch_class_labels.append(class_labels)
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)
        return bbox_offset, bbox_mask, class_labels

    def assign_anchor_to_box(
        self, ground_truth: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Assign closest ground-truth bounding boxes to anchor boxes.
        see https://d2l.ai/chapter_computer-vision/anchor.html#assigning-ground-truth-bounding-boxes-to-anchor-boxes
        Args:
            ground_truth (torch.Tensor): The ground-truth bounding boxes [num_gt_box, 4] - ulw, ulh, drw, drh
            anchors (torch.Tensor): Anchors boxes [num_anchors, 4] - ulw, ulh, drw, drh
        Returns:
            torch.Tensor: Tensor with ground truth box indices [num_anchors]
        """
        num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
        # The Jaccard index measures the similarity between two sets [num_anchors, num_gt_box]
        jaccard = box.box_iou(anchors, ground_truth)

        # Initialize the tensor to hold the assigned ground-truth bounding box for each anchor
        anchors_box_map = torch.full(
            (num_anchors,), -1, dtype=torch.long, device=anchors.device
        )

        # Assign ground-truth bounding boxes according to the threshold
        max_ious, indices = torch.max(jaccard, dim=1)
        # Indexes of non-empty boxes
        mask = max_ious >= self.iou_threshold
        anc_i = torch.nonzero(mask).reshape(-1)
        if len(anc_i) == 0:
            tqdm.write("[WARN]: There is no suitable anchor")
        box_j = indices[mask]
        # Each anchor is assigned a gt_box with the highest iou if it is greater than the threshold
        anchors_box_map[anc_i] = box_j

        # For each gt_box we assign an anchor with maximum iou
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)  # Find the largest IoU
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_box_map[anc_idx] = box_idx
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return anchors_box_map
