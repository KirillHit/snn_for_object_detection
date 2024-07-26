import torch
from tqdm import tqdm


class RoI:
    """Label anchor boxes using ground-truth bounding boxes."""

    def __init__(self, iou_threshold=0.5) -> None:
        """
        Args:
            iou_threshold (float): Minimum acceptable iou. TODO
        """
        self.iou_threshold = iou_threshold

    def __call__(self, anchors: torch.Tensor, labels: torch.Tensor):
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
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
            # Label classes of anchor boxes using their assigned ground-truth
            # bounding boxes. If an anchor box is not assigned any, we label its
            # class as background (the value remains zero)
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            # Offset transformation
            offset = self.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset)
            batch_mask.append(bbox_mask)
            batch_class_labels.append(class_labels)
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)
        return bbox_offset, bbox_mask, class_labels

    def offset_boxes(self, anchors, assigned_bb, eps=1e-6):
        """Transform for anchor box offsets.
        see https://d2l-ai.translate.goog/chapter_computer-vision/anchor.html?_x_tr_sl=auto&_x_tr_tl=ru&_x_tr_hl=ru#labeling-classes-and-offsets"""
        # TODO Найти формулу получше
        c_anc = self.box_corner_to_center(anchors)
        c_assigned_bb = self.box_corner_to_center(assigned_bb)
        offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
        offset = torch.concat([offset_xy, offset_wh], axis=1)
        return offset

    def box_corner_to_center(self, boxes):
        """Convert from (upper-left, lower-right) to (center, width, height)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = torch.stack((cx, cy, w, h), axis=-1)
        return boxes

    def box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU across two lists of anchor or bounding boxes.

        Args:
            boxes1 (torch.Tensor): anchors [num_anchors, 4] - ulw, ulh, drw, drh
            boxes2 (torch.Tensor): ground truth [num_gt_box, 4] - ulw, ulh, drw, drh

        Returns:
            IoU (torch.Tensor): [num_anchors, num_gt_box] Element x_ij in the i-th row and j-th
            column is the IoU of the anchor box i and the ground-truth bounding box j
        """
        assert boxes1.shape == (boxes1.shape[0], 4), "Wrong box shape"
        assert boxes2.shape == (boxes2.shape[0], 4), "Wrong box shape"

        # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
        # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
        areas1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=1)
        areas2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=1)
        # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
        # boxes1, no. of boxes2, 2)
        inter_up_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_low_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = torch.clamp(inter_low_rights - inter_up_lefts, min=0)
        # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
        inter_areas = torch.prod(inters, dim=2)
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas

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
        jaccard = self.box_iou(anchors, ground_truth)

        # Initialize the tensor to hold the assigned ground-truth bounding box for each anchor
        anchors_box_map = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)

        # Assign ground-truth bounding boxes according to the threshold
        max_ious, indices = torch.max(jaccard, dim=1)
        # Indexes of non-empty boxes
        mask = max_ious >= self.iou_threshold
        anc_i = torch.nonzero(mask).reshape(-1)
        if len(anc_i) == 0:
            tqdm.write("Warning: There is no suitable anchor")
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
