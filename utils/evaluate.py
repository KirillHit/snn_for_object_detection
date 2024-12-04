import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class SODAeval:
    annotations = []
    results = []
    images = []
    image_id = 0

    def __init__(self, labelmap):
        self.labelmap = labelmap
        self.categories = [
            {"id": id + 1, "name": class_name, "supercategory": "none"}
            for id, class_name in enumerate(labelmap)
        ]

    def add(self, gts: torch.Tensor, preds: torch.Tensor, img: torch.Tensor) -> None:
        """Adds new predictions
        Args:
            gts (torch.Tensor): Shape [batch, anchors, 5].
                One preds contains [class id, xlu, ylu, xrd, yrd]
            preds (torch.Tensor): Shape [batch, anchors, 6].
                One preds contains [class, iou, xlu, ylu, xrd, yrd]
            img (torch.Tensor): Shape [batch, p, h, w].
        """
        batch = gts.shape[0]

        for idx in range(batch):
            self._to_coco_format(gts[idx], preds[idx], img[idx])

    def reset(self):
        self.annotations.clear()
        self.results.clear()
        self.images.clear()
        self.image_id = 0

    def get_eval(self):
        dataset = {
            "info": {},
            "licenses": [],
            "type": "instances",
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_pred = coco_gt.loadRes(self.results)

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.params.imgIds = np.arange(1, self.image_id + 1, dtype=int)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def _to_coco_format(
        self, gt: torch.Tensor, pred: torch.Tensor, img: torch.Tensor
    ) -> None:
        self.image_id += 1
        _, height, width = img.shape
        self.images.append(
            {
                "date_captured": "2019",
                "file_name": "n.a",
                "id": self.image_id,
                "license": 1,
                "url": "",
                "height": height,
                "width": width,
            }
        )

        gt[:, [1, 3]] *= width
        gt[:, [2, 4]] *= height

        for bbox in gt:
            class_id = int(bbox[0].item())
            if class_id == -1:
                break
            x, y = int(bbox[1].item()), int(bbox[2].item())
            w, h = int(bbox[3].item()) - x, int(bbox[4].item()) - y
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": self.image_id,
                "bbox": [x, y, w, h],
                "category_id": class_id + 1,
                "id": len(self.annotations) + 1,
            }
            self.annotations.append(annotation)

        masked_pred = pred[pred[:, 0] >= 0]
        masked_pred[:, [2, 4]] *= width
        masked_pred[:, [3, 5]] *= height
        for bbox in masked_pred:
            x, y = int(bbox[2].item()), int(bbox[3].item())
            w, h = int(bbox[4].item()) - x, int(bbox[5].item()) - y
            image_result = {
                "image_id": self.image_id,
                "category_id": int(bbox[0].item()) + 1,
                "score": bbox[1].item(),
                "bbox": [x, y, w, h],
            }
            self.results.append(image_result)
