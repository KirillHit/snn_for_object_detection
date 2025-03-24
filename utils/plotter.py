"""Tool for displaying event videos, predictions and boxes"""

import cv2
import numpy as np
import torch
from typing import List, Optional
import os
import matplotlib.colors as mcolors


class Plotter:
    """Tool for displaying event videos, predictions and boxes"""

    def __init__(
        self,
        threshold: float = 0.8,
        show_video: bool = True,
        save_video: bool = False,
        file_path: str = "log",
        file_name: str = "out",
    ):
        """
        :param threshold: Threshold value for displaying box. Defaults to 0.8.
        :type threshold: float, optional
        :param show_video: If true, shows video in window. Defaults to True.
        :type show_video: bool, optional
        :param save_video: If true, saves the video to a file. Defaults to False.
        :type save_video: bool, optional
        :param file_path: Folder for saved video. Defaults to "log".
        :type file_path: str, optional
        :param file_name: Save file name. Defaults to "out".
        :type file_name: str, optional
        """
        self.threshold = threshold
        self.show_video = show_video
        self.save_video = save_video
        self.file_path = file_path
        self.file_name = file_name
        self.colors = [
            list(reversed([int(c * 255) for c in mcolors.to_rgb(color)]))
            for color in mcolors.TABLEAU_COLORS
        ]
        self.labels = None

    def __call__(
        self, video: List[np.ndarray], interval: int, batch_idx: str = ""
    ) -> None:
        """Displays frames obtained by the apply method and saves them

        :param video: List of frames
        :type video: List[np.ndarray]
        :param interval: Time between frames in milliseconds
        :type interval: int
        :param batch_idx: Batch number
        :type batch_idx: str, optional
        """
        if self.show_video:
            self._show_video(video, interval, batch_idx)
        if self.save_video:
            self._save_video(video, interval, batch_idx)

    def apply(
        self,
        image: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Prepares a frame from an event camera for display
        and overlays prediction and target boxes on it

        :param image: Image from event camera. Tensor shape [channel, h, w]. Expects 2 channels.
        :type image: torch.Tensor
        :param predictions: Tensor shape [anchor, 6],
            one label contains (class, iou, xlu, ylu, xrd, yrd).
        :type predictions: Optional[torch.Tensor]
        :param target: Ground Truth. Tensor shape [count_box, 5],
            one label contains (class id, xlu, ylu, xrd, yrd)
        :type target: Optional[torch.Tensor]
        :return: Returns an image that can be processed by opensv
        :rtype: np.ndarray
        """
        _, h, w = image.shape
        plt_image = image.permute(1, 2, 0).cpu()
        res_img = np.zeros((h, w, 3), dtype=np.uint8)
        res_img[plt_image[..., 0] > 0] = [255, 150, 0]
        res_img[plt_image[..., 1] > 0, 2] = 255
        target = self._prepare_targets(target, h, w)
        predictions = self._prepare_preds(predictions, h, w)
        self._draw_target_boxes(res_img, target)
        self._draw_preds_box(res_img, predictions)
        return res_img

    def _prepare_preds(
        self, preds: Optional[torch.Tensor], height: int, width: int
    ) -> Optional[torch.Tensor]:
        if preds is None:
            return None
        preds = preds[(preds[:, 0] >= 0) & (preds[:, 1] >= self.threshold)]
        preds[:, [2, 4]] = preds[:, [2, 4]] * width
        preds[:, [3, 5]] = preds[:, [3, 5]] * height
        preds[..., 1] *= 100
        return preds.int().cpu()

    def _prepare_targets(
        self, target: Optional[torch.Tensor], height: int, width: int
    ) -> Optional[torch.Tensor]:
        if target is None:
            return None
        target = target[target[:, 0] >= 0]
        target[:, [1, 3]] = target[:, [1, 3]] * width
        target[:, [2, 4]] = target[:, [2, 4]] * height
        return target.int().cpu()

    def _draw_preds_box(self, image: np.ndarray, preds: Optional[torch.Tensor]) -> None:
        if preds is None:
            return
        for box in preds:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[int(box[0].item()) % len(self.colors)],
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text="%.2f %s"
                % (
                    box[1].item() / 100,
                    self.labels[box[0].item()] if self.labels is not None else "",
                ),
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
            )

    def _draw_target_boxes(
        self, image: np.ndarray, target: Optional[torch.Tensor]
    ) -> None:
        if target is None:
            return
        for box in target:
            start_point = (box[1].item(), box[2].item())
            end_point = (box[3].item(), box[4].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[int(box[0].item()) % len(self.colors)],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def _show_video(
        self, video: List[np.ndarray], interval: int, batch_idx: str = ""
    ) -> None:
        while True:
            for img in video:
                cv2.imshow("Res", img)
                if cv2.waitKey(interval) == ord("q"):
                    cv2.destroyWindow("Res " + batch_idx)
                    return
            if cv2.waitKey() == ord("q"):
                cv2.destroyWindow("Res " + batch_idx)
                return

    def _save_video(
        self, video: List[np.ndarray], interval: int, batch_idx: str = ""
    ) -> None:
        h, w, _ = video[0].shape
        os.makedirs(self.file_path, exist_ok=True)
        out = cv2.VideoWriter(
            os.path.join(self.file_path, self.file_name + batch_idx + ".avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            1000 / interval,
            (w, h),
        )
        for img in video:
            out.write(img)
