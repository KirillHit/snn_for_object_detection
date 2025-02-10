"""Tool for displaying images, predictions and boxes"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Union


class Plotter:
    """Tool for displaying images, predictions and boxes

    `OpenCV <https://github.com/opencv/opencv>`_ is used for displaying.
    """

    def __init__(
        self,
        threshold=0.8,
        labels: Optional[List[str]] = None,
        interval: int = 200,
        columns: int = 4,
    ):
        """
        :param threshold: Threshold value for displaying box. Defaults to 0.8.
        :type threshold: float, optional
        :param labels: List of class names. Defaults to None.
        :type labels: Optional[List[str]], optional
        :param interval: Time interval between frames in milliseconds. Defaults to 200.
        :type interval: int, optional
        :param columns: Images are displayed in a grid. This parameter determines its width. Defaults to 4.
        :type columns: int, optional
        """
        self.threshold = int(threshold * 100)
        self.labels = labels
        self.colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (0, 128, 128),
        ]
        self.interval = interval
        self.columns = columns

    def display(
        self,
        images: torch.Tensor,
        predictions: Optional[torch.Tensor],
        target: Optional[Union[List[torch.Tensor], torch.Tensor]],
    ):
        """Plays video from tensor

        :param images: Shape [ts, batch, channel, h, w]. Expects 2 channels.
        :type images: torch.Tensor
        :param predictions: Shape [ts, batch, anchor, 6].

            One label contains (class, iou, xlu, ylu, xrd, yrd).
        :type predictions: Optional[torch.Tensor]
        :param target: Ground Truth. The length of the list is equal to the number of batch.

            Expects format:

                Tensor shape [count_box, 6]

                One label contains (ts, class id, xlu, ylu, xrd, yrd)

            or

                Tensor shape [count_box, 5]

                One label contains (class id, xlu, ylu, xrd, yrd)
        :type target: Optional[Union[List[torch.Tensor], torch.Tensor]]
        """
        ts, b, _, h, w = images.shape
        plt_images = images.permute(0, 1, 3, 4, 2)
        grey_imgs = 127 * np.ones((ts, b, h, w), dtype=np.uint8)
        grey_imgs[plt_images[..., 0] > 0] = 0
        grey_imgs[plt_images[..., 1] > 0] = 255
        con_video = self._concatenate_video(grey_imgs).repeat(3, axis=-1)
        prep_target, prep_preds = None, None
        if target is not None:
            if isinstance(target, torch.Tensor):
                target = self._transform_targets(target, images.shape[0] - 1)
            prep_target = self._prepare_targets(target, h, w)
        if predictions is not None:
            prep_preds = self._prepare_preds(predictions, h, w)
        boxed_video = self._apply_boxes(con_video, prep_preds, prep_target)
        while self._show_video(boxed_video):
            cmd = cv2.waitKey()
            if cmd == ord("s"):
                self._save_video(boxed_video)
            if cmd == ord("q") or cmd == ord("s"):
                cv2.destroyWindow("Res")
                break

    def _transform_targets(
        self, target: torch.Tensor, time_step: int
    ) -> List[torch.Tensor]:
        """Transform targets from [batch_size, num_box, 5] [class id, xlu, ylu, xrd, yrd]
        to List[torch.Tensor[num_box, 6]] [ts, class id, xlu, ylu, xrd, yrd]
        """
        new_target = []
        for batch_idx in range(target.shape[0]):
            batch = target[batch_idx]
            batch = batch[batch[:, 0] >= 0]
            time_tens = torch.ones((batch.shape[0], 1)) * time_step
            new_target.append(torch.concatenate((time_tens, batch), dim=1))
        return new_target

    def _concatenate_video(self, video: np.ndarray) -> np.ndarray:
        """Combines a batch of videos into one

        :param video: Shape [ts, b, h, w]
        :type video: np.ndarray
        :return: Combines video. Shape [ts, h, w]
        :rtype: np.ndarray
        """
        b = video.shape[1]
        video = np.pad(
            video,
            pad_width=(
                (0, 0),
                (0, self.columns - b % self.columns),
                (0, 0),
                (0, 0),
            ),
            constant_values=0,
        )
        con_imgs = []
        for time_stamp in video:
            arr = [
                np.concatenate(time_stamp[idx : idx + self.columns], axis=1)
                for idx in range(0, b, self.columns)
            ]
            con_imgs.append(np.concatenate(arr, axis=0))
        return np.stack(con_imgs)[..., None]

    def _prepare_preds(
        self, predictions: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Changes the coordinates of the boxes according to the position of the batch

        :param predictions: Shape [ts, batch, anchors, 6]

            One label contains (class, iou, xlu, ylu, xrd, yrd)
        :type predictions: torch.Tensor
        :param height: height img
        :type height: int
        :param width: width img
        :type width: int
        :return: Shape [ts, count_boxes, 6]

            One label contains (class, iou, xlu, ylu, xrd, yrd)
        :rtype: torch.Tensor
        """
        for batch_idx in range(predictions.shape[1]):
            predictions[:, batch_idx, :, [2, 4]] = (
                torch.clamp(predictions[:, batch_idx, :, [2, 4]], min=0.0, max=1.0)
                * width
                + (batch_idx % self.columns) * width
            )
            predictions[:, batch_idx, :, [3, 5]] = (
                torch.clamp(predictions[:, batch_idx, :, [3, 5]], min=0.0, max=1.0)
                * height
                + (batch_idx // self.columns) * height
            )
        predictions[..., 1] *= 100
        return torch.flatten(predictions, start_dim=1, end_dim=2).type(torch.int32)

    def _prepare_targets(
        self, target: List[torch.Tensor], height: int, width: int
    ) -> torch.Tensor:
        """Changes the coordinates of the boxes according to the position of the batch

        :param target: The length of the list is equal to the number of packs

            Tensor shape [count_box, 6]

            One label contains (ts, class id, xlu, ylu, xrd, yrd)
        :type target: List[torch.Tensor]
        :param height: height img
        :type height: int
        :param width: width img
        :type width: int
        :return: Shape [count_boxes, 6]

            One label contains (ts, class id, xlu, ylu, xrd, yrd)
        :rtype: torch.Tensor
        """
        for batch_idx, t_batch in enumerate(target):
            t_batch[:, [2, 4]] = (
                torch.clamp(t_batch[:, [2, 4]], min=0.0, max=1.0) * width
                + (batch_idx % self.columns) * width
            )
            t_batch[:, [3, 5]] = (
                torch.clamp(t_batch[:, [3, 5]], min=0.0, max=1.0) * height
                + (batch_idx // self.columns) * height
            )
        return torch.concatenate(target, dim=0).type(torch.int32)

    def _show_video(self, video: np.ndarray) -> bool:
        """Playing video

        :param video: Shape [ts, h, w, channels]
        :type video: np.ndarray
        :return: Returns ``False`` if "q" is pressed, otherwise ``True``
        :rtype: bool
        """
        for img in video:
            cv2.imshow("Res", img)
            if cv2.waitKey(self.interval) == ord("q"):
                cv2.destroyWindow("Res")
                return False
        return True

    def _save_video(self, video: np.ndarray) -> None:
        _, h, w, _ = video.shape
        out = cv2.VideoWriter(
            "log/out.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (w, h)
        )
        for img in video:
            out.write(img)
        for _ in range(60):
            out.write(img)
        return True

    def _apply_boxes(
        self,
        video: np.ndarray,
        preds: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> np.ndarray:
        """Adds boxes to frames

        :param video: Shape [ts, h, w, channel]
        :type video: np.ndarray
        :param preds: Shape [ts, count_box, 6]

            One label contains (class, iou, xlu, ylu, xrd, yrd)
        :type preds: Optional[torch.Tensor]
        :param target: Shape [count_box, 6]

            One label contains (ts, class id, xlu, ylu, xrd, yrd)
        :type target: Optional[torch.Tensor]
        :return: Video with boxes
        :rtype: np.ndarray
        """
        if (target is None) and (preds is None):
            return video
        boxed_video = np.empty_like(video, dtype=video.dtype)
        for ts, img in enumerate(video):
            if target is not None:
                self._draw_target_boxes(img, target[target[:, 0] == ts])
            if preds is not None:
                self._draw_preds_box(img, preds[ts])
            boxed_video[ts] = img
        return boxed_video

    def _draw_preds_box(self, image: np.ndarray, preds: torch.Tensor) -> None:
        """Draw bounding boxes for preds

        :param image: Shape [h, w, channel]
        :type image: np.ndarray
        :param preds: Shape [count_box, 6]

            One label contains (class, iou, xlu, ylu, xrd, yrd)
        :type preds: torch.Tensor
        """
        mask = (preds[:, 0] >= 0) & (preds[:, 1] >= self.threshold)
        for box in preds[mask]:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[box[0] % len(self.colors)],
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text="%.2f %s" % (box[1].item() / 100, self.labels[box[0]]),
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=(255, 0, 0),
                lineType=cv2.LINE_AA,
            )

    def _draw_target_boxes(self, image: np.ndarray, target: torch.Tensor) -> None:
        """Draw bounding boxes for targets

        :param image: Image for drawing
        :type image: np.ndarray
        :param target: Shape [count_box, 7]

            One label contains (ts, class id, xlu, ylu, xrd, yrd)
        :type target: torch.Tensor
        """
        for box in target:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=[c / 2 for c in self.colors[box[1] % len(self.colors)]],
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text=self.labels[box[1]],
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=(0, 60, 0),
                lineType=cv2.LINE_AA,
            )
