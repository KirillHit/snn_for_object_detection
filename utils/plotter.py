import cv2
import numpy as np
import torch
from typing import List, Optional, Union


class Plotter:
    """Displays images, predictions and boxes"""

    def __init__(self, threshold=0.8, labels=None, interval=200, columns=4):
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
        """
        Plays video from tensor
        Args:
            images (torch.Tensor): Shape [ts, batch, p, h, w]
            predictions (Optional[torch.Tensor]): Shape [ts, batch, anchors, 6].
                One label contains [class, iou, xlu, ylu, xrd, yrd]
            target (List[Tensor] | Tensor | None): The length of the list is equal to the number of batch.
                One Tensor contains [count_box, 6]
                One label contains [ts, class id, xlu, ylu, xrd, yrd]
                or
                One Tensor contains [count_box, 5]
                One label contains [class id, xlu, ylu, xrd, yrd]
        """
        ts, b, _, h, w = images.shape
        plt_images = images.permute(0, 1, 3, 4, 2)
        grey_imgs = 127 * np.ones((ts, b, h, w), dtype=np.uint8)
        grey_imgs[plt_images[..., 0] > 0] = 0
        grey_imgs[plt_images[..., 1] > 0] = 255
        con_video = self.concatenate_video(grey_imgs)
        prep_target, prep_preds = None, None
        if target is not None:
            if isinstance(target, torch.Tensor):
                target = self.transform_targets(target, images.shape[0] - 1)
            prep_target = self.prepare_targets(target, h, w)
        if predictions is not None:
            prep_preds = self.prepare_preds(predictions, h, w)
        while self.show_video(con_video, prep_preds, prep_target):
            cmd = cv2.waitKey()
            if cmd == ord("q"):
                cv2.destroyWindow("Res")
                break

    def transform_targets(
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

    def concatenate_video(self, video: np.ndarray):
        """Combines a batch of videos into one
        Args:
            video (np.ndarray): [ts, b, h, w]
        Returns:
            Combines video (np.ndarray): [ts, h, w, c]
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
        return np.stack(con_imgs)[..., None].repeat(3, axis=-1)

    def prepare_preds(
        self, predictions: torch.Tensor, hight: int, wight: int
    ) -> torch.Tensor:
        """Changes the coordinates of the boxes according to the position of the batch
        Args:
            predictions (torch.Tensor): Shape [ts, batch, anchors, 6].
                One label contains [class, iou, xlu, ylu, xrd, yrd]
        Returns:
            torch.Tensor: Shape [ts, count_boxes, 6].
                One label contains [class, iou, xlu, ylu, xrd, yrd]
        """
        for batch_idx in range(predictions.shape[1]):
            predictions[:, batch_idx, :, [2, 4]] = (
                predictions[:, batch_idx, :, [2, 4]] * wight
                + (batch_idx % self.columns) * wight
            )
            predictions[:, batch_idx, :, [3, 5]] = (
                predictions[:, batch_idx, :, [3, 5]] * hight
                + (batch_idx // self.columns) * hight
            )
        predictions[..., 1] *= 100
        return torch.flatten(predictions, start_dim=1, end_dim=2).type(torch.int32)

    def prepare_targets(self, target: List[torch.Tensor], hight: int, wight: int):
        """Changes the coordinates of the boxes according to the position of the batch
        Args:
            target (List[torch.Tensor]): The length of the list is equal to the number of packs.
                One Tensor contains [count_box, 6]
                One label contains [ts, class id, xlu, ylu, xrd, yrd]
        Returns:
            torch.Tensor: Shape [count_boxes, 6].
                One label contains [ts, class id, xlu, ylu, xrd, yrd]
        """
        for batch_idx, t_batch in enumerate(target):
            t_batch[:, [2, 4]] = (
                torch.clamp(t_batch[:, [2, 4]], min=0.0, max=1.0) * wight
                + (batch_idx % self.columns) * wight
            )
            t_batch[:, [3, 5]] = (
                torch.clamp(t_batch[:, [3, 5]], min=0.0, max=1.0) * hight
                + (batch_idx // self.columns) * hight
            )
        return torch.concatenate(target, dim=0).type(torch.int32)

    def show_video(
        self,
        video: np.ndarray,
        preds: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ):
        """Playing video
        Args:
            video (np.ndarray): Shape [ts, h, w, c]
            predictions (torch.Tensor): Shape [ts, count_box, 6].
                One label contains [class, iou, xlu, ylu, xrd, yrd]
            target (torch.Tensor): Shape [count_box, 6].
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        Returns:
            bool: Returns false if "q" is pressed
        """
        for ts, capture in enumerate(video):
            if target is not None:
                capture = self.draw_target_boxes(capture, target[target[:, 0] == ts])
            if preds is not None:
                capture = self.draw_preds_box(capture, preds[ts])
            cv2.imshow("Res", capture)
            if cv2.waitKey(self.interval) == ord("q"):
                cv2.destroyWindow("Res")
                return False
        return True

    def draw_preds_box(self, image: np.ndarray, preds: torch.Tensor) -> np.ndarray:
        """Draw bounding boxes for preds
        Args:
            image (np.ndarray): Shape [h, w, c]
            preds (torch.Tensor): Shape [count_box, 6].
                One label contains [class, iou, xlu, ylu, xrd, yrd]
        Returns:
            np.ndarray: Image with boxes
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
            )
            cv2.putText(
                image,
                text="%.2f %s" % (box[1].item() / 100, self.labels[box[0]]),
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=2,
                color=(255, 0, 0),
            )
        return image

    def draw_target_boxes(self, image: np.ndarray, target: torch.Tensor) -> np.ndarray:
        """Draw bounding boxes for targets
        Args:
            image (np.ndarray): Image for drawing
            target (torch.Tensor): Shape [count_box, 7].
                One label contains [ts, class id, xlu, ylu, xrd, yrd]
        Returns:
            np.ndarray: Image with boxes
        """
        for box in target:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[box[1] % len(self.colors)],
                thickness=2,
            )
            cv2.putText(
                image,
                text=self.labels[box[1]],
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=2,
                color=(255, 0, 0),
            )
        return image
