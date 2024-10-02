import cv2
import numpy as np
import torch


class Plotter:
    """Displays images, predictions and boxes"""

    def __init__(self, threshold=0.8, labels=None, interval=200, columns=4):
        self.threshold = threshold
        self.labels = labels
        self.colors = [(0, 255, 0), (0, 0, 255)]
        self.interval = interval
        self.columns = columns

    def display(self, images, predictions=None, target=torch.empty((0,))):
        """Plays video from tensor
        Args:
            images (torch.Tensor): Shape [ts, batch, p, h, w]
            predictions (torch.Tensor): TODO. Defaults to None.
            target (torch.Tensor): Shape [batch, max_len, 6]. Empty fill -1.
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd].
                Defaults to None.
        """
        ts, b, _, h, w = images.shape
        plt_images = images.permute(0, 1, 3, 4, 2)
        grey_imgs = 127 * np.ones((ts, b, h, w), dtype=np.uint8)
        grey_imgs[plt_images[..., 0] > 0] = 0
        grey_imgs[plt_images[..., 1] > 0] = 255
        con_video = self.concatenate_video(grey_imgs)
        prep_target = self.prepare_targets(target, h, w)
        while self.show_video(con_video, prep_target):
            pass

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
                np.concatenate(time_stamp[idx : idx + 4], axis=1)
                for idx in range(0, b, self.columns)
            ]
            con_imgs.append(np.concatenate(arr, axis=0))
        return np.stack(con_imgs)[..., None].repeat(3, axis=-1)

    def prepare_targets(self, target: torch.Tensor, hight: int, wight: int):
        """Changes the coordinates of the boxes according to the position of the batch
        Args:
            target (torch.Tensor): Shape [batch, max_len, 6]. Empty fill -1.
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        Returns:
            torch.Tensor: Shape [count_boxes, 6].
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        """
        arr = []
        for batch_idx, t_batch in enumerate(target):
            filtered_targets = t_batch[t_batch[:, 0] >= 0]
            filtered_targets[:, [2, 4]] += (batch_idx % self.columns) * wight
            filtered_targets[:, [3, 5]] += (batch_idx // self.columns) * hight
            arr.append(filtered_targets)
        return torch.concatenate(arr, dim=0)

    def show_video(self, video: np.ndarray, target: torch.Tensor):
        """Playing video
        Args:
            video (np.ndarray): Shape [ts, h, w, c]
            target (torch.Tensor): Shape [count_box, 6].
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        Returns:
            bool: Returns false if "q" is pressed
        """
        for idx, time_stamp in enumerate(video):
            img = self.show_target_boxes(time_stamp, target[target[:, 0] == idx])
            cv2.imshow("Res", img)
            if cv2.waitKey(self.interval) == ord("q"):
                cv2.destroyWindow("Res")
                return False
        return True

    def show_target_boxes(self, image: np.ndarray, target: torch.Tensor):
        """Show bounding boxes for targets
        Args:
            image (np.ndarray): Image for drawing
            target (torch.Tensor): Shape [count_box, 7].
                One label contains [ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd]
        Returns:
            np.ndarray: Image with boxes
        """
        for box in target:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image, start_point, end_point, color=self.colors[box[1]], thickness=1
            )
            cv2.putText(
                image,
                text=self.labels[box[1]],
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(255, 0, 0),
            )
        return image
