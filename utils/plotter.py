import cv2
import numpy as np
from utils import box


class Plotter:
    def __init__(self, threshold=0.8, labels=None, interval=200, wight=4):
        self.threshold = threshold
        self.labels = labels
        self.colors = ["g", "b", "r", "m", "c"]
        self.interval = interval
        self.wight = wight

    def concatenate_video(self, video: np.ndarray):
        """Combines a batch of videos into one
        Args:
            video (np.ndarray): [ts, b, h, w]
        Returns:
            Combines video (np.ndarray): [ts, h, w]
        """
        b = video.shape[1]
        video = np.pad(
            video,
            pad_width=(
                (0, 0),
                (0, self.wight - b % self.wight),
                (0, 0),
                (0, 0),
            ),
            constant_values=0,
        )
        con_imgs = []
        for time_stamp in video:
            arr = [
                np.concatenate(time_stamp[idx : idx + 4], axis=1)
                for idx in range(0, b, self.wight)
            ]
            con_imgs.append(np.concatenate(arr, axis=0))
        return np.stack(con_imgs)

    def show_video(self, video: np.ndarray):
        """Playing video
        Args:
            video (np.ndarray): [ts, h, w]
        Returns:
            bool: Returns false if "q" is pressed
        """
        for time_stamp in video:
            cv2.imshow("Res", time_stamp)
            if cv2.waitKey(self.interval) == ord("q"):
                cv2.destroyWindow("Res")
                return False
        return True

    def display(self, images, predictions=None, target=None):
        """Plays video from tensor
        Args:
            images (torch.Tensor): [ts, batch, p, h, w]
            predictions (_type_, optional): TODO. Defaults to None.
            target (_type_, optional): TODO. Defaults to None.
        """
        ts, b, p, h, w = images.shape
        plt_images = images.permute(0, 1, 3, 4, 2)
        grey_imgs = 127 * np.ones((ts, b, h, w), dtype=np.uint8)
        grey_imgs[plt_images[..., 0] > 0] = 0
        grey_imgs[plt_images[..., 1] > 0] = 255
        con_video = self.concatenate_video(grey_imgs)
        while self.show_video(con_video):
            pass

    def show_bboxes(self, axes, bboxes):
        """Show bounding boxes."""
        bboxes = bboxes.detach().numpy()
        for i, bbox in enumerate(bboxes):
            if (bbox[0] == -1) or (bbox[1] < self.threshold):
                continue
            color = self.colors[int(bbox[0]) % len(self.colors)]
            rect = box.bbox_to_rect(bbox[2:], color)
            axes.add_patch(rect)

            axes.text(
                bbox[2],
                bbox[3],
                self.labels[int(bbox[0])] + f": {bbox[1]:.2f}"
                if self.labels is not None
                else f"{bbox[1]:.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                color="r",
                size="xx-small",
            )
