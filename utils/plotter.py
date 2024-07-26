import torch
from matplotlib import pyplot as plt
from utils import box


class Plotter:
    def __init__(self, rows=2, columns=4, threshold=0.8, labels=None):
        self.rows = rows
        self.columns = columns
        self.threshold = threshold
        self.labels = labels
        self.colors = ["b", "g", "r", "m", "c"]

    def display(self, images, predictions, target):
        num_img, _, h, w = images.shape
        predictions[:, :, 2] *= w
        predictions[:, :, 4] *= w
        predictions[:, :, 3] *= h
        predictions[:, :, 5] *= h
        fig = plt.figure(figsize=(h, w))
        images = images.permute(0, 2, 3, 1)
        ax = []
        for i in range(num_img):
            ax.append(fig.add_subplot(self.rows, self.columns, i + 1))
            plt.imshow(images[i])
            self.show_bboxes(ax[-1], predictions[i])

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
                f"{bbox[1]:.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
            )
