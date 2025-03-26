"""Tool for generating prediction anchors"""

import torch
from torch import nn
from models.generator import Storage


class AnchorGenerator(nn.Module):
    """Tool for generating prediction anchors"""

    def __init__(
        self,
        storage: Storage,
        sizes: torch.Tensor,
        ratios: torch.Tensor,
    ) -> None:
        """
        :param storage: Storage in which the anchors will be stored.
        :type storage: Storage
        :param sizes: Box scales (0,1] = S'/S.
        :type sizes: torch.Tensor
        :param ratios: Ratio of width to height of boxes (w/h).
        :type ratios: torch.Tensor
        """
        super().__init__()
        self.storage, self.sizes, self.ratios = storage, sizes, ratios

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Generate reference blocks with different shapes,
        centered on each pixel and save them in storage

        Returns input tensor back

        :param X: Feature map
        :type X: torch.Tensor
        :return: Input tensor
        :rtype: torch.Tensor
        """
        if not hasattr(self, "anchors"):
            self._cal_anchors(X)
            self.storage.forward(self.anchors)
        return X

    def _cal_anchors(self, X: torch.Tensor) -> None:
        in_height, in_width = X.shape[-2:]
        device, num_sizes, num_ratios = X.device, len(self.sizes), len(self.ratios)
        boxes_per_pixel = num_sizes * num_ratios
        # Offsets are required to move the anchor to the center of a pixel. Since
        # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height  # Scaled steps in y axis
        steps_w = 1.0 / in_width  # Scaled steps in x axisr

        # Generate all center points for the anchor boxes
        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # Generate `boxes_per_pixel` number of heights and widths that are later
        # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
        w = torch.cat([self.sizes * ratio for ratio in self.ratios]) * in_height / in_width
        h = torch.cat([self.sizes / ratio for ratio in self.ratios]) * in_width / in_height
        # Divide by 2 to get half height and half width
        anchor_manipulations = (
            torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
        ).to(device)

        # Each center point will have `boxes_per_pixel` number of anchor boxes, so
        # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(
            boxes_per_pixel, dim=0
        )

        self.anchors = out_grid + anchor_manipulations
