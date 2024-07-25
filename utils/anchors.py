import torch
from torch import nn


class AnchorGenerator(nn.Module):
    """Extracts multiple region proposals on the input image"""

    def __init__(
        self,
        sizes: tuple = (0.1, 0.3, 0.95),
        ratios: tuple = (0.7, 1.0, 1.3),
        step=1,
    ):
        """Extracts multiple region proposals on the input image

        Args:
            sizes (tuple, optional): Box scales (0,1] = S'/S.
            ratios (tuple, optional): Ratio of width to height of boxes.
            step (int, optional): Box per pixel.
        """
        super().__init__()
        self.sizes, self.ratios, self.step = sizes, ratios, step

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Returns hypotheses

        Args:
            X (torch.Tensor): Feature map

        Returns:
            torch.Tensor: Tensor with hypotheses. Shape: (number of anchors, 4). Data: [l_up_w, l_up_h, r_down_w, r_down_h]
        """
        if not hasattr(self, "anchors"):
            self.cal_anchors(X)

        return self.anchors

    def cal_anchors(self, X: torch.Tensor) -> None:
        """Calculates new anchors

        Args:
            X (torch.Tensor): Img tensor
        """
        img_h, img_w = X.shape[-2:]
        device = X.device

        sizes = torch.tensor(self.sizes, device=device, dtype=torch.float)
        ratios = torch.tensor(self.ratios, device=device, dtype=torch.float)
        steps_w = 1.0 / img_w  # Scaled steps in x axis
        steps_h = 1.0 / img_h  # Scaled steps in y axis

        # Calculation of displacements
        # (s_0, r_j), (s_i, r_0)
        # w'/w = sqrt(s*h*r/w)
        # h'/h = sqrt(s*w/(h*r))
        box_w_h = torch.sqrt(
            torch.stack(
                (
                    torch.cat(
                        (
                            sizes[0] * ratios * img_h / img_w,
                            sizes[1:] * ratios[0] * img_h / img_w,
                        )
                    ),  # dw boxs
                    torch.cat(
                        (
                            sizes[0] * img_w / (ratios * img_h),
                            sizes[1:] * img_w / (ratios[0] * img_h),
                        )
                    ),  # dh boxs
                ),
                dim=1,
            )
        )
        # The anchor cannot be larger than the frame
        box_w_h[:] = torch.clamp(box_w_h, max=1.0)

        grid = torch.stack(
            torch.meshgrid(
                (torch.arange(0, img_w, self.step, device=device) + 0.5) * steps_w,
                (torch.arange(0, img_h, self.step, device=device) + 0.5) * steps_h,
                indexing="ij",
            ),
            dim=2,
        ).reshape(-1, 2)

        # Prepare the grid
        grid = torch.unsqueeze(grid, dim=1).repeat(1, len(box_w_h), 1)

        # Add offset and combine
        left_up = torch.reshape(grid[:] - (box_w_h / 2), (-1, 2))
        right_down = torch.reshape(grid[:] + (box_w_h / 2), (-1, 2))
        boxes = torch.cat((left_up, right_down), dim=1)

        # remove boxes that go outside the border
        """ threshold_w = (self.step + 8) / img_w
        threshold_h = (self.step + 8) / img_h
        mask = (
            (boxes[:, 0] >= (0.0 - threshold_w))
            & (boxes[:, 1] >= (0.0 - threshold_h))
            & (boxes[:, 2] <= (1.0 + threshold_w))
            & (boxes[:, 3] <= (1.0 + threshold_h))
        ) """
        self.anchors = boxes  # [mask]
