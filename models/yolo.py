"""
Network configuration similar to yolo8
"""

from models.generator import ListGen, BaseConfig
from models.module.generators import *


class Yolo(BaseConfig):
    """Generates a model similar to yolo8

    See https://viso.ai/deep-learning/yolov8-guide/.
    """

    def backbone_cfgs(self) -> ListGen:
        return [
            *self._conv(64, 3, 2),
            *self._c2f(64, 2),
            *self._conv(128, 3, 2),
            *self._c2f(128, 3),
        ]

    def neck_cfgs(self) -> ListGen:
        return [
            *self._conv(256, 3, 2),
            *self._c2f(256, 4),
            Return(),
            *self._conv(256, 3, 2),
            *self._c2f(256, 3),
            Return(),
            *self._conv(256, 3, 2),
            *self._c2f(256, 2),
            Return(),
        ]

    def head_cfgs(self, box_out: int, cls_out: int) -> ListGen:
        return [
            [Conv(kernel_size=1), Norm(), LI(state_storage=self.state_storage), Tanh()],
            [
                Conv(box_out, 1),
            ],
            [
                Conv(cls_out, 1),
            ],
        ]

    def _conv(self, out_channels: int = None, kernel: int = 3, stride: int = 1):
        return (
            Conv(out_channels, stride=stride, kernel_size=kernel),
            Norm(),
            LIF(state_storage=self.state_storage),
        )

    def _bottleneck(self, shortcut: bool = True):
        net = (*self._conv(),)
        if shortcut:
            return Residual([[*net], [Pass()]])
        else:
            return net

    def _rec_block(self, n: int, shortcut: bool):
        if n == 0:
            return []
        return (
            Dense(
                [
                    [self._bottleneck(shortcut), *self._rec_block(n - 1, shortcut)],
                    [Pass()],
                ]
            ),
        )

    def _c2f(self, out_channels: int, n: int, shortcut: bool = True):
        return (
            Conv(out_channels, 1),
            Dense(
                [
                    [Conv(int(out_channels / 2), 1), *self._rec_block(n, shortcut)],
                    [Conv(int(out_channels / 2), 1)],
                ]
            ),
            Conv(out_channels, 1),
        )
