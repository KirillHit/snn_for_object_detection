"""
Network configuration similar to yolo8
"""

from models.soda import SODa
from models.generator import ListGen
from models.modules import *


class Yolo(SODa):
    """Generates a model similar to yolo8

    See https://github.com/ultralytics/ultralytics/issues/189.
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        :param model: model type - n, s, m, l, x
        :type model: str
        :raises KeyError: Invalid model type.
        """
        super().__init__(*args, **kwargs)
        model_types = {
            "n": (3, 0.25, 2.0),
            "s": (3, 0.50, 2.0),
            "m": (2 / 3, 0.75, 1.5),
            "l": (1, 1.00, 1.0),
            "x": (1, 1.25, 1.0),
        }
        # d: depth_multiple, w: width_multiple, r: ratio
        self.d, self.w, self.r = model_types[model]

    def base_cfgs(self) -> ListGen:
        storage_4 = Storage()
        storage_6 = Storage()
        storage_9 = Storage()
        storage_12 = Storage()
        storage_return = Storage()
        return [
            *self._conv(int(64 * self.w), 3, 2),
            *self._conv(int(128 * self.w), 3, 2),
            *self._c2f(int(128 * self.w), int(3 / self.d)),
            *self._conv(int(256 * self.w), 3, 2),
            *self._c2f(int(256 * self.w), int(6 / self.d)),
            Store(storage_4),
            *self._conv(int(512 * self.w), 3, 2),
            *self._c2f(int(512 * self.w), int(6 / self.d)),
            Store(storage_6),
            *self._conv(int(512 * self.w * self.r), 3, 2),
            *self._c2f(int(512 * self.w * self.r), int(3 / self.d)),
            *self._sppf(),
            Store(storage_9),
            Up(),
            Store(storage_6),
            Dense(storage_6),
            *self._c2f(int(512 * self.w), int(3 / self.d), False),
            Store(storage_12),
            Up(),
            Store(storage_4),
            Dense(storage_4),
            *self._c2f(int(256 * self.w), int(3 / self.d), False),
            Store(storage_return),
            *self._conv(int(256 * self.w), 3, 2),
            Store(storage_12),
            Dense(storage_12),
            *self._c2f(int(512 * self.w), int(3 / self.d), False),
            Store(storage_return),
            *self._conv(int(512 * self.w), 3, 2),
            Store(storage_9),
            Dense(storage_9),
            *self._c2f(int(512 * self.w * self.r), int(3 / self.d), False),
            Store(storage_return),
        ]

    def head_cfgs(self, box_out: int, cls_out: int) -> ListGen:
        storage = Storage()
        return [
            [
                Store(storage),
            ],
            [
                Get(storage),
                *self._conv(),
                *self._conv(),
                LI(state_storage=self.hparams.state_storage),
                Tanh(),
                Conv(box_out, 1),
            ],
            [
                Get(storage),
                *self._conv(),
                *self._conv(),
                LI(state_storage=self.hparams.state_storage),
                Tanh(),
                Conv(cls_out, 1),
            ],
        ]

    def _conv(self, out_channels: int = None, kernel: int = 3, stride: int = 1):
        return (
            Conv(out_channels, stride=stride, kernel_size=kernel),
            Norm(),
            LIF(state_storage=self.hparams.state_storage),
        )

    def _bottleneck(self, shortcut: bool = True):
        net = (*self._conv(), *self._conv())
        if shortcut:
            storage = Storage()
            return [Store(storage), *net, Store(storage), Residual(storage)]
        else:
            return net

    def _sppf(self, out_channels: int = None):
        storage = Storage()
        return (
            Conv(out_channels, kernel_size=1, stride=1),
            Store(storage),
            Pool("M", kernel_size=5, stride=1),
            Store(storage),
            Pool("M", kernel_size=5, stride=1),
            Store(storage),
            Pool("M", kernel_size=5, stride=1),
            Store(storage),
            Dense(storage),
            Conv(out_channels, kernel_size=1, stride=1),
        )

    def _c2f(self, out_channels: int, n: int, shortcut: bool = True):
        in_storage = Storage()
        dense_storage = Storage()
        net = []
        for _ in range(n):
            net + [self._bottleneck(shortcut), Store(dense_storage)]
        return (
            Store(in_storage),
            Conv(int(out_channels / 2), 1),
            Store(dense_storage),
            Get(in_storage),
            Conv(int(out_channels / 2), 1),
            Store(dense_storage),
            *net,
            Dense(dense_storage),
            Conv(out_channels, 1),
        )
