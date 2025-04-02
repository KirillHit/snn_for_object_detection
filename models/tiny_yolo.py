"""
Network configuration similar to yolo8
"""

import torch
from typing import List
from models.soda import SODa
from models.generator import *
from models.layers import *


class TinyYolo(SODa):
    """Simplified model of yolo8 without pyramid of features

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
            "n": (1 / 3, 0.25, 2.0),
            "s": (1 / 3, 0.50, 2.0),
            "m": (2 / 3, 0.75, 1.5),
            "l": (1.0, 1.00, 1.0),
            "x": (1.0, 1.25, 1.0),
        }
        # d: depth_multiple, w: width_multiple, r: ratio
        self.d, self.w, self.r = model_types[model]
        self._out_configure()
        self._prepare_net()

    def _out_configure(self):
        max = 0.75
        min = 0.08
        size_per_pix = 3

        sizes = torch.arange(min, max, (max - min) / (3 * size_per_pix))
        self.sizes = sizes.reshape((-1, size_per_pix))
        self.ratios = torch.tensor((0.5, 1.0, 2))

        self.num_anchors = size_per_pix * len(self.ratios)
        self.num_class_out = self.num_anchors * (self.hparams.num_classes + 1)
        self.num_box_out = self.num_anchors * 4

    def get_cfgs(self) -> List[LayerGen]:
        storage_detect = Storage()
        return [
            *self._conv(int(64 * self.w), 3, 2),
            *self._c2f(int(64 * self.w), int(2 * self.d)),
            *self._conv(int(128 * self.w), 3, 2),
            *self._c2f(int(128 * self.w), int(3 * self.d)),
            *self._conv(int(256 * self.w), 3, 2),
            *self._c2f(int(256 * self.w), int(4 * self.d)),
            Store(storage_detect),
            *self._conv(int(256 * self.w), 3, 2),
            *self._c2f(int(256 * self.w), int(3 * self.d)),
            Store(storage_detect),
            *self._conv(int(256 * self.w * self.r), 3, 2),
            *self._c2f(int(256 * self.w * self.r), int(2 * self.d)),
            Store(storage_detect),
            *self._detect(storage_detect, 0),
            *self._detect(storage_detect, 1),
            *self._detect(storage_detect, 2),
        ]

    def _detect(
        self,
        storage_detect: Storage,
        idx: int,
    ) -> List[LayerGen]:
        storage = Storage()
        return [
            Get(storage_detect, idx),
            Anchors(self.storage_anchor, self.sizes[idx], self.ratios),
            Conv(kernel_size=1),
            Norm(),
            LI(state_storage=self.hparams.state_storage),
            Tanh(),
            Store(storage),
            Conv(self.num_box_out, 1),
            Store(self.storage_box),
            Get(storage),
            Conv(self.num_class_out, 1),
            Store(self.storage_cls),
        ]

    def _conv(self, out_channels: int = None, kernel: int = 3, stride: int = 1):
        return (
            Conv(out_channels, stride=stride, kernel_size=kernel),
            Norm(),
            LIF(state_storage=self.hparams.state_storage),
        )

    def _bottleneck(self, shortcut: bool = True):
        net = (*self._conv(),)
        if shortcut:
            storage = Storage()
            return [Store(storage), *net, Store(storage), Residual(storage)]
        else:
            return net

    def _c2f(self, out_channels: int, n: int, shortcut: bool = True):
        in_storage = Storage()
        dense_storage = Storage()
        net = []
        for _ in range(n):
            net + [*self._bottleneck(shortcut), Store(dense_storage)]
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
