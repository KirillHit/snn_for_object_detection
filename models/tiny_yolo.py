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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_net(self.get_cfgs())

    def get_cfgs(self) -> List[LayerGen]:
        return [
            *self._conv(64, 3, 2),
            *self._c2f(64, 2),
            *self._conv(128, 3, 2),
            *self._c2f(128, 3),
            *self._conv(256, 3, 2),
            *self._c2f(256, 4),
            Store(self.storage_feature),
            *self._conv(256, 3, 2),
            *self._c2f(256, 3),
            Store(self.storage_feature),
            *self._conv(256, 3, 2),
            *self._c2f(256, 2),
            Store(self.storage_feature),
            *self._detect(self.storage_feature, 0),
            *self._detect(self.storage_feature, 1),
            *self._detect(self.storage_feature, 2),
        ]

    def _detect(
        self,
        storage_detect: Storage,
        idx: int,
    ) -> List[LayerGen]:
        storage = Storage()
        return (
            Get(storage_detect, idx),
            Conv(kernel_size=1),
            Norm(),
            LI(state_storage=self.hparams.state_storage),
            Tanh(),
            Store(storage),
            Conv(self.num_box_out[idx], 1),
            Store(self.storage_box),
            Get(storage),
            Conv(self.num_class_out[idx], 1),
            Store(self.storage_cls),
        )

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
            net += [*self._bottleneck(shortcut), Store(dense_storage)]
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
