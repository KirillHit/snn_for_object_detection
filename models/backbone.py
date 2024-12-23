"""
Model base generator
"""

import torch
from typing import Dict
from models.modules import *
from typing import Tuple


#####################################################################
#                        Backbone Generator                         #
#####################################################################


class BackboneGen(ModelGen):
    """Model base generator

    Returns the tensor from the last layer of the network.
    """

    def _load_cfg(self):
        self.default_cfgs.update(vgg())
        self.default_cfgs.update(resnet())

    def forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        return self.net(X, state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = []
        state = None
        for time_step_x in X:
            Y, state = self.forward_impl(time_step_x, state)
            out.append(Y)
        return torch.stack(out)


#####################################################################
#                       Model configurations                        #
#####################################################################


def vgg() -> Dict[str, ListGen]:
    """Default configuration generator

    Architectures are based on vgg.

    See source code.

    :return: Lists of layer generators.
    :rtype: Dict[str, ListGen]
    """
    fun = LIF

    def vgg_block(out_channels: int, kernel: int = 3):
        return Conv(out_channels, kernel), Norm(), fun()

    # fmt: off
    cfgs: Dict[str, ListGen] = {
        "vgg3": [*vgg_block(8), Pool("S"), *vgg_block(32), Pool("S"), *vgg_block(64), Pool("S")],
        "vgg6": [*vgg_block(32, 7), *vgg_block(32, 1), Pool("S"), *vgg_block(64, 5), *vgg_block(64, 1), Pool("S"),
                 *vgg_block(128), *vgg_block(128, 1), Pool("S")],
    }
    # fmt: on
    return cfgs


def resnet() -> Dict[str, ListGen]:
    """Default configuration generator

    Architectures include residual links.

    See source code.

    :return: Lists of layer generators.
    :rtype: Dict[str, ListGen]
    """

    def conv(out_channels: int, kernel: int = 3, stride: int = 1):
        return (
            Conv(out_channels, stride=stride, kernel_size=kernel),
            Norm(),
            LIF(),
        )

    def res_block(out_channels: int, kernel: int = 3):
        return (
            Conv(out_channels, 1),
            [
                [
                    *conv(out_channels, kernel),
                    [[*conv(out_channels, kernel)], [Conv(out_channels, 1)]],
                ],
                [Conv(out_channels, 1)],
            ],
            Conv(out_channels, 1),
        )

    cfgs: Dict[str, ListGen] = {
        "res": [*conv(64, 7, 2), *res_block(64, 5), *conv(128, 5, 2), *res_block(128)],
    }
    return cfgs
