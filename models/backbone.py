import torch
from typing import Dict
from models.modules import *
from typing import Tuple

#####################################################################
#                        Backbone Generator                         #
#####################################################################


class BackboneGen(ModelGen):
    def _load_cfg(self):
        self.default_cfgs.update(vgg())
        self.default_cfgs.update(resnet())

    def forward_impl(
        self, X: torch.Tensor, state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        return self.net(X, state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Input tensor. Shape is [ts, batch, p, h, w].
        Returns:
            torch.Tensor.
        """
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
    def vgg_block(out_channels: int, kernel: int = 3):
        return Conv(out_channels, kernel), Norm(), LIF()

    # fmt: off
    cfgs: Dict[str, ListGen] = {
        "vgg3": [*vgg_block(8), Pool("S"), *vgg_block(32), Pool("S"), *vgg_block(64), Pool("S")],
        "vgg6": [*vgg_block(32, 7), *vgg_block(32, 1), Pool("S"), *vgg_block(64, 5), *vgg_block(64, 1), Pool("S"),
                 *vgg_block(128), *vgg_block(128, 1), Pool("S")],
    }
    # fmt: on
    return cfgs


def resnet() -> Dict[str, ListGen]:
    def res_block(out_channels: int, kernel: int = 3):
        return [
            [
                Conv(out_channels, kernel),
                Norm(),
                LIF(),
                Conv(out_channels, 1),
                Norm(),
                LIF(),
            ],
            [Conv(out_channels, 1)],
        ]

    # fmt: off
    cfgs: Dict[str, ListGen] = {
        "res5": [Conv(32, 7), Norm(), LIF(), Pool("S"), res_block(64, 5), res_block(64),
                 Pool("S"), res_block(128), Pool("S")],
    }
    # fmt: on
    return cfgs
