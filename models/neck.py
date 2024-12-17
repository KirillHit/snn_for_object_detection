import torch
from typing import Dict, List, Tuple
from models.modules import *


#####################################################################
#                          Neck Generator                           #
#####################################################################


class NeckGen(ModelGen):
    out_shape: List[int]

    def __init__(
        self, cfg: str | ListGen, in_channels: int = 2, init_weights: bool = False
    ):
        super().__init__(cfg, in_channels, init_weights)
        self.out_shape = self.search_out(self.net_cfg)

    def search_out(self, cfg: str | ListGen) -> List[int]:
        out: List[int] = []
        for module in cfg:
            if isinstance(module, Return):
                out.append(module.out_channels)
            elif isinstance(module, list):
                out += self.search_out(module)
        return out

    def _load_cfg(self):
        self.default_cfgs.update(ssd())
        self.default_cfgs.update(pyramid())
        self.default_cfgs.update(yolo())

    def forward_impl(
        self, X: List[torch.Tensor], state: ListState | None
    ) -> Tuple[torch.Tensor, ListState]:
        out = []
        _, state = self.net(X, state)
        for module in self.net.modules():
            if isinstance(module, Storage):
                out.append(module.get_storage())
        return out, state

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Input tensor. Shape is [ts, batch, p, h, w].
        Returns:
            torch.Tensor.
        """
        storage: List[List[torch.Tensor]] = [[] for _ in self.out_shape]
        state = None
        for time_step_x in X:
            Y, state = self.forward_impl(time_step_x, state)
            for idx, res in enumerate(Y):
                storage[idx].append(res)
        out: List[torch.Tensor] = []
        for ret_layer in storage:
            out.append(torch.stack(ret_layer))
        return out


#####################################################################
#                       Model configurations                        #
#####################################################################


def ssd() -> Dict[str, ListGen]:
    def conv(out_channels: int, kernel: int = 3, stride: int = 1):
        return [
            [
                Conv(out_channels, stride=stride, kernel_size=kernel),
                Norm(),
                LIF(),
            ]
        ]
    
    def res_block(out_channels: int, kernel: int = 3):
        return [
            [
                [
                    [
                        Conv(out_channels, kernel),
                        Norm(),
                        LIF(),
                    ],
                    [Conv(out_channels, 1)],
                ]
            ],
            [Conv(out_channels, 1)],
        ]

    cfgs: Dict[str, ListGen] = {
        "ssd3": [
            conv(128, 7),
            Return(),
            conv(256, 5, stride=2),
            res_block(256, 5),
            Return(),
            conv(256, stride=2),
            res_block(256),
            Return(),
        ],
    }
    return cfgs


def pyramid() -> Dict[str, ListGen]:
    fun = LIF

    def block(out_channels: int, kernel: int = 3):
        return Conv(out_channels, kernel), Norm(), fun()

    cfgs: Dict[str, ListGen] = {
        "pyramid": [
            Return(),
            *block(128, 7),
            [
                [
                    Pool("S"),
                    *block(256, 5),
                    [
                        [Pool("S"), *block(256, 5), Return(), Up()],
                        [Conv(256, 1)],
                    ],
                    *block(256, 5),
                    Return(),
                    Up(),
                ],
                [Conv(256, 1)],
            ],
            *block(256, 5),
            Return(),
        ],
    }
    return cfgs


def yolo() -> Dict[str, ListGen]:
    def conv(out_channels: int, kernel: int = 3, stride: int = 1):
        return [
            [
                Conv(out_channels, stride=stride, kernel_size=kernel),
                Norm(),
                LIF(),
            ]
        ]

    def res_block(out_channels: int):
        return [
            [
                conv(out_channels, 1),
                [[conv(out_channels)], [Pass()]],
                conv(out_channels, 1),
            ]
        ]

    cfgs: Dict[str, ListGen] = {
        "yolo8": [
            res_block(128),
            [
                [
                    conv(256, 3, 2),
                    res_block(256),
                    [
                        [
                            conv(256, 3, 2),
                            res_block(256),
                            Return(),
                            Up(),
                        ],
                        [Pass()],
                    ],
                    res_block(256),
                    Return(),
                    Up(),
                ],
                [Conv(256, 1)],
            ],
            res_block(256),
            Return(),
        ],
    }
    return cfgs
