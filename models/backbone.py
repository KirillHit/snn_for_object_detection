import torch
from torch import nn
from typing import Any, cast, Dict, List, Optional, Union
import norse.torch as snn


class Backbone(nn.Module):
    net: snn.SequentialState

    def __init__(
        self,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="sigmoid"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes, state = self.net(x)
        return spikes

class VggBased(Backbone):
    # fmt: off
    cfgs: Dict[str, List[Union[str, int]]] = {
        "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    # fmt: on
    def __init__(
        self,
        model: str,
        in_channels=2,
        init_weights=True,
        batch_norm=False,
    ) -> None:
        super().__init__(init_weights)

        self.net = self.make_layers(model, batch_norm, in_channels)

    def make_layers(
        cfg: List[Union[str, int]], batch_norm: bool, in_channels: int
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [snn.Lift(nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        snn.Lift(nn.BatchNorm2d(v)),
                        snn.LIF(),
                    ]
                else:
                    layers += [conv2d, snn.LIF()]
                in_channels = v
        return snn.SequentialState(*layers)
