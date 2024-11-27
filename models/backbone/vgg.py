import torch
from torch import nn
from typing import cast, Dict, List, Union
import norse.torch as snn
from models.modules import SumPool2d

class VGGBackbone(nn.Module):
    # fmt: off
    cfgs: Dict[str, List[Union[str, int]]] = {
        "s11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "s13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "s16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "s19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        "6": [32, "S", 64, "S", 128, "S", 256],
        "3": [8, "S", 32, "S", 64, "S"]
    }
    # fmt: on

    out_channels: int = 0

    def __init__(
        self,
        layers: str,
        in_channels=2,
        init_weights=True,
        batch_norm=False,
    ) -> None:
        """
        Args:
            layers (str): s11, s13, s16, s19, 6 available
            in_channels (int, optional): Defaults to 2.
            init_weights (bool, optional): Defaults to True.
            batch_norm (bool, optional): Defaults to False.
        """
        super().__init__()

        self.make_layers(self.cfgs[layers], batch_norm, in_channels)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.9, std=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def make_layers(
        self, cfg: List[Union[str, int]], batch_norm: bool, in_channels: int
    ) -> None:
        state_layer: List[int] = []
        layers: List[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif v == "S":
                layers += [SumPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        snn.LIFCell(),
                    ]
                else:
                    layers += [conv2d, snn.LIFCell()]
                state_layer.append(len(layers) - 1)
                in_channels = v
        self.out_channels = in_channels
        self.state_full = [idx in state_layer for idx in range(len(layers))]
        self.net = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): [ts, batch, p, h, w]
        Returns:
            torch.Tensor: [ts, batch, p, h, w]
        """
        states = [None] * len(self.net)
        spikes = []
        for ts in range(X.shape[0]):
            Z = X[ts]
            for idx, layer in enumerate(self.net):
                if self.state_full[idx]:
                    Z, states[idx] = layer(Z, states[idx])
                else:
                    Z = layer(Z)
            spikes.append(Z)
        return torch.stack(spikes)
