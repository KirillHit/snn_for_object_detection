import torch
from torch import nn
from typing import cast, Dict, List, Union
import norse.torch as snn


class SSDNeck(nn.Module):
    # fmt: off
    """
    int: Number of output convolution layers
    "M": MaxPool2d(kernel_size=2, stride=2)
    "R": Appends a feature map to the returned list
    "S": The next convolution will have a 1*1 kernel, the rest 3*3
    "D": Dropout
    """
    cfgs: Dict[str, List[Union[str, int]]] = {
        "s10": ["R", "M", 1024, "S", 1024, "R", "M", "S", 256, 512, "R", "M", "S",
                128, 256, "R", "M", "S", 128, 256, "R", "M", "S", 128, 256, "R"], # Standard head. See https://arxiv.org/pdf/1512.02325
        "6": ["R", "M", 512, "S", 512, "R", "M", "S", 128, 256, "R", "M", "S", 128, 256, "R"]
    }
    # fmt: on

    # Description of neck output for head generation. Contains the number of channels for each map
    out_shape: List[int] = []

    def __init__(
        self, type: str, in_channels: int, init_weights=True, batch_norm=False, dropout=0.5
    ) -> None:
        """
        Args:
            type (str): std available
            in_channels (int): equal to the number of channels at the output of the CNN
            batch_norm (bool, optional): Defaults to False.
            dropout (float, optional): Defaults to 0.5.
        """
        super().__init__()
        self.net = self.make_layers(self.cfgs[type], in_channels, batch_norm, dropout)
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(
                        m.weight, mean=0.2, std=1.0
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def make_layers(
        self,
        cfg: List[Union[str, int]],
        in_channels: int,
        batch_norm: bool,
        dropout: int,
    ) -> snn.SequentialState:
        layers: List[snn.SequentialState] = [nn.Identity()]
        conv_kernel = 3
        self.return_idx: List[int] = []
        for v in cfg:
            if v == "M":
                layers += [snn.Lift(nn.MaxPool2d(kernel_size=2, stride=2))]
            elif v == "S":
                conv_kernel = 1
            elif v == "D":
                layers += [snn.Lift(nn.Dropout(p=dropout))]
            elif v == "R":
                self.out_shape.append(in_channels)
                self.return_idx.append(len(layers) - 1)
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(
                    in_channels,
                    v,
                    kernel_size=conv_kernel,
                    padding=int(conv_kernel == 3),
                )
                if batch_norm:
                    layers += [
                        snn.Lift(conv2d),
                        snn.Lift(nn.BatchNorm2d(v)),
                        snn.LIF(),
                    ]
                else:
                    layers += [snn.Lift(conv2d), snn.LIF()]
                in_channels = v
                conv_kernel = 3
        return snn.SequentialState(*layers, return_hidden=True)

    def forward(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            X (torch.Tensor): Image after convolutional network [ts, batch, in_channels, h, w]
        Returns:
            List[torch.Tensor]: Shape - [ts, batch, in_channels, h, w] a list of several feature maps of different sizes
        """
        maps, state = self.net(X)
        return [maps[idx] for idx in self.return_idx]
