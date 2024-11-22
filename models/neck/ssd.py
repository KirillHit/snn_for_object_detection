import torch
from torch import nn
from typing import cast, Dict, List, Union
import norse.torch as snn
from models.modules import SumPool2d


class SSDNeck(nn.Module):
    # fmt: off
    """
    int: Number of output convolution layers
    "M": MaxPool2d(kernel_size=2, stride=2)
    "A": AvgPool2d(kernel_size=2, stride=2)
    "S": SumPool2d(kernel_size=2, stride=2)
    "R": Appends a feature map to the returned list
    "O": The next convolution will have a 1*1 kernel, the rest 3*3
    "D": Dropout
    """
    cfgs: Dict[str, List[Union[str, int]]] = {
        "s10": ["R", "M", 1024, "O", 1024, "R", "M", "O", 256, 512, "R", "M", "O",
                128, 256, "R", "M", "O", 128, 256, "R", "M", "O", 128, 256, "R"], # Standard head. See https://arxiv.org/pdf/1512.02325
        "6": ["R", 512, "O", 512, "S", 512, "O", 512, "S", "R", 512, "O", 512, "S", "R"],
        "3": ["R", 128, "S", "R", 128, "S", "R", 128, "S", "R"],
    }
    # fmt: on

    # Description of neck output for head generation. Contains the number of channels for each map
    out_shape: List[int] = []

    def __init__(
        self,
        type: str,
        in_channels: int,
        init_weights=True,
        batch_norm=False,
        dropout=0.5,
    ) -> None:
        """
        Args:
            type (str): std available
            in_channels (int): equal to the number of channels at the output of the CNN
            batch_norm (bool, optional): Defaults to False.
            dropout (float, optional): Defaults to 0.5.
        """
        super().__init__()
        self.make_layers(self.cfgs[type], in_channels, batch_norm, dropout)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.5, std=0.1)
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
    ) -> None:
        layers: List[nn.Module] = [nn.Identity()]
        conv_kernel = 3
        return_idx: List[int] = []
        state_layer: List[int] = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif v == "S":
                layers += [SumPool2d(kernel_size=2, stride=2)]
            elif v == "O":
                conv_kernel = 1
            elif v == "D":
                layers += [nn.Dropout(p=dropout)]
            elif v == "R":
                self.out_shape.append(in_channels)
                return_idx.append(len(layers) - 1)
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
                        conv2d,
                        nn.BatchNorm2d(v),
                        snn.LIFCell(),
                    ]
                else:
                    layers += [conv2d, snn.LIFCell()]
                state_layer.append(len(layers) - 1)
                in_channels = v
                conv_kernel = 3
        self.state_full = [idx in state_layer for idx in range(len(layers))]
        self.return_layer = []
        layer_idx = 0
        for idx in range(len(layers)):
            if idx in return_idx:
                self.return_layer.append(layer_idx)
                layer_idx += 1
            else:
                self.return_layer.append(None)
        self.net = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            X (torch.Tensor): Image after convolutional network [ts, batch, in_channels, h, w]
        Returns:
            List[torch.Tensor]: Shape - [ts, batch, in_channels, h, w] a list of several feature maps of different sizes
        """
        states = [None] * len(self.net)
        spikes_list = [[] for _ in range(len(self.out_shape))]
        for ts in range(X.shape[0]):
            Z = X[ts]
            for idx, layer in enumerate(self.net):
                if self.state_full[idx]:
                    Z, states[idx] = layer(Z, states[idx])
                else:
                    Z = layer(Z)
                layer_idx = self.return_layer[idx]
                if layer_idx is not None:
                    spikes_list[layer_idx].append(Z)
        return [torch.stack(spikes) for spikes in spikes_list]
