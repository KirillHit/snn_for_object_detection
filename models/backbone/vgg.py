import torch
from torch import nn
from typing import cast, Dict, List, Union, Optional
import norse.torch as snn


class VGGBackbone(nn.Module):
    # fmt: off
    cfgs: Dict[str, List[Union[str, int]]] = {
        "s11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "s13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "s16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "s19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        "6": [32, "M", 64, "M", 128, 128, "M", 256, 256, "M"],
        "3": [8, "A", 32, "A", 64, "A"]
    }
    # fmt: on

    out_channels: int = 0
    states: List[Optional[snn.LIFFeedForwardState]] = None

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

        self.net = self.make_layers(self.cfgs[layers], batch_norm, in_channels)

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
        self, cfg: List[Union[str, int]], batch_norm: bool, in_channels: int
    ) -> snn.SequentialState:
        layers: List[nn.Module] = []
        for v in cfg:
            if v == "M":
                layers += [snn.Lift(nn.MaxPool2d(kernel_size=2, stride=2))]
            elif v == "A":
                layers += [snn.Lift(nn.AvgPool2d(kernel_size=2, stride=2))]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        snn.Lift(conv2d),
                        snn.Lift(nn.BatchNorm2d(v)),
                        snn.LIF(),
                    ]
                else:
                    layers += [snn.Lift(conv2d), snn.LIF()]
                in_channels = v
        self.out_channels = in_channels
        return snn.SequentialState(*layers)
    
    def detach_states(self):
        if self.states is None:
            return
        new_states = [None] * len(self.states)
        for idx, state in enumerate(self.states):
            if state is not None:
                new_state = snn.LIFFeedForwardState(
                    v=state.v.detach(),
                    i=state.i.detach(),
                )
                new_state.v.requires_grad = True
                new_states[idx] = new_state
        self.states = new_states
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.detach_states()
        spike, self.states = self.net(X, self.states)
        return spike
