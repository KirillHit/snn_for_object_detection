# see https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from typing import Dict, Callable, List, Optional, Type, Union, Tuple

import torch
import torch.nn as nn
import norse.torch as snn
from torch import Tensor
from models.modules import SumPool2d, BatchNorm2dNoBias


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2dNoBias
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.lif1 = snn.LIFCell()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.lif2 = snn.LIFCell()
        self.stride = stride

    def forward(
        self, x: Tensor, state: Union[Tuple[Tensor, Tensor], None]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if state is None:
            state = (None, None)
        new_state = [None, None]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out, new_state[0] = self.lif1(out, state[0])

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out, new_state[1] = self.lif2(out, state[1])

        return out, new_state


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2dNoBias
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.lif1 = snn.LIFCell()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.lif2 = snn.LIFCell()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.lif3 = snn.LIFCell()
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        state: Union[Tuple[Tensor, Tensor, Tensor], None],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        if state is None:
            state = (None, None, None)
        new_state = [None, None, None]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out, new_state[0] = self.lif1(out, state[0])

        out = self.conv2(out)
        out = self.bn2(out)
        out, new_state[1] = self.lif2(out, state[1])

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out, new_state[2] = self.lif3(out, state[2])

        return out, new_state


class ResNetBackbone(nn.Module):
    # Config contains [block, layers, groups, width_per_group]
    cfgs: Dict[str, tuple[nn.Module, List[int], int, int]] = {
        "m": (Bottleneck, [1, 1, 1, 1], 1, 64),
        "18": (BasicBlock, [2, 2, 2, 2], 1, 64),  # resnet
        "34": (BasicBlock, [3, 4, 6, 3], 1, 64),
        "50": (Bottleneck, [3, 4, 6, 3], 1, 64),
        "101": (Bottleneck, [3, 4, 23, 3], 1, 64),
        "152": (Bottleneck, [3, 8, 36, 3], 1, 64),
        "50_32x4d": (Bottleneck, [3, 4, 6, 3], 32, 4),  # resnext
        "101_32x8d": (Bottleneck, [3, 4, 23, 3], 32, 8),
        "101_64x4d": (Bottleneck, [3, 4, 23, 3], 64, 4),
        "50_2": (Bottleneck, [3, 4, 6, 3], 1, 64 * 2),  # wide_resnet
        "101_2": (Bottleneck, [3, 4, 23, 3], 1, 64 * 2),
    }
    
    out_channels: int = 0

    def __init__(
        self,
        cfg: str,
        in_channels=2,
        init_weights=False,
        batch_norm=False,
        zero_init_residual: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = BatchNorm2dNoBias,
    ) -> None:
        super().__init__()
        block, layers, groups, width_per_group = self.cfgs[cfg]
        self.out_channels = 512 * block.expansion

        if not batch_norm:
            norm_layer = nn.Identity
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.lif = snn.LIFCell()
        self.sumpool = SumPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2]
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.ModuleList:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.ModuleList(layers)

    def _process_layer(
        self, x: Tensor, layer: nn.ModuleList, state: Union[List[Tensor], None]
    ) -> Tuple[Tensor, List[Tensor]]:
        if state is None:
            state = [None] * len(layer)
        new_state = [None] * len(layer)
        
        for idx, block in enumerate(layer):
            x, new_state[idx] = block(x, state[idx])
        return x, new_state

    def _forward_impl(self, X: Tensor) -> Tensor:
        # See note [TorchScript super()]
        s0, s1, s2, s3, s4 = None, None, None, None, None
        spikes = []
        for x in X:
            x = self.conv1(x)
            x = self.bn1(x)
            x, s0 = self.lif(x, s0)
            x = self.sumpool(x)

            x, s1 = self._process_layer(x, self.layer1, s1)
            x, s2 = self._process_layer(x, self.layer2, s2)
            x, s3 = self._process_layer(x, self.layer3, s3)
            x, s4 = self._process_layer(x, self.layer4, s4)

            spikes.append(x)

        return torch.stack(spikes)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
