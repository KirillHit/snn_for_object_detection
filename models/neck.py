import torch
from typing import Dict, List, Tuple
from models.modules import *


#####################################################################
#                          Neck Generator                           #
#####################################################################


class NeckGen(ModelGen):
    out_shape: List[int] = []
    
    def __init__(self, cfg, in_channels=2, init_weights=False):
        super().__init__(cfg, in_channels, init_weights)
        
        # TODO out_shape

    def _load_cfg(self):
        self.default_cfgs.update(ssd())

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
        storage: List[List[torch.Tensor]] = []
        state = None
        for time_step_x in X:
            Y, state = self.forward_impl(time_step_x, state)
            storage.append(Y)
        out: List[torch.Tensor] = []
        for idx in range(len(storage[0])):
            out.append(torch.stack(storage[:][idx]))
        return torch.stack(out)


#####################################################################
#                       Model configurations                        #
#####################################################################


def ssd() -> Dict[str, ListGen]:
    def ssd_block(out_channels: int, kernel: int = 3):
        return Conv(out_channels, kernel), Norm(), LIF()

    # fmt: off
    cfgs: Dict[str, ListGen] = {
        "ssd3": [Return(), *ssd_block(128), Pool("S"), Return(), *ssd_block(128), Pool("S"), Return(),
                 *ssd_block(128), Pool("S"), Return()],
    }
    # fmt: on
    return cfgs
