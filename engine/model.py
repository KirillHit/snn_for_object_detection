import os
import torch
from torch import nn


class Module(nn.Module):
    """The base class of models."""

    def __init__(self):
        super().__init__()

    def loss(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels_batch: list[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, list[torch.Tensor]]
    ) -> torch.Tensor:
        raise NotImplementedError

    def test_step(self, batch: tuple[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(
        self, batch: tuple[torch.Tensor, list[torch.Tensor]]
    ) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_params(self, name: str = "") -> None:
        os.makedirs("./nets", exist_ok=True)
        if not name:
            name = name = self.__class__.__name__
        torch.save(self.state_dict(), os.path.join("./nets", name + ".params"))
        print("[INFO]: Model parameters saved")

    def load_params(self, name: str = "") -> None:
        if not name:
            name = name = self.__class__.__name__
        file = os.path.join("./nets", name + ".params")
        if not os.path.exists(file):
            print("[ERROR]: The parameters file does not exist. Check path: " + file)
            return
        self.load_state_dict(torch.load(file, weights_only=True))
        print("[INFO]: Model parameters loaded")
