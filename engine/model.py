"""
Interface for models
"""

import os
import torch
from torch import nn
from typing import Tuple, Any


class Model(nn.Module):
    """Class of interfaces for networks"""

    def loss(
        self,
        preds: Any,
        labels: Any,
    ) -> torch.Tensor:
        """Loss calculation function
        
        This method must be overridden in child classes.

        :param preds: Predictions made by a neural network.
        :type preds: Any
        :param labels: Ground truth.
        :type labels: Any
        :raises NotImplementedError: This method is not implemented.
        :return: Value of losses
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> Any:
        """Direct network pass
        
        This method must be overridden in child classes.

        :param X: Input data
        :type X: torch.Tensor
        :raises NotImplementedError: This method is not implemented.
        :return: Result of the network operation
        :rtype: Any
        """
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Network training step

        :param batch: Training data. Contains:

            1. Input data
            2. Labels
        :type batch: Tuple[torch.Tensor, torch.Tensor]
        :return: Value of losses
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Network test step

        :param batch: Training data. Contains:

            1. Input data
            2. Labels
        :type batch: Tuple[torch.Tensor, torch.Tensor]
        :return: Value of losses
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Network validation step

        :param batch: Training data. Contains:

            1. Input data
            2. Labels
        :type batch: Tuple[torch.Tensor, torch.Tensor]
        :return: Value of losses
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns the configured optimizer for training the network"""
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Returns the network's predictions based on the input data

        :param X: Input data.
        :type X: torch.Tensor
        :return: Network Predictions.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def save_params(self, name: str) -> None:
        """Saves network weights

        :param name: Parameters file name. The file will be saved to the ``nets`` folder.
        :type name: str, optional
        """
        os.makedirs("./nets", exist_ok=True)
        if not name:
            name = name = self.__class__.__name__
        torch.save(self.state_dict(), os.path.join("./nets", name + ".params"))
        print("[INFO]: Model parameters saved")

    def load_params(self, name: str) -> None:
        """Loads network weights from a file
        
        The file must be in the ``nets`` folder and end with ``.params``

        :param name: Parameters file name. 
        :type name: str
        """
        if not name:
            name = name = self.__class__.__name__
        file = os.path.join("./nets", name + ".params")
        if not os.path.exists(file):
            print("[ERROR]: The parameters file does not exist. Check path: " + file)
            return
        self.load_state_dict(torch.load(file, weights_only=True))
        print("[INFO]: Model parameters loaded")
