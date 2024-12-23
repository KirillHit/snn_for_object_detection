"""
Implements tools for training neural networks
"""

import torch
from engine.data import DataModule
from engine.model import Model
import utils.devices as devices
from utils.progress_board import ProgressBoard
from tqdm import tqdm


class Trainer:
    """Class for training a model on a selected dataset"""

    _train_batch_idx: int = 0
    _test_batch_idx: int = 0
    _val_batch_idx: int = 0
    _stop_flag: bool = False

    def __init__(
        self, board: ProgressBoard, gpu_index: int = 0, epoch_size: int = 60
    ) -> None:
        """
        :param board: The board that plots data points in animation.
        :type board: ProgressBoard
        :param gpu_index: CUDA index for the GPU selected for training.
            See :external:ref:`CUDA semantics <cuda-semantics>`. Defaults to 0.
        :type gpu_index: int, optional
        :param epoch_size: Size of one epoch, defaults to 60
        :type epoch_size: int, optional
        """
        self.board, self.epoch_size = board, epoch_size
        self.gpu = devices.try_gpu(gpu_index)

    def prepare(self, model: Model, data: DataModule) -> None:
        """Prepares the model and data module

        Must be called before training begins.

        :param model: Model for training
        :type model: Model
        :param data: Data used for training
        :type data: DataModule
        """
        self._prepare_data(data)
        self._prepare_model(model)
        self.optim = model.configure_optimizers()

    def _prepare_data(self, data: DataModule) -> None:
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.val_dataloader_iter = iter(self.val_dataloader)

    def _prepare_model(self, model: Model) -> None:
        if self.gpu:
            model.to(self.gpu)
        self.model = model

    def _prepare_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.gpu:
            return batch
        return batch[0].to(self.gpu), batch[1].to(self.gpu)

    def _plot(self, loss: torch.Tensor, split: str) -> None:
        match split:
            case "train":
                x = self._train_batch_idx
            case "test":
                x = (
                    self._train_batch_idx
                    - self.epoch_size
                    + self._test_batch_idx % self.epoch_size
                )
            case "val":
                x = (
                    self._train_batch_idx
                    - self.epoch_size
                    + self._val_batch_idx % self.epoch_size
                )
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')
        self.board.draw(
            x,
            loss.to(devices.cpu()).item(),
            split + " loss",
        )

    def stop(self) -> None:
        """Interrupts training

        The state is saved and training can be continued.
        """
        self._stop_flag = True

    def fit(self, num_epochs: int = 1) -> None:
        """Begins training the model

        :param num_epochs: Number of training epochs, defaults to 1.
        :type num_epochs: int, optional
        """
        self._stop_flag = False
        for self.epoch in tqdm(range(num_epochs), leave=False, desc="[Epoch]"):
            if self._stop_flag:
                return
            self.fit_epoch()
        self.test()

    def fit_epoch(self) -> None:
        """Starts one epoch of model training

        Error values are saved in :class:`utils.progress_board.ProgressBoard`,
        progress is displayed in console.
        """
        self.model.train()
        for _ in (pbar := tqdm(range(self.epoch_size), leave=False, desc="[Train]")):
            if self._stop_flag:
                return
            batch = next(self.train_dataloader_iter)
            train_loss = self.model.training_step(self._prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                train_loss.backward()
                self.optim.step()
                self._plot(train_loss, split="train")
                pbar.set_description("[Train] Loss %.4f / Progress " % train_loss)
            self._train_batch_idx += 1

    def test(self):
        """Starts one epoch of model testing

        Error values are saved in :class:`utils.progress_board.ProgressBoard`,
        progress is displayed in console.
        """
        self.model.eval()
        for _ in tqdm(range(self.epoch_size), leave=False, desc="[Test]"):
            if self._stop_flag:
                return
            batch = next(self.test_dataloader_iter)
            with torch.no_grad():
                test_loss = self.model.test_step(self._prepare_batch(batch))
                self._plot(test_loss, split="test")
            self._test_batch_idx += 1

    def validation(self):
        """Starts one epoch of model evaluation

        Error values are saved in :class:`utils.progress_board.ProgressBoard`,
        progress is displayed in console."""
        self.model.eval()
        for _ in tqdm(range(self.epoch_size), leave=False, desc="[Val] "):
            if self._stop_flag:
                return
            batch = next(self.val_dataloader_iter)
            with torch.no_grad():
                val_loss = self.model.validation_step(self._prepare_batch(batch))
                self._plot(val_loss, split="val")
            self._val_batch_idx += 1

    def predict(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the network's prediction for a random sample

        :return: Three tensors: data, predictions and targets.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        tensors, targets = next(self.test_dataloader_iter)
        if self.gpu:
            tensors = tensors.to(self.gpu)
        predictions = self.model.predict(tensors).to(devices.cpu())
        if self.gpu:
            tensors = tensors.to(devices.cpu())
        return tensors, predictions, targets
