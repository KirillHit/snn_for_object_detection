import torch
from engine.data import DataModule
from engine.model import Module
import utils.devices as devices
from utils.progress_board import ProgressBoard
from utils.plotter import Plotter
from tqdm import tqdm


class Trainer:
    """The base class for training models with data."""

    train_batch_idx: int = 0
    test_batch_idx: int = 0
    val_batch_idx: int = 0
    epoch: int = 0
    stop_flag: bool = False

    def __init__(self, board: ProgressBoard, num_gpus=0, epoch_size=64):
        self.board, self.epoch_size = board, epoch_size
        self.gpus = [devices.gpu(i) for i in range(min(num_gpus, devices.num_gpus()))]

    def prepare(self, model: Module, data: DataModule) -> None:
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

    def prepare_data(self, data: DataModule) -> None:
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.val_dataloader_iter = iter(self.val_dataloader)

    def prepare_model(self, model: Module) -> None:
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def prepare_batch(
        self, batch: tuple[torch.Tensor, list[torch.Tensor]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not self.gpus:
            return batch
        return batch[0].to(self.gpus[0]), [
            labels.to(self.gpus[0]) for labels in batch[1]
        ]

    def plot(self, loss, split):
        match split:
            case "train":
                self.board.draw(
                    self.train_batch_idx,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Train loss",
                )
            case "test":
                self.board.draw(
                    self.train_batch_idx
                    - self.epoch_size
                    + self.test_batch_idx % self.epoch_size,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Test loss",
                )
            case "val":
                self.board.draw(
                    self.train_batch_idx
                    - self.epoch_size
                    + self.val_batch_idx % self.epoch_size,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Val loss",
                )
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')

    def stop(self):
        self.stop_flag = True

    def fit(self, num_epochs=1):
        self.stop_flag = False
        for self.epoch in tqdm(range(num_epochs), leave=False, desc="Epoch"):
            if self.stop_flag:
                return
            self.fit_epoch()
        self.test()

    def fit_epoch(self):
        self.model.train()
        for _ in tqdm(range(self.epoch_size), leave=False, desc="Train: "):
            if self.stop_flag:
                return
            batch = next(self.train_dataloader_iter)
            train_loss = self.model.training_step(self.prepare_batch(batch))
            if not train_loss.requires_grad:
                tqdm.write(
                    "[WARN]: The loss tensor does not require a gradient. \n"
                    "        Not a single sample from the pack contains targets."
                )
                self.train_batch_idx += 1
                continue
            self.optim.zero_grad()
            with torch.no_grad():
                train_loss.backward()
                self.optim.step()
                self.plot(train_loss, split="train")
            self.train_batch_idx += 1

    def test(self):
        self.model.eval()
        for _ in tqdm(range(self.epoch_size), leave=False, desc="Test: "):
            if self.stop_flag:
                return
            batch = next(self.test_dataloader_iter)
            with torch.no_grad():
                test_loss = self.model.test_step(self.prepare_batch(batch))
                self.plot(test_loss, split="test")
            self.test_batch_idx += 1

    def validation(self):
        self.stop_flag = False
        self.model.eval()
        for _ in tqdm(range(self.epoch_size), leave=False, desc="Val: "):
            if self.stop_flag:
                return
            batch = next(self.val_dataloader_iter)
            with torch.no_grad():
                val_loss = self.model.validation_step(self.prepare_batch(batch))
                self.plot(val_loss, split="val")
            self.val_batch_idx += 1

    def test_model(self, plotter: Plotter):
        tensors, target = next(self.test_dataloader_iter)
        if self.gpus:
            tensors = tensors.to(self.gpus[0])
        predictions = self.model.predict(tensors).to(devices.cpu())
        if self.gpus:
            tensors = tensors.to(devices.cpu())
        plotter.display(tensors, predictions, target)
