import torch
from engine.data import DataModule
from engine.model import Module
import utils.devices as devices
from utils.plots import ProgressBoard
from utils.plotter import Plotter
from tqdm import tqdm


class Trainer:
    """The base class for training models with data."""

    train_batch_idx = 0
    test_batch_idx = 0
    val_batch_idx = 0
    epoch = 0

    def __init__(self, num_gpus=0, display=True, every_n=4):
        self.display, self.every_n = display, every_n
        self.gpus = [devices.gpu(i) for i in range(min(num_gpus, devices.num_gpus()))]
        self.board = ProgressBoard(yscale="log", display=self.display)

    def prepare(self, model: Module, data: DataModule):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = len(self.test_dataloader)
        self.num_val_batches = len(self.val_dataloader)

    def prepare_model(self, model: Module):
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [torch.Tensor.to(a, self.gpus[0]) for a in batch]
        return batch

    def fit(self, num_epochs=1):
        for self.epoch in tqdm(range(num_epochs), leave=False, desc="Epoch"):
            self.fit_epoch()

    def plot(self, loss, split):
        if not self.display:
            return
        match split:
            case "train":
                self.board.draw(
                    self.train_batch_idx,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Train loss",
                    every_n=self.every_n,
                )
            case "test":
                self.board.draw(
                    self.train_batch_idx
                    - self.num_test_batches
                    + self.test_batch_idx % self.num_test_batches,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Test loss",
                    every_n=self.every_n,
                )
            case "val":
                self.board.draw(
                    self.train_batch_idx
                    - self.num_val_batches
                    + self.val_batch_idx % self.num_test_batches,
                    torch.Tensor.to(loss, devices.cpu()),
                    "Val loss",
                    every_n=self.every_n,
                )
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')

    def fit_epoch(self):
        # Training
        self.model.train()
        for batch in tqdm(self.train_dataloader, leave=False, desc="Batch: "):
            train_loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                train_loss.backward()
                self.optim.step()
                self.plot(train_loss, split="train")
            self.train_batch_idx += 1
        # Testing
        if self.test_dataloader is None:
            return
        self.model.eval()
        for batch in self.test_dataloader:
            with torch.no_grad():
                test_loss = self.model.training_step(self.prepare_batch(batch))
                self.plot(test_loss, split="test")
            self.test_batch_idx += 1

    def validation(self):
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                val_loss = self.model.validation_step(self.prepare_batch(batch))
                self.plot(val_loss, split="val")
            self.val_batch_idx += 1

    def test_model(self, data: DataModule, plotter: Plotter, is_train=False):
        images, tensors, target = data.get_test_batch(
            plotter.rows * plotter.columns, is_train
        )
        if self.gpus:
            tensors = tensors.to(self.gpus[0])
        predictions = self.model.predict(tensors).to(devices.cpu())
        # predictions is array of tensor [num_pred, 6] - [class, roi, luw, luh, rdw, rdh]
        plotter.display(images, predictions, target)
