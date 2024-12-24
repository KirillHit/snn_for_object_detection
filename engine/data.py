"""
Interface for data module
"""

from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List


class DataModule:
    """Class of interfaces for data modules

    .. warning::

        This class can only be used as a base class for inheritance.
        
    read_data and get_labels methods must be overridden in the child class.
    """

    def __init__(
        self,
        root: str = "./data",
        num_workers: int = 4,
        batch_size: int = 1,
    ):
        """
        :param root: The directory where datasets are stored. Defaults to "./data".
        :type root: str, optional
        :param num_workers: A positive integer will turn on multi-process data loading with the
            specified number of loader worker processes. Defaults to 4.
        :type num_workers: int, optional
        :param batch_size: Number of elements in a batch. Defaults to 1.
        :type batch_size: int, optional
        """
        self._root = root
        self._num_workers = num_workers
        self._train_dataset: IterableDataset = None
        self._test_dataset: IterableDataset = None
        self._val_dataset: IterableDataset = None
        self.batch_size = batch_size

    def _get_dataloader(self, batch_size: int, split="train") -> DataLoader:
        self._update_dataset(split)
        return DataLoader(
            self._get_dataset(split),
            batch_size,
            num_workers=self._num_workers,
            collate_fn=_stack_data,
            persistent_workers=True,
        )

    def _get_dataset(self, split: str) -> IterableDataset:
        match split:
            case "train":
                return self._train_dataset
            case "test":
                return self._test_dataset
            case "val":
                return self._val_dataset
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader"""
        return self._get_dataloader(self.batch_size, split="train")

    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader"""
        return self._get_dataloader(self.batch_size, split="test")

    def val_dataloader(self) -> DataLoader:
        """Returns a validation dataloader"""
        return self._get_dataloader(self.batch_size, split="val")

    def _update_dataset(self, split: str) -> None:
        if self._get_dataset(split) is None:
            self.read_data(split)

    def read_data(self, split: str) -> None:
        """Read the dataset images and labels

        :param split: "train", "test" or "val"
        :type split: str
        """
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        """Returns a list of class names"""
        return []


def _stack_data(batch):
    """Combines samples into a batch taking into account the time dimension"""
    features = torch.stack([sample[0] for sample in batch], dim=1)
    targets = pad_sequence(
        [sample[1] for sample in batch],
        batch_first=True,
        padding_value=-1,
    )
    return features, targets
