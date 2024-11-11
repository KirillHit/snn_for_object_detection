from torch.utils.data import IterableDataset, DataLoader
import torch


class DataModule:
    """The base class of data."""

    def __init__(
        self,
        root="./data",
        num_workers=4,
        batch_size=32,
    ):
        """
        Args:
            root (str, optional): The directory where datasets are stored. Defaults to "./data".
            num_workers (int, optional): A positive integer will turn on multi-process data loading
                with the specified number of loader worker processes. See torch.utils.data.DataLoader.
                Defaults to 4.
            batch_size (int, optional): _description_. Defaults to 32.
        """
        self._root = root
        self._num_workers = num_workers
        self._train_dataset: IterableDataset = None
        self._test_dataset: IterableDataset = None
        self._val_dataset: IterableDataset = None
        self.batch_size = batch_size

    def get_dataloader(self, batch_size: int, split="train"):
        self.update_dataset(split)
        return DataLoader(
            self.get_dataset(split),
            batch_size,
            num_workers=self._num_workers,
            collate_fn=stack_data,
            persistent_workers=True,
        )

    def get_dataset(self, split: str):
        match split:
            case "train":
                return self._train_dataset
            case "test":
                return self._test_dataset
            case "val":
                return self._val_dataset
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')

    def train_dataloader(self):
        return self.get_dataloader(self.batch_size, split="train")

    def test_dataloader(self):
        return self.get_dataloader(self.batch_size, split="test")

    def val_dataloader(self):
        return self.get_dataloader(self.batch_size, split="val")

    def update_dataset(self, split: str):
        if self.get_dataset(split) is None:
            self.read_data(split)

    def read_data(self, split: str):
        """Read the dataset images and labels."""
        raise NotImplementedError

    def get_labels(self):
        return []


def stack_data(batch):
    features = torch.stack([sample[0] for sample in batch], dim=1)
    labels = [sample[1] for sample in batch]
    # Return features format (ts, batch, p, h, w)
    # Return labels format list[torch.Tensor]. Len = batch
    # One label contains (ts, class id, xlu, ylu, xrd, yrd)
    return features, labels
