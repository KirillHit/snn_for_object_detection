from torch.utils.data import Dataset, DataLoader


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
            num_workers (int, optional): A positive integer will turn on multi-process data loading with the specified number of loader worker processes. See torch.utils.data.DataLoader. Defaults to 4.
            batch_size (int, optional): _description_. Defaults to 32.
        """
        self._root = root
        self._num_workers = num_workers
        self._train_dataset: CustomDataset = None
        self._test_dataset: CustomDataset = None
        self._val_dataset: CustomDataset = None
        self.batch_size = batch_size

    def get_dataloader(self, batch_size: int, split="train", shuffle=True):
        self.update_dataset(split)
        return DataLoader(
            self.__get_dataset(split),
            batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
        )

    def __get_dataset(self, split: str):
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
        return self.get_dataloader(self.batch_size, split="train", shuffle=True)

    def test_dataloader(self):
        return self.get_dataloader(self.batch_size, split="test", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.batch_size, split="val", shuffle=False)

    def update_dataset(self, split: str):
        if self.__get_dataset(split) is None:
            self.read_data(split)

    def read_data(self, split: str):
        """Read the dataset images and labels."""
        raise NotImplementedError

    def get_labels(self):
        return []


class CustomDataset(Dataset):
    """A customized dataset to load the banana detection dataset."""

    def __init__(self, data: list):
        self.features, self.labels = data

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
