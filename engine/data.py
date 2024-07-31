from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torch


class DataModule:
    """The base class of data."""

    def __init__(
        self,
        root="./data",
        num_workers=4,
        batch_size=32,
        save_tensor=False,
        resize: v2.Resize = None,
        normalize: v2.Normalize = None,
    ):
        self._root = root
        self._num_workers = num_workers
        self._train_dataset: CustomDataset = None
        self._val_dataset: CustomDataset = None
        self.batch_size = batch_size
        self.save_tensor = save_tensor

        transform_arr = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        undo_transform_arr = []
        if resize is not None:
            transform_arr.append(resize)
        if normalize is not None:
            transform_arr.append(normalize)
            mean = torch.tensor(normalize.mean)
            std = torch.tensor(normalize.std)
            undo_transform_arr.append(
                v2.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            )
        self.transform = v2.Compose(transform_arr)
        self.undo_transform = v2.Compose(undo_transform_arr)

    def get_dataloader(self, batch_size: int, is_train=True, shuffle=True):
        self.update_dataset(is_train)

        return DataLoader(
            self._train_dataset if is_train else self._val_dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.batch_size, is_train=True, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.batch_size, is_train=False, shuffle=False)

    def update_dataset(self, is_train=True):
        if is_train:
            if self._train_dataset is not None:
                return
        elif self._val_dataset is not None:
            return
        self.read_data(is_train)

    def read_data(self, is_train=True):
        """Read the dataset images and labels."""
        raise NotImplementedError

    def __getitem__(self, idx: int):
        if self._val_dataset is None:
            self.read_data(is_train=False)
        return self._val_dataset[idx]

    def get_names(self):
        return []

    def get_test_batch(self, num: int, is_train=False):
        dataloader = self.get_dataloader(num, is_train=is_train, shuffle=True)
        images, targets = next(iter(dataloader))
        return self.undo_transform(images), images, targets


class CustomDataset(Dataset):
    """A customized dataset to load the banana detection dataset."""

    def __init__(self, data: list):
        self.features, self.labels = data

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
