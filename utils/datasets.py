import os
import PIL.Image
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import xml.etree.ElementTree as ET
from utils.downloads import download_extract
from engine.data import DataModule, CustomDataset
from torch.nn.utils.rnn import pad_sequence
import PIL


class ImagenetteDataset(DataModule):
    """Imagenette is a subset of 10 easily classified classes from Imagenet."""

    def read_data(self, is_train=True):
        data = torchvision.datasets.Imagenette(
            root=self._root,
            split="train" if is_train else "val",
            size="320px",
            transform=self.transform,
            download=not os.path.isdir(os.path.join(self._root, "imagenette2-320")),
        )

        if is_train:
            self._train_dataset = data
        else:
            self._val_dataset = data

    def get_name(self, idx: int):
        names = (
            "tench",
            "English" "springer",
            "cassette player",
            "chain saw",
            "church",
            "French horn",
            "garbage truck",
            "gas pump",
            "golf ball",
            "parachute",
        )
        return names[idx]


class Cifar10Dataset(DataModule):
    """The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class."""

    def read_data(self, is_train=True):
        data = torchvision.datasets.CIFAR10(
            root=self._root, train=is_train, transform=self.transform, download=True
        )

        if is_train:
            self._train_dataset = data
        else:
            self._val_dataset = data

    def get_name(self, idx: int):
        names = (
            "airplanes",
            "cars",
            "birds",
            "cats",
            "deer",
            "dogs",
            "frogs",
            "horses",
            "ships",
            "trucks",
        )
        return names[idx]


class BananasDataset(DataModule):
    """A customized dataset to load the banana detection dataset."""

    def read_data(self, is_train=True):
        data_dir = download_extract("banana-detection", folder=self._root)
        csv_fname = os.path.join(
            data_dir, "bananas_train" if is_train else "bananas_val", "label.csv"
        )
        csv_data = pd.read_csv(csv_fname)
        csv_data = csv_data.set_index("img_name")
        images, targets = [], []
        for img_name, target in csv_data.iterrows():
            images.append(
                self.transform(
                    PIL.Image.open(
                        os.path.join(
                            data_dir,
                            "bananas_train" if is_train else "bananas_val",
                            "images",
                            f"{img_name}",
                        )
                    )
                )
            )
            targets.append(list(target))
            # Here `target` contains (class, upper-left x, upper-left y,
            # lower-right x, lower-right y), where all the images have the same
            # banana class (index 0)
        if is_train:
            self._train_dataset = CustomDataset((images, torch.tensor(targets).unsqueeze(1) / 256))
        else:
            self._val_dataset = CustomDataset((images, torch.tensor(targets).unsqueeze(1) / 256))
        print(
            "Read "
            + str(len(targets))
            + (" training examples" if is_train else " validation examples")
        )
