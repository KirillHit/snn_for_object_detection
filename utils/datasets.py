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

    def get_names(self):
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
        return names


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

    def get_names(self):
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
        return names


class HardHatDataset(DataModule):
    """Dataset for identifying hard hat"""

    def read_data(self, is_train=True):
        data_dir = download_extract("Hardhat", folder=self._root)
        data_dir = os.path.join(data_dir, "Train" if is_train else "Test")

        targets_path = os.path.join(data_dir, "targets.pt")
        images_path = os.path.join(data_dir, "images.pt")

        if os.path.exists(targets_path) and os.path.exists(images_path):
            targets = torch.load(targets_path)
            images = torch.load(images_path)
            print("Tensors load")
        else:
            images, targets = self.parse(data_dir)
            if self.save_tensor:
                torch.save(targets, targets_path)
                torch.save(images, images_path)

        if is_train:
            self._train_dataset = CustomDataset((images, targets))
        else:
            self._val_dataset = CustomDataset((images, targets))

    def parse(self, data_dir):
        labels_dir = os.path.join(data_dir, "Annotation")
        images_dir = os.path.join(data_dir, "JPEGImage")

        images, targets = [], []
        for xml_file in os.listdir(labels_dir):
            tree = ET.parse(os.path.join(labels_dir, xml_file))
            root = tree.getroot()
            size = root.find("size")
            if size.find("depth").text != "3":
                print("Warning: wrong depth")
                continue
            w = float(size.find("width").text)
            h = float(size.find("height").text)
            labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                classes = ("helmet",)  # "head", "person"
                if name not in classes:
                    # print("Warning: wrong name - " + name + " " + xml_file)
                    continue
                coord = obj.find("bndbox")
                labels.append(
                    (
                        classes.index(name),
                        float(coord.find("xmin").text) / w,
                        float(coord.find("ymin").text) / h,
                        float(coord.find("xmax").text) / w,
                        float(coord.find("ymax").text) / h,
                    )
                )
            if not labels:
                continue
            targets.append(labels)
            filename = root.find("filename").text
            img = PIL.Image.open(os.path.join(images_dir, filename))
            images.append(self.transform(img))
        targets = pad_sequence(
            [torch.tensor(target) for target in targets],
            batch_first=True,
            padding_value=-1,
        )

        return torch.stack(images), targets

    def get_names(self):
        names = ("helmet",)
        return names


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
            self._train_dataset = CustomDataset(
                (images, torch.tensor(targets).unsqueeze(1) / 256)
            )
        else:
            self._val_dataset = CustomDataset(
                (images, torch.tensor(targets).unsqueeze(1) / 256)
            )
        print(
            "Read "
            + str(len(targets))
            + (" training examples" if is_train else " validation examples")
        )

    def get_names(self):
        names = ("bananas",)
        return names
