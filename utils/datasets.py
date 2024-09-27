import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from engine.data import DataModule
from utils.downloads import download_extract

from .io.psee_loader import PSEELoader


class Gen1Dataset(DataModule):
    """Event-Based Dataset to date"""

    duration: int = 1000  # ms
    time_step: int = 1  # ms

    def read_data(self, split: str):
        # Data dir: ./data/gen1/<test, train, val>
        data_dir = os.path.join(self._root, "gen1", split)
        if not os.path.isdir(data_dir):
            raise RuntimeError(f'Directory "{data_dir}" does not exist!')
        # Get files name
        gt_files = glob.glob(data_dir + "/*_bbox.npy")
        data_files = [p.replace("_bbox.npy", "_td.dat") for p in gt_files]
        # npy format: ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        # class_id 0 for cars and 1 for pedestrians
        # boxes are updated once per second
        self.gt_boxes_list = [np.load(p) for p in gt_files]
        self.events_loader = [PSEELoader(td_file) for td_file in data_files]
        features = self.parse_features()

    def parse_features(self):
        """Transforms events into a video stream"""
        # [dur, batch, c [0-negative, 1-positive], h, w]
        height, width = self.events_loader[0].get_size()
        features = torch.zeros(
            [self.duration, self.batch_size, 2, height, width],
            dtype=torch.float32,
        )
        # [self.duration, self.width, self.height, 2],
        for batch_num in range(self.batch_size):
            # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
            events = self.events_loader[batch_num].load_delta_t(self.duration * 1000)
            time_stamp = events[0]["t"] // (self.time_step * 1000)
            events[:]["t"] = (events[:]["t"] // (self.time_step * 1000)) - time_stamp
            mask = np.stack(
                [events[:]["t"], events[:]["p"], events[:]["y"], events[:]["x"]],
                axis=1,
                dtype=np.int32,
            )
            features[mask[:, 0], batch_num, mask[:, 1], mask[:, 2], mask[:, 3]] = 1
        return features

    def parse_targets(self):
        pass


class CustomDataset(Dataset):
    """A customized dataset to load the banana detection dataset."""

    def __init__(self, data: list):
        self.features, self.labels = data

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


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

    def get_labels(self):
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

    def get_labels(self):
        names = ("bananas",)
        return names
