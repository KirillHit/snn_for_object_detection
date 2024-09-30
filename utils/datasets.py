import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2._geometry import Resize
from torchvision.transforms.v2._misc import Normalize
from engine.data import DataModule
from .io.psee_loader import PSEELoader


class Gen1DataModule(DataModule):
    """Event-Based Dataset to date"""

    def __init__(
        self,
        root="./data",
        num_workers=4,
        batch_size=32,
        resize: Resize = None,
        normalize: Normalize = None,
        duration=10000,
    ):
        super().__init__(root, num_workers, batch_size, resize, normalize)
        self.duration = duration

    def read_data(self, split: str):
        # Data dir: ./data/gen1/<test, train, val>
        data_dir = os.path.join(self._root, "gen1", split)
        if not os.path.isdir(data_dir):
            raise RuntimeError(f'Directory "{data_dir}" does not exist!')
        # Get files name
        gt_files = glob.glob(data_dir + "/*_bbox.npy")
        data_files = [p.replace("_bbox.npy", "_td.dat") for p in gt_files]

        if not data_files or not gt_files:
            raise RuntimeError(
                f"Directory '{data_dir}' does not contain data! I'm expecting: ./data/{data_dir}/*_bbox.npy (and *_td.dat). The dataset can be downloaded from this link: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/"
            )

        gt_boxes_list = [np.load(p) for p in gt_files]
        events_loaders = [PSEELoader(td_file) for td_file in data_files]

        dataset = self.create_dataset((events_loaders, gt_boxes_list))

        match split:
            case "train":
                self._train_dataset = dataset
            case "test":
                self._test_dataset = dataset
            case "val":
                self._val_dataset = dataset

        print(
            str(len(events_loaders))
            + " "
            + split
            + " samples loaded from Gen1 dataset..."
        )

    def create_dataset(self, data: list):
        raise NotImplementedError

    def get_labels(self):
        names = ("car", "person")
        return names


class Gen1Fixed(Gen1DataModule):
    """Returns a sequence of event histograms with a fixed time step"""

    def __init__(
        self,
        root="./data",
        num_workers=4,
        batch_size=32,
        resize: Resize = None,
        normalize: Normalize = None,
        duration=10000,
        time_step=100,
    ):
        super().__init__(root, num_workers, batch_size, resize, normalize, duration)
        self.time_step = time_step

    def create_dataset(self, data: list):
        events_loaders, gt_boxes_list = data

        for gt_boxes in gt_boxes_list:
            gt_boxes[:]["ts"] //= self.time_step * 1000

        return Gen1FixedDataset(
            (events_loaders, gt_boxes_list),
            self.time_step,
            self.duration,
        )


class Gen1FixedDataset(Dataset):
    """A customized dataset to load the banana detection dataset."""

    record_time = 60000  # ms

    def __init__(self, data: list, time_step, duration):
        self.events_loaders, self.gt_boxes_list = data
        self.duration, self.time_step = duration, time_step
        self.record_steps = ((self.record_time - 1) // self.duration) + 1

    def __getitem__(self, idx):
        return self.parse_data(idx)

    def __len__(self):
        return len(self.events_loaders) * self.record_steps

    def parse_data(self, idx):
        """Transforms events into a video stream"""
        data_idx = idx // self.record_steps
        height, width = self.events_loaders[0].get_size()

        ############ Features preparing ############
        # Features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.duration, 2, height, width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        events = self.events_loaders[data_idx].load_delta_t(self.duration * 1000)
        time_stamps = events[:]["t"] // (self.time_step * 1000)
        features[
            time_stamps[:] - time_stamps[0],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        ############ Labels preparing ############
        # npy format ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        # Box update frequency 1-4 Hz
        gt_boxes = self.gt_boxes_list[data_idx]
        gt_boxes_masked = gt_boxes[
            (gt_boxes[:]["ts"] >= time_stamps[0])
            & (gt_boxes[:]["ts"] <= time_stamps[-1])
        ]
        # Labels format (ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        labels = torch.tensor(
            [
                gt_boxes_masked[:]["ts"],
                gt_boxes_masked[:]["class_id"],
                gt_boxes_masked[:]["x"],
                gt_boxes_masked[:]["y"],
                gt_boxes_masked[:]["x"] + gt_boxes_masked[:]["w"],
                gt_boxes_masked[:]["y"] + gt_boxes_masked[:]["h"],
            ],
            dtype=torch.float32,
        ).t()

        return (features, labels)


class Gen1Adaptive(Gen1DataModule):
    """Returns a sequence of event histograms with an adaptive time step"""

    def __init__(
        self,
        root="./data",
        num_workers=4,
        batch_size=32,
        resize: Resize = None,
        normalize: Normalize = None,
        duration=10000,
        event_step=100,
    ):
        super().__init__(root, num_workers, batch_size, resize, normalize, duration)
        self.event_step = event_step

    def create_dataset(self, data: list):
        # TODO
        raise NotImplementedError
