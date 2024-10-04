import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from engine.data import DataModule
from prophesee_toolbox.src.io.psee_loader import PSEELoader


class Gen1DataModule(DataModule):
    """Event-Based Dataset to date"""

    def __init__(
        self,
        root="./data",
        batch_size=32,
        duration=1000,
    ):
        super().__init__(root, num_workers=0, batch_size=batch_size)
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
            case _:
                raise ValueError(f'The split parameter cannot be "{split}"!')

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
        batch_size=32,
        duration=1000,
        time_step=100,
    ):
        super().__init__(root, batch_size, duration)
        self.time_step = time_step

    def create_dataset(self, data: list):
        events_loaders, gt_boxes_list = data

        # Numpy format ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        # Box update frequency 1-4 Hz
        # Labels format (ts [ms], class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        labels = [
            torch.from_numpy(
                np.array(
                    [
                        gt_boxes[:]["ts"] // (self.time_step * 1000),
                        gt_boxes[:]["class_id"],
                        gt_boxes[:]["x"],
                        gt_boxes[:]["y"],
                        gt_boxes[:]["x"] + gt_boxes[:]["w"],
                        gt_boxes[:]["y"] + gt_boxes[:]["h"],
                    ],
                    dtype=np.int32,
                )
            ).t()
            for gt_boxes in gt_boxes_list
        ]

        return Gen1FixedDataset(
            (events_loaders, labels),
            self.time_step,
            self.duration,
        )


class Gen1FixedDataset(Dataset):
    """A customized dataset to load the banana detection dataset."""

    record_time = 60000  # ms
    # wight = 304
    # hight = 240

    def __init__(self, data: list, time_step, duration):
        self.events_loaders, self.gt_boxes_list = data
        self.duration, self.time_step = duration, time_step
        self.record_steps = ((self.record_time - 1) // self.duration) + 1
        self.sequence_len = ((self.duration - 1) // self.time_step) + 1

    def __getitem__(self, idx):
        return self.parse_data(idx)

    def __len__(self):
        return len(self.events_loaders) * self.record_steps

    def parse_data(self, idx):
        """Transforms events into a video stream"""
        data_idx = idx // self.record_steps
        height, width = self.events_loaders[0].get_size()

        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.sequence_len, 2, height, width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        if self.events_loaders[data_idx].done:
            self.events_loaders[data_idx].reset()
        events = self.events_loaders[data_idx].load_delta_t(self.duration * 1000)
        time_stamps = events[:]["t"] // (self.time_step * 1000)
        features[
            time_stamps[:] - time_stamps[0],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        ############ Labels preparing ############
        # Return labels format (ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        gt_boxes = self.gt_boxes_list[data_idx]
        labels = gt_boxes[
            (gt_boxes[:, 0] >= time_stamps[0]) & (gt_boxes[:, 0] <= time_stamps[-1])
        ]
        labels[:, 0] -= time_stamps[0]

        return (features, labels)


class Gen1Adaptive(Gen1DataModule):
    """Returns a sequence of event histograms with an adaptive time step"""

    def __init__(
        self,
        root="./data",
        batch_size=32,
        duration=1000,
        event_step=100,
    ):
        super().__init__(root, batch_size, duration)
        self.event_step = event_step

    def create_dataset(self, data: list):
        # TODO
        raise NotImplementedError
