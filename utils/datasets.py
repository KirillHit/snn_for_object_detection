import glob
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from engine.data import DataModule
from prophesee_toolbox.src.io.psee_loader import PSEELoader
from random import shuffle
from typing import Generator, Iterator
from tqdm import tqdm
import itertools


class Gen1DataModule(DataModule):
    """Event-Based Dataset to date"""

    def __init__(self, root="./data", batch_size=32, duration=1000, num_workers=4):
        super().__init__(root, num_workers=num_workers, batch_size=batch_size)
        self.duration = duration

    def read_data(self, split: str):
        # Data dir: ./data/gen1/<test, train, val>
        data_dir = os.path.join(self._root, "gen1", split)
        if not os.path.isdir(data_dir):
            raise RuntimeError(f'Directory "{data_dir}" does not exist!')
        # Get files name
        gt_files = glob.glob(data_dir + "/*_bbox.npy")
        data_files = [p.replace("_bbox.npy", "_td.dat") for p in gt_files]

        if not data_files or not gt_files or len(data_files) != len(gt_files):
            raise RuntimeError(
                f"Directory '{data_dir}' does not contain data or data is invalid! I'm expecting: "
                f"./data/{data_dir}/*_bbox.npy (and *_td.dat). "
                "The dataset can be downloaded from this link: "
                "https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/"
            )

        dataset = self.create_dataset(gt_files, data_files)

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
            str(len(data_files)) + " " + split + " samples loaded from Gen1 dataset..."
        )

    def create_dataset(
        self, gt_files: list[str], data_files: list[str]
    ) -> IterableDataset:
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
        num_load_file=50,
        num_workers=4,
    ):
        super().__init__(root, batch_size, duration, num_workers)
        self.time_step, self.num_load_file = time_step, num_load_file

    def create_dataset(
        self, gt_files: list[str], data_files: list[str]
    ) -> IterableDataset:
        return Gen1FixedDataset(
            gt_files, data_files, self.time_step, self.duration, self.num_load_file
        )


class Gen1FixedDataset(IterableDataset):
    """A customized dataset to load the banana detection dataset."""

    record_time = 60000  # ms
    width: int = 304
    height: int = 240

    def __init__(
        self,
        gt_files: list[str],
        data_files: list[str],
        time_step: int,
        duration: int,
        num_load_file: int,
    ):
        """
        Args:
            gt_files (list[str]): List of ground truth file paths
            data_files (list[str]): List of data file paths
            time_step (int): Duration for one frame (ms)
            duration (int): Sample duration (ms)
            num_load_file (int): Number of examples loaded at a time
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"

        self.gt_files, self.data_files = gt_files, data_files
        self.duration, self.time_step = duration, time_step
        self.num_load_file = num_load_file
        self.record_steps = self.record_time // self.duration
        self.sequence_len = ((self.duration - 1) // self.time_step) + 1

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.samples_generator())

    def __len__(self) -> int:
        return self.num_load_file * self.record_steps

    def samples_generator(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        file_loader = self.load_generator()
        while True:
            gt_boxes_list, events_loaders = next(file_loader)
            for idx in range(len(self)):
                data_idx = idx % self.num_load_file
                yield self.parse_data(gt_boxes_list[data_idx], events_loaders[data_idx])

    def parse_data(
        self, gt_boxes: torch.Tensor, events_loader: PSEELoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms events into a video stream"""

        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.sequence_len, 2, self.height, self.width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        if events_loader.done:
            events_loader.reset()
        events = events_loader.load_delta_t(self.duration * 1000)
        time_stamps = events[:]["t"] // (self.time_step * 1000)

        if not time_stamps.size:
            return (features, gt_boxes[0:0])

        features[
            time_stamps[:] - time_stamps[0],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        ############ Labels preparing ############
        # Return labels format (ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        labels = gt_boxes[
            (gt_boxes[:, 0] >= time_stamps[0]) & (gt_boxes[:, 0] <= time_stamps[-1])
        ]
        labels[:, 0] -= time_stamps[0]

        return (features, labels)

    def load_generator(
        self,
    ) -> Generator[tuple[list[torch.Tensor], list[PSEELoader]], None, None]:
        """Loads num_load_file new samples from the list.
        This is necessary when working with a large number of files."""
        shuffle_file_idx = list(range(len(self.gt_files)))
        shuffle(shuffle_file_idx)
        idx_file_iter = itertools.cycle(iter(shuffle_file_idx))

        """ TODO 
        worker_info = get_worker_info()
        print(worker_info.id) """

        while True:
            labels: list[np.ndarray] = []
            loaders: list[PSEELoader] = []
            for count, file_idx in enumerate(idx_file_iter):
                if count >= self.num_load_file:
                    break
                labels.append(np.load(self.gt_files[file_idx]))
                loaders.append(PSEELoader(self.data_files[file_idx]))
            tqdm.write(f"[INFO]: {self.num_load_file} files uploaded")
            yield self.labels_prepare(labels), loaders

    def labels_prepare(self, labels: list[np.ndarray]) -> list[torch.Tensor]:
        """Converts labels from numpy.ndarray format to torch.
        Args:
            labels (list[np.ndarray]): Labels in numpy format
                * Numpy format ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
                * Tensor format (ts [ms], class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        Returns:
            list[torch.Tensor]: Labels in torch format
        """

        return [
            torch.from_numpy(
                np.array(
                    [
                        gt_boxes[:]["ts"] // (self.time_step * 1000),
                        gt_boxes[:]["class_id"],
                        gt_boxes[:]["x"] / self.width,
                        gt_boxes[:]["y"] / self.height,
                        (gt_boxes[:]["x"] + gt_boxes[:]["w"]) / self.width,
                        (gt_boxes[:]["y"] + gt_boxes[:]["h"]) / self.height,
                    ],
                    dtype=np.float32,
                )
            ).t()
            for gt_boxes in labels
        ]


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

    def create_dataset(
        self, events_loaders: list[PSEELoader], gt_boxes_list: list[np.ndarray]
    ) -> IterableDataset:
        # TODO
        raise NotImplementedError
