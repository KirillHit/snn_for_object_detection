import glob
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from engine.data import DataModule
from prophesee_toolbox.src.io.psee_loader import PSEELoader
from random import shuffle
from typing import Generator, Iterator, List, Optional, Tuple
from tqdm import tqdm
import itertools


class PropheseeDataModule(DataModule):
    """Base class for Prophesee dataset data modules"""

    def __init__(self, name: str, root="./data", batch_size=32, num_workers=4):
        super().__init__(root, num_workers=num_workers, batch_size=batch_size)
        self.name = name
        match self.name:
            case "gen1":
                self.labels_name = ("car", "person")
            case "1mpx":
                self.labels_name = (
                    "pedestrians",
                    "two wheelers",
                    "cars",
                    "trucks",
                    "buses",
                    "signs",
                    "traffic lights",
                )
            case _:
                raise ValueError(f'[ERROR]: The name parameter cannot be "{name}"!')

    def get_labels(self):
        return self.labels_name

    def read_data(self, split: str):
        # Data dir: ./data/<gen1, 1mpx>/<test, train, val>
        data_dir = os.path.join(self._root, self.name, split)
        # Get files name
        gt_files = glob.glob(data_dir + "/*_bbox.npy")
        data_files = [p.replace("_bbox.npy", "_td.dat") for p in gt_files]

        if not data_files or not gt_files or len(data_files) != len(gt_files):
            tqdm.write(
                f"[WARN]: Directory '{data_dir}' does not contain data or data is invalid! I'm expecting: "
                f"./data/{data_dir}/*_bbox.npy (and *_td.dat). "
                "The dataset can be downloaded from this link: "
                "https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/ or"
                "https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/"
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
                raise ValueError(f'[ERROR]: The split parameter cannot be "{split}"!')

        print(
            "[INFO]: "
            + str(len(data_files))
            + " "
            + split
            + " samples loaded from "
            + self.name
            + " dataset..."
        )

    def create_dataset(
        self, gt_files: List[str], data_files: List[str]
    ) -> IterableDataset:
        raise NotImplementedError


class MTProphesee(PropheseeDataModule):
    """
    Multi-target Prophesee's gen1 and 1mpx datasets. The packages are provided in a multi-target form,
    meaning that one example can contain target labels at multiple time steps.
    Records are split into fixed-step chunks that are returned sequentially.
    Intended for testing. There is single-target dataset for training.
    """

    def __init__(
        self,
        name: str,
        root="./data",
        batch_size=4,
        num_steps=128,
        time_step=16,
        num_load_file=8,
        num_workers=4,
    ):
        super().__init__(name, root, batch_size, num_workers)
        self.time_step, self.num_load_file = time_step, num_load_file
        self.num_steps = num_steps

    def create_dataset(
        self, gt_files: List[str], data_files: List[str]
    ) -> IterableDataset:
        return MTPropheseeDataset(
            gt_files,
            data_files,
            self.time_step,
            self.num_steps,
            self.num_load_file,
            self.name,
        )


class STProphesee(PropheseeDataModule):
    """
    Single-target Prophesee gen1 and 1mpx datasets. Labeling is provided for the last time step only.
    Intended for training.
    """

    def __init__(
        self,
        name: str,
        root="./data",
        batch_size=4,
        num_steps=16,
        time_step=16,
        num_load_file=8,
        num_workers=4,
    ):
        super().__init__(name, root, batch_size, num_workers)
        self.time_step, self.num_load_file = time_step, num_load_file
        self.num_steps = num_steps

    def create_dataset(
        self, gt_files: List[str], data_files: List[str]
    ) -> IterableDataset:
        return STPropheseeDataset(
            gt_files,
            data_files,
            self.time_step,
            self.num_steps,
            self.num_load_file,
            self.name,
        )


class PropheseeDatasetBase(IterableDataset):
    """Base class for Prophesee dataset iterators"""

    record_time = 60000000  # us
    width: int
    height: int
    time_step_name: str

    def __init__(
        self,
        gt_files: List[str],
        data_files: List[str],
        time_step: int,
        num_load_file: int,
        name: str,
    ):
        """
        Args:
            gt_files (List[str]): List of ground truth file paths
            data_files (List[str]): List of data file paths
            num_load_file (int): Number of examples loaded at a time
            name (str): "gen1" or "1mpx"
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"

        self.gt_files, self.data_files = gt_files, data_files
        self.num_load_file = num_load_file
        self.time_step = time_step
        self.time_step_us = self.time_step * 1000

        match name:
            case "gen1":
                self.width = 304
                self.height = 240
                self.time_step_name = "ts"
            case "1mpx":
                self.width = 1280
                self.height = 720
                self.time_step_name = "t"
            case _:
                raise ValueError(f'[ERROR]: The dataset parameter cannot be "{name}"!')

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.samples_generator())

    def load_generator(
        self,
    ) -> Generator[Tuple[List[torch.Tensor], List[PSEELoader]], None, None]:
        """Loads num_load_file new samples from the list.
        This is necessary when working with a large number of files."""

        worker_info = get_worker_info()
        id = worker_info.id
        per_worker = len(self.gt_files) // worker_info.num_workers

        # Shuffle loaded files
        shuffle_file_idx = list(range(per_worker * id, per_worker * (id + 1)))
        shuffle(shuffle_file_idx)
        idx_file_iter = itertools.cycle(iter(shuffle_file_idx))

        while True:
            labels: List[np.ndarray] = []
            loaders: List[PSEELoader] = []
            for count, file_idx in enumerate(idx_file_iter):
                if count >= self.num_load_file:
                    break
                labels.append(np.load(self.gt_files[file_idx]))
                loaders.append(PSEELoader(self.data_files[file_idx]))
            yield self.labels_prepare(labels), loaders

    def labels_prepare(self, labels: List[np.ndarray]) -> List[torch.Tensor]:
        """Converts labels from numpy.ndarray format to torch.
        Args:
            labels (List[np.ndarray]): Labels in numpy format
                * Numpy format ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
                * Tensor format (ts [ms], class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        Returns:
            List[torch.Tensor]: Labels in torch format
        """
        for gt_boxes in labels:
            gt_boxes[:]["x"].clip(0, self.width - 1)
        return [
            torch.from_numpy(
                np.array(
                    [
                        gt_boxes[:][self.time_step_name] // (self.time_step_us),
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

    def __len__(self) -> int:
        raise NotImplementedError

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        raise NotImplementedError

    def parse_data(
        self, gt_boxes: torch.Tensor, events_loader: PSEELoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms events into a video stream"""
        raise NotImplementedError


class MTPropheseeDataset(PropheseeDatasetBase):
    def __init__(
        self,
        gt_files: List[str],
        data_files: List[str],
        time_step: int,
        num_steps: int,
        num_load_file: int,
        name: str,
    ):
        """
        Args:
            gt_files (List[str]): List of ground truth file paths
            data_files (List[str]): List of data file paths
            time_step (int): Duration for one frame (ms)
            num_steps (int): Number of time steps
            num_load_file (int): Number of examples loaded at a time
            name (str): "gen1" or "1mpx"
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"
        super().__init__(gt_files, data_files, time_step, num_load_file, name)
        self.num_steps = num_steps
        self.duration_us = self.time_step_us * self.num_steps
        self.record_steps = self.record_time // self.duration_us

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        if not self.gt_files:
            raise RuntimeError("Attempt to access unloaded part of dataset")
        file_loader = self.load_generator()
        shuffle_idx = list(range(self.num_load_file * self.record_steps))
        shuffle(shuffle_idx)
        while True:
            gt_boxes_list, events_loaders = next(file_loader)
            for idx in shuffle_idx:
                data_idx = idx % self.num_load_file
                yield self.parse_data(gt_boxes_list[data_idx], events_loaders[data_idx])

    def parse_data(
        self, gt_boxes: torch.Tensor, events_loader: PSEELoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms events into a video stream"""

        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.num_steps, 2, self.height, self.width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        if events_loader.done:
            events_loader.reset()
        start_time = events_loader.current_time // (self.time_step_us)
        end_time = start_time + self.num_steps
        events = events_loader.load_delta_t(self.duration_us)
        time_stamps = (events[:]["t"] // (self.time_step_us)) - start_time

        if not time_stamps.size:
            return (features, gt_boxes[0:0])

        features[
            time_stamps[:],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        ############ Labels preparing ############
        # Return labels format (ts, class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        labels = gt_boxes[(gt_boxes[:, 0] >= start_time) & (gt_boxes[:, 0] < end_time)]
        labels[:, 0] -= start_time

        return (features, labels)


class STPropheseeDataset(PropheseeDatasetBase):
    events_threshold = 4000

    def __init__(
        self,
        gt_files: List[str],
        data_files: List[str],
        time_step: int,
        num_steps: int,
        num_load_file: int,
        name: str,
    ):
        """
        Args:
            gt_files (List[str]): List of ground truth file paths
            data_files (List[str]): List of data file paths
            time_step (int): Duration for one frame (ms)
            num_steps (int): Number of time steps
            num_load_file (int): Number of examples loaded at a time
            name (str): "gen1" or "1mpx"
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"
        super().__init__(gt_files, data_files, time_step, num_load_file, name)
        self.num_steps = num_steps

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        if not self.gt_files:
            raise RuntimeError("Attempt to access unloaded part of dataset")
        file_loader = self.load_generator()
        while True:
            gt_boxes_list, events_loaders = next(file_loader)
            shuffle_idx = list(range(self.num_load_file))
            while shuffle_idx:
                new_shuffle_idx = []
                for idx in shuffle_idx:
                    data, res = self.parse_data(gt_boxes_list[idx], events_loaders[idx])
                    if res:
                        new_shuffle_idx.append(idx)
                    if data is not None:
                        yield data
                shuffle_idx = new_shuffle_idx
                shuffle(shuffle_idx)

    def parse_data(
        self, gt_boxes: torch.Tensor, events_loader: PSEELoader
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], bool]:
        """Transforms events into a video stream"""
        if events_loader.done:
            return None, False

        ############ Labels preparing ############
        # Return labels format (class id (0 car, 1 person), xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        start_time_us = events_loader.current_time
        start_step = start_time_us // self.time_step_us
        gt_boxes = gt_boxes[gt_boxes[:, 0].ge(start_step + self.num_steps)]
        if not gt_boxes.numel():
            return None, False
        labels = gt_boxes[gt_boxes[:, 0] == gt_boxes[0, 0]]

        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.num_steps, 2, self.height, self.width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        first_label_time_us = labels[0, 0].item() * self.time_step_us
        first_event_time_us = first_label_time_us - (self.time_step_us * self.num_steps)
        events = events_loader.load_delta_t(first_label_time_us - start_time_us)
        events = events[events[:]["t"] >= first_event_time_us]
        if (events.shape[0] // self.num_steps) < self.events_threshold:
            return None, True

        time_stamps = (events[:]["t"] - first_event_time_us) // self.time_step_us

        if not time_stamps.size:
            return None, False

        features[
            time_stamps[:],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        return (features, labels[:, 1:]), True
