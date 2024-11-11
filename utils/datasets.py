import glob
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from engine.data import DataModule
from prophesee_toolbox.src.io.psee_loader import PSEELoader
from random import shuffle
from typing import Generator, Iterator, List
from tqdm import tqdm
import itertools


class PropheseeDataModule(DataModule):
    """Event-Based Dataset to date"""

    name: str

    def __init__(self, root="./data", batch_size=32, num_steps=16, num_workers=4):
        super().__init__(root, num_workers=num_workers, batch_size=batch_size)
        self.num_steps = num_steps

    def read_data(self, split: str):
        # Data dir: ./data/gen1/<test, train, val>
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
        return PropheseeDataset(
            gt_files,
            data_files,
            self.time_step,
            self.num_steps,
            self.num_load_file,
            self.name
        )


class Gen1(PropheseeDataModule):
    """Returns a sequence of event histograms with a fixed time step"""

    name: str = "gen1"

    def __init__(
        self,
        root="./data",
        batch_size=32,
        num_steps=16,
        time_step=100,
        num_load_file=50,
        num_workers=4,
    ):
        super().__init__(root, batch_size, num_steps, num_workers)
        self.time_step, self.num_load_file = time_step, num_load_file

    def get_labels(self):
        names = ("car", "person")
        return names


class Megapixel(PropheseeDataModule):
    """Returns a sequence of event histograms with a fixed time step"""

    name: str = "1mpx"

    def __init__(
        self,
        root="./data",
        batch_size=32,
        num_steps=16,
        time_step=100,
        num_load_file=50,
        num_workers=4,
    ):
        super().__init__(root, batch_size, num_steps, num_workers)
        self.time_step, self.num_load_file = time_step, num_load_file

    def get_labels(self):
        names = (
            "pedestrians",
            "two wheelers",
            "cars",
            "trucks",
            "buses",
            "signs",
            "traffic lights",
        )
        return names


class PropheseeDataset(IterableDataset):
    """A customized dataset to load the banana detection dataset."""

    record_time = 60000000  # ns
    width: int
    height: int
    time_step_name: str

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
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"

        self.gt_files, self.data_files = gt_files, data_files
        self.num_steps, self.time_step = num_steps, time_step
        self.num_load_file = num_load_file
        self.time_step_ns = self.time_step * 1000
        self.duration_ns = self.time_step_ns * self.num_steps
        self.record_steps = self.record_time // self.duration_ns
        
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
                raise ValueError(f'[ERROR]: The split parameter cannot be "{name}"!')

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.samples_generator())

    def __len__(self) -> int:
        return len(self.gt_files) * self.record_steps

    def samples_generator(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        start_time = events_loader.current_time // (self.time_step_ns)
        end_time = start_time + self.num_steps
        events = events_loader.load_delta_t(self.duration_ns)
        time_stamps = (events[:]["t"] // (self.time_step_ns)) - start_time

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

    def load_generator(
        self,
    ) -> Generator[tuple[List[torch.Tensor], List[PSEELoader]], None, None]:
        """Loads num_load_file new samples from the list.
        This is necessary when working with a large number of files."""

        worker_info = get_worker_info()
        id = worker_info.id
        num_workers = worker_info.num_workers
        per_worker = len(self.gt_files) // num_workers

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

        return [
            torch.from_numpy(
                np.array(
                    [
                        gt_boxes[:][self.time_step_name] // (self.time_step_ns),
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
