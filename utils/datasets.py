"""Tool for downloading gen1 and 1mpx datasets"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from prophesee_toolbox.src.io.psee_loader import PSEELoader
from random import shuffle
from typing import Generator, Iterator, List, Optional, Tuple
import itertools
from torch.nn.utils.rnn import pad_sequence
import lightning as L


def _stack_data(batch):
    """Combines samples into a batch taking into account the time dimension"""
    features = torch.stack([sample[0] for sample in batch], dim=1)
    targets = pad_sequence(
        [sample[1] for sample in batch],
        batch_first=True,
        padding_value=-1,
    )
    return features, targets


class PropheseeDataModule(L.LightningDataModule):
    """Prophesee gen1 and 1mpx datasets"""

    def __init__(
        self,
        data_dir="./data",
        dataset="gen1",
        batch_size=4,
        num_workers=4,
        num_load_file=8,
        num_steps=32,
        time_step=16,
        time_shift=8,
        one_label=True,
    ):
        """
        :param data_dir: The directory where datasets are stored. Defaults to "./data".
        :type data_dir: str, optional
        :param dataset: The name of the dataset. Supported ``gen1`` and ``1mpx``.
        :type dataset: str, optional
        :param batch_size: Number of elements in a batch. Defaults to 4.
        :type batch_size: int, optional
        :param num_workers: A positive integer will turn on multi-process data loading with the
            specified number of loader worker processes. Defaults to 4.
        :type num_workers: int, optional
        :param num_load_file: Number of concurrently open files in each thread. Defaults to 8.
        :type num_load_file: int, optional
        :param num_steps: Number of frames. Defaults to 16.
        :type num_steps: int, optional
        :param time_step: Time between frames. Defaults to 16.
        :type time_step: int, optional
        :param time_shift: The number of time steps that labels are moved forward relative to their
            frame. Defaults to 8.
        :type time_shift: int, optional
        :param one_label: If true labeling is provided for the last time step only. Intended for training. Defaults to True.
        :type one_label: bool, optional
        :raises ValueError: Invalid dataset name.
        """
        super().__init__()
        self.save_hyperparameters()

        match self.hparams.dataset:
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
                raise ValueError(
                    f'[ERROR]: The name parameter cannot be "{self.hparams.dataset}"!'
                )

    def get_labels(self):
        """Returns a list of class names"""
        return self.labels_name

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._create_dataset(*self._get_files_list("train"))
        if stage in ("fit", "validate"):
            self.val_dataset = self._create_dataset(*self._get_files_list("val"))
        if stage in ("test", "predict"):
            self.test_dataset = self._create_dataset(*self._get_files_list("test"))

    def _get_files_list(self, split: str) -> Tuple[List[str], List[str]]:
        # Data dir: ./data/<gen1, 1mpx>/<test, train, val>
        data_dir = os.path.join(self.hparams.data_dir, self.hparams.dataset, split)
        # Get files name
        gt_files = glob.glob(data_dir + "/*_bbox.npy")
        data_files = [p.replace("_bbox.npy", "_td.dat") for p in gt_files]

        if not data_files or not gt_files or len(data_files) != len(gt_files):
            raise RuntimeError(
                f"Directory '{data_dir}' does not contain data or data is invalid! I'm expecting: "
                f"./data/{data_dir}/*_bbox.npy (and *_td.dat). "
                "The dataset can be downloaded from this link: "
                "https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/ or "
                "https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/"
            )

        return gt_files, data_files

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def _get_dataloader(self, dataset: IterableDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=_stack_data,
            persistent_workers=True,
        )

    def _create_dataset(
        self, gt_files: List[str], data_files: List[str]
    ) -> IterableDataset:
        """Initializes dataset

        :param gt_files: List of files with targets
        :type gt_files: List[str]
        :param data_files: List of files with records
        :type data_files: List[str]
        :return: Ready dataset
        :rtype: IterableDataset
        """
        if self.hparams.one_label:
            return STPropheseeDataset(
                num_steps=self.hparams.num_steps,
                time_shift=self.hparams.time_shift,
                gt_files=gt_files,
                data_files=data_files,
                time_step=self.hparams.time_step,
                num_load_file=self.hparams.num_load_file,
                name=self.hparams.dataset,
            )
        else:
            return MTPropheseeDataset(
                num_steps=self.hparams.num_steps,
                gt_files=gt_files,
                data_files=data_files,
                time_step=self.hparams.time_step,
                num_load_file=self.hparams.num_load_file,
                name=self.hparams.dataset,
            )


class PropheseeDatasetBase(IterableDataset):
    """Base class for Prophesee dataset iterators

    .. warning::

        This class can only be used as a base class for inheritance.

    The samples_generator and parse_data method must be overridden in the child class.
    """

    def __init__(
        self,
        gt_files: List[str],
        data_files: List[str],
        time_step: int,
        num_load_file: int,
        name: str,
    ):
        """
        :param gt_files: List of ground truth file paths.
        :type gt_files: List[str]
        :param data_files: List of data file paths.
        :type data_files: List[str]
        :param time_step: Time between frames.
        :type time_step: int
        :param num_load_file: Number of examples loaded at a time.
        :type num_load_file: int
        :param name: The name of the dataset to download. Supported ``gen1`` and ``1mpx``.
        :type name: str
        :raises ValueError: Invalid dataset name.
        """
        assert num_load_file > 0, "The number of loaded files must be more than zero"

        self.gt_files, self.data_files = gt_files, data_files
        self.num_load_file = num_load_file
        self.time_step = time_step
        self.time_step_us = self.time_step * 1000
        self.record_time = 60000000  # us

        match name:
            case "gen1":
                self._width = 304
                self._height = 240
                self._time_step_name = "ts"
            case "1mpx":
                self._width = 1280
                self._height = 720
                self._time_step_name = "t"
            case _:
                raise ValueError(f'[ERROR]: The dataset parameter cannot be "{name}"!')

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns an iterator for the dataset"""
        return iter(self.samples_generator())

    def _load_generator(
        self,
    ) -> Generator[Tuple[List[torch.Tensor], List[PSEELoader]], None, None]:
        """Loads num_load_file new samples from the list

        This is necessary when working with a large number of files.
        """

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
            yield self._labels_prepare(labels), loaders

    def _labels_prepare(self, labels: List[np.ndarray]) -> List[torch.Tensor]:
        """Converts labels from numpy.ndarray format to torch

        :param labels: Labels in numpy format ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        :type labels: List[np.ndarray]
        :return: Labels in torch format (ts [ms], class id, xlu, ylu, xrd, yrd)
        :rtype: List[torch.Tensor]
        """
        return [
            torch.from_numpy(
                np.array(
                    [
                        gt_boxes[:][self._time_step_name] // (self.time_step_us),
                        gt_boxes[:]["class_id"],
                        gt_boxes[:]["x"] / self._width,
                        gt_boxes[:]["y"] / self._height,
                        (gt_boxes[:]["x"] + gt_boxes[:]["w"]) / self._width,
                        (gt_boxes[:]["y"] + gt_boxes[:]["h"]) / self._height,
                    ],
                    dtype=np.float32,
                )
            ).t()
            for gt_boxes in labels
        ]

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Creates a new sample generator"""
        raise NotImplementedError

    def parse_data(
        self, gt_boxes: torch.Tensor, events_loader: PSEELoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms events into a video stream"""
        raise NotImplementedError


class MTPropheseeDataset(PropheseeDatasetBase):
    def __init__(self, num_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.duration_us = self.time_step_us * self.num_steps
        self.record_steps = self.record_time // self.duration_us

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        if not self.gt_files:
            raise RuntimeError("Attempt to access unloaded part of dataset")
        file_loader = self._load_generator()
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
        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.num_steps, 2, self._height, self._width],
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
        # Return labels format (ts, class id, xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        labels = gt_boxes[(gt_boxes[:, 0] >= start_time) & (gt_boxes[:, 0] < end_time)]
        labels[:, 0] -= start_time

        return (features, labels)


class STPropheseeDataset(PropheseeDatasetBase):
    r"""Single-target Prophesee gen1 and 1mpx iterable datasets"""

    def __init__(self, num_steps: int, time_shift: int, **kwargs):
        super().__init__(**kwargs)
        self.num_steps, self.time_shift = num_steps, time_shift
        # Minimum average number of events in a sample to use it
        self.events_threshold: int = 4000
        # Minimum acceptable box size relative to frame area
        self.box_size_threshold: float = 0.01

    def samples_generator(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        if not self.gt_files:
            raise RuntimeError("Attempt to access unloaded part of dataset")
        file_loader = self._load_generator()
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
        if events_loader.done:
            return None, False

        ############ Labels preparing ############
        # Return labels format (class id, xlu, ylu, xrd, yrd)
        # Box update frequency 1-4 Hz
        start_time_us = events_loader.current_time
        start_step = start_time_us // self.time_step_us
        gt_boxes = gt_boxes[gt_boxes[:, 0].ge(start_step + self.num_steps)]
        if not gt_boxes.numel():
            return None, False
        labels = gt_boxes[gt_boxes[:, 0] == gt_boxes[0, 0]]

        # Removing boxes that are smaller than a specified size
        lab_mask = (
            (labels[:, 4] - labels[:, 2]) * (labels[:, 5] - labels[:, 3])
        ) > self.box_size_threshold
        labels = labels[lab_mask]
        if not labels.numel():
            return None, False

        ############ Features preparing ############
        # Return features format (ts, c [0-negative, 1-positive], h, w)
        features = torch.zeros(
            [self.num_steps, 2, self._height, self._width],
            dtype=torch.float32,
        )
        # Events format ('t' [us], 'x', 'y', 'p' [1-positive/0-negative])
        first_label_time_us = labels[0, 0].item() * self.time_step_us
        first_event_time_us = first_label_time_us - (
            self.time_step_us * (self.num_steps - self.time_shift)
        )
        events = events_loader.load_delta_t(
            first_label_time_us + self.time_step_us * self.time_shift - start_time_us
        )
        events = events[events[:]["t"] >= first_event_time_us]
        if (events.shape[0] // self.num_steps) < self.events_threshold:
            return None, True

        time_stamps = (events[:]["t"] - first_event_time_us) // self.time_step_us

        if not time_stamps.size:
            return None, False

        # For some reason in 1mpx there are events that go beyond the frame boundaries
        events[:]["x"] = events[:]["x"].clip(0, self._width - 1)

        features[
            time_stamps[:],
            events[:]["p"].astype(np.uint32),
            events[:]["y"].astype(np.uint32),
            events[:]["x"].astype(np.uint32),
        ] = 1

        return (features, labels[:, 1:]), True
