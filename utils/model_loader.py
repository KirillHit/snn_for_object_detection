"""Generates a model and dataset based on parameters from a configuration file"""

import yaml
from torch.nn.utils import parameters_to_vector as p2v
import engine
import models
import utils
import utils.devices
from typing import Any


class ModelLoader:
    """Generates a model and dataset based on parameters from a configuration file

    For details see the source code.
    """

    def __init__(self, cfg_path="config/config.yaml"):
        with open(cfg_path, "r") as f:
            self.data = yaml.load(f, Loader=yaml.SafeLoader)
        if self.get("Mode") != 1:
            self.data["Display"] = False
        self.print_info()

    def get_train_dataset(self) -> engine.DataModule:
        return utils.STProphesee(
            self.get("Dataset"),
            batch_size=self.get("BatchSize"),
            time_step=self.get("TimeStep"),
            num_steps=self.get("NumSteps"),
            num_load_file=self.get("NumLoadFile"),
            num_workers=self.get("NumWorkers"),
        )

    def get_test_dataset(self) -> engine.DataModule:
        return utils.MTProphesee(
            self.get("Dataset"),
            batch_size=self.get("TestBatchSize"),
            time_step=self.get("TimeStep"),
            num_steps=self.get("TestNumSteps"),
            num_load_file=self.get("NumLoadFile"),
            num_workers=self.get("NumWorkers"),
        )



    def get_model(self, data: engine.DataModule) -> engine.Model:
        config_gen : models.BaseConfig = models.config_list[self.get("Model")]()
        
        backbone_net = models.BackboneGen(
            config_gen,
            in_channels=2,
            init_weights=self.get("InitWeights"),
        )
        neck_net = models.NeckGen(
            config_gen,
            backbone_net.out_channels,
            init_weights=self.get("InitWeights"),
        )
        head_net = models.Head(
            config_gen,
            len(data.get_labels()),
            neck_net.out_shape,
            self.get("InitWeights"),
        )

        model = models.SODa(
            backbone_net,
            neck_net,
            head_net,
            loss_ratio=self.get("LossRatio"),
            time_window=self.get("TimeWindow"),
        )
        print(f"[INFO]: Number of model parameters: {p2v(model.parameters()).numel()}")

        return model

    def get_progress_board(self) -> utils.ProgressBoard:
        return utils.ProgressBoard(
            yscale="log",
            xlabel="Batch idx",
            ylabel="Average loss",
            display=self.get("Display"),
            ylim=(1.2, 0.1),
            every_n=self.get("EveryN"),
        )

    def get_params_file_name(self) -> str:
        return (
            f"{self.get("Model")}_{self.get("Dataset")}"
        )

    def get_trainer(self):
        return engine.Trainer(
            self.get_progress_board(),
            gpu_index=self.get("gpu_index"),
            epoch_size=self.get("EpochSize"),
        )

    def get_plotter(self, data: engine.DataModule) -> utils.Plotter:
        return utils.Plotter(
            threshold=self.get("PlotterThreshold"),
            labels=data.get_labels(),
            interval=self.get("TimeStep"),
            columns=self.get("PlotterColumns"),
        )

    def get_evaluate(self, data: engine.DataModule) -> utils.SODAeval:
        return utils.SODAeval(labelmap=data.get_labels())

    def get(self, name: str) -> Any:
        """Get data from configuration file

        :param name: Parameter name
        :type name: str
        :return: Parameter value
        :rtype: Any
        :raises KeyError: The key was not found in the set of existing keys
        """
        return self.data[name]

    def print_info(self) -> None:
        """Prints basic information from the model configuration"""
        print(
            "[INFO]: Training parameters:\n"
            f"\tMode:{self.get("Mode")}\n"
            f"\tNumTrainRounds:{self.get("NumTrainRounds")}\n"
            f"\tNumRoundEpoch:{self.get("NumRoundEpoch")}\n"
            "\tModel architecture:\n"
            f"\tModel: {self.get("Model")}\n"
            f"\t\tInitWeights: {self.get("InitWeights")}\n"
            f"\t\tLossRatio: {self.get("LossRatio")}\n"
            f"\tDataset: {self.get("Dataset")}\n"
            f"\t\tBatchSize: {self.get("BatchSize")}\n"
            f"\t\tTimeStep: {self.get("TimeStep")}\n"
            f"\t\tNumSteps: {self.get("NumSteps")}\n"
            f"\t\tNumLoadFile: {self.get("NumLoadFile")}\n"
            f"\t\tNumWorkers: {self.get("NumWorkers")}"
        )
