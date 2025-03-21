"""Generates a model and dataset based on parameters from a configuration file"""

import yaml
import models
import utils
from typing import Any
import lightning as L
from typing import List


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

    def get_dataset(self) -> utils.datasets.PropheseeDataModule:
        return utils.datasets.PropheseeDataModule(
            num_steps=self.get("NumSteps"),
            time_shift=self.get("TimeShift"),
            name=self.get("Dataset"),
            batch_size=self.get("BatchSize"),
            time_step=self.get("TimeStep"),
            num_load_file=self.get("NumLoadFile"),
            num_workers=self.get("NumWorkers"),
        )

    def get_model(self, classes: List[str]) -> L.LightningModule:
        config_gen: models.BaseConfig = models.config_list[self.get("Model")]()
        config_gen.state_storage = self.get("Mode") == 4

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
            len(classes),
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

        return model

    def get_params_file_name(self) -> str:
        return f"{self.get('Model')}_{self.get('Dataset')}"

    def get_trainer(self):
        return L.Trainer()

    def get_plotter(self, classes: List[str]) -> utils.Plotter:
        return utils.Plotter(
            threshold=self.get("PlotterThreshold"),
            labels=classes,
            interval=self.get("TimeStep"),
            columns=self.get("PlotterColumns"),
        )

    def get_evaluate(self, classes: List[str]) -> utils.SODAeval:
        return utils.SODAeval(classes)

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
            f"\tMode:{self.get('Mode')}\n"
            f"\tSaveFolder:{self.get('SaveFolder')}\n"
            f"\tNumTrainRounds:{self.get('NumTrainRounds')}\n"
            f"\tNumRoundEpoch:{self.get('NumRoundEpoch')}\n"
            f"\tModel: {self.get('Model')}\n"
            f"\t\tInitWeights: {self.get('InitWeights')}\n"
            f"\t\tLossRatio: {self.get('LossRatio')}\n"
            f"\tDataset: {self.get('Dataset')}\n"
            f"\t\tBatchSize: {self.get('BatchSize')}\n"
            f"\t\tTimeStep: {self.get('TimeStep')}\n"
            f"\t\tNumSteps: {self.get('NumSteps')}\n"
            f"\t\tTimeWindow: {self.get('TimeWindow')}\n"
            f"\t\tTimeShift: {self.get('TimeShift')}\n"
            f"\t\tNumLoadFile: {self.get('NumLoadFile')}\n"
            f"\t\tNumWorkers: {self.get('NumWorkers')}"
        )
