import yaml
from torch.nn.utils import parameters_to_vector as p2v
import engine
import models
import utils
import utils.devices


class ModelLoader:
    """Generates a model and dataset based on parameters from a configuration file"""

    def __init__(self, cfg_path="config/config.yaml"):
        with open(cfg_path, "r") as f:
            self.data = yaml.load(f, Loader=yaml.SafeLoader)
        if self.data["Mode"] == 2:
            self.data["Display"] = False
        self.print_info()

    def get_train_dataset(self) -> engine.DataModule:
        return utils.STProphesee(
            self.data["Dataset"],
            batch_size=self.data["BatchSize"],
            time_step=self.data["TimeStep"],
            num_steps=self.data["NumSteps"],
            num_load_file=self.data["NumLoadFile"],
            num_workers=self.data["NumWorkers"],
        )

    def get_test_dataset(self) -> engine.DataModule:
        return utils.MTProphesee(
            self.data["Dataset"],
            batch_size=self.data["TestBatchSize"],
            time_step=self.data["TestTimeStep"],
            num_steps=self.data["TestNumSteps"],
            num_load_file=self.data["TestNumLoadFile"],
            num_workers=self.data["TestNumWorkers"],
        )

    def get_model(self) -> engine.Module:
        match self.data["BackboneName"]:
            case "vgg":
                backbone = models.VGGBackbone
            case _:
                raise RuntimeError("Wrong backbone name")
        match self.data["NeckName"]:
            case "ssd":
                neck = models.SSDNeck
            case _:
                raise RuntimeError("Wrong neck name")

        backbone_net = backbone(
            str(self.data["BackboneVersion"]),
            batch_norm=self.data["BatchNorm"],
            init_weights=self.data["InitWeights"],
        )
        neck_net = neck(
            str(self.data["NeckVersion"]),
            backbone_net.out_channels,
            batch_norm=self.data["BatchNorm"],
            init_weights=self.data["InitWeights"],
            dropout=self.data["Dropout"],
        )

        match self.data["Dataset"]:
            case "gen1":
                num_classes = 2
            case "1mpx":
                num_classes = 7
            case _:
                raise RuntimeError("Wrong dataset name")

        model = models.SODa(
            backbone_net,
            neck_net,
            num_classes=num_classes,
        )
        print(f"[INFO]: Number of model parameters: {p2v(model.parameters()).numel()}")

        return model

    def get_progress_board(self) -> utils.ProgressBoard:
        return utils.ProgressBoard(
            yscale="log",
            xlabel="Batch idx",
            ylabel="Average loss",
            display=self.data["Display"],
            ylim=(1.2, 0.1),
            every_n=self.data["EveryN"],
        )

    def get_params_file_name(self) -> str:
        return (
            f"{self.data["BackboneName"]}{self.data["BackboneVersion"]}-"
            f"{self.data["NeckName"]}{self.data["NeckVersion"]}-"
            f"{self.data["Dataset"]}"
        )

    def get_trainer(self):
        return engine.Trainer(
            self.get_progress_board(),
            num_gpus=self.data["NumGpus"],
            epoch_size=self.data["EpochSize"],
        )

    def get_plotter(self, data: engine.DataModule) -> utils.Plotter:
        return utils.Plotter(
            threshold=self.data["PlotterThreshold"],
            labels=data.get_labels(),
            interval=data.time_step,
            columns=self.data["PlotterColumns"],
        )
        
    def get_data(self, str):
        return self.data[str]

    def print_info(self) -> None:
        print(
            "[INFO]: Training parameters:\n"
            f"\tMode:{self.data["Mode"]}\n"
            f"\tNumTrainRounds:{self.data["NumTrainRounds"]}\n"
            f"\tNumRoundEpoch:{self.data["NumRoundEpoch"]}\n"
            "\tModel architecture:\n"
            f"\t\tBackbone: {self.data["BackboneName"]}{self.data["BackboneVersion"]}\n"
            f"\t\tNeck: {self.data["NeckName"]}{self.data["NeckVersion"]}\n"
            f"\t\tBatchNorm: {self.data["BatchNorm"]}\n"
            f"\t\tInitWeights: {self.data["InitWeights"]}\n"
            f"\t\tDropout: {self.data["Dropout"]}\n"
            f"\tDataset: {self.data["Dataset"]}\n"
            f"\t\tBatchSize: {self.data["BatchSize"]}\n"
            f"\t\tTimeStep: {self.data["TimeStep"]}\n"
            f"\t\tNumSteps: {self.data["NumSteps"]}\n"
            f"\t\tNumLoadFile: {self.data["NumLoadFile"]}\n"
            f"\t\tNumWorkers: {self.data["NumWorkers"]}"
        )
