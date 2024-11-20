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
        if self.get("Mode") == 2:
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
            time_step=self.get("TestTimeStep"),
            num_steps=self.get("TestNumSteps"),
            num_load_file=self.get("TestNumLoadFile"),
            num_workers=self.get("TestNumWorkers"),
        )

    def get_model(self) -> engine.Module:
        match self.get("BackboneName"):
            case "vgg":
                backbone = models.VGGBackbone
            case _:
                raise RuntimeError("Wrong backbone name")
        match self.get("NeckName"):
            case "ssd":
                neck = models.SSDNeck
            case _:
                raise RuntimeError("Wrong neck name")

        backbone_net = backbone(
            str(self.get("BackboneVersion")),
            batch_norm=self.get("BatchNorm"),
            init_weights=self.get("InitWeights"),
        )
        neck_net = neck(
            str(self.get("NeckVersion")),
            backbone_net.out_channels,
            batch_norm=self.get("BatchNorm"),
            init_weights=self.get("InitWeights"),
            dropout=self.get("Dropout"),
        )

        match self.get("Dataset"):
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
            loss_ratio=self.get("LossRatio"),
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
            f"{self.get("BackboneName")}{self.get("BackboneVersion")}-"
            f"{self.get("NeckName")}{self.get("NeckVersion")}-"
            f"{self.get("Dataset")}"
        )

    def get_trainer(self):
        return engine.Trainer(
            self.get_progress_board(),
            num_gpus=self.get("NumGpus"),
            epoch_size=self.get("EpochSize"),
        )

    def get_plotter(self, data: engine.DataModule) -> utils.Plotter:
        return utils.Plotter(
            threshold=self.get("PlotterThreshold"),
            labels=data.get_labels(),
            interval=data.time_step,
            columns=self.get("PlotterColumns"),
        )

    def get(self, str):
        return self.data[str]

    def print_info(self) -> None:
        print(
            "[INFO]: Training parameters:\n"
            f"\tMode:{self.get("Mode")}\n"
            f"\tNumTrainRounds:{self.get("NumTrainRounds")}\n"
            f"\tNumRoundEpoch:{self.get("NumRoundEpoch")}\n"
            "\tModel architecture:\n"
            f"\t\tBackbone: {self.get("BackboneName")}{self.get("BackboneVersion")}\n"
            f"\t\tNeck: {self.get("NeckName")}{self.get("NeckVersion")}\n"
            f"\t\tBatchNorm: {self.get("BatchNorm")}\n"
            f"\t\tInitWeights: {self.get("InitWeights")}\n"
            f"\t\tDropout: {self.get("Dropout")}\n"
            f"\t\LossRatio: {self.get("LossRatio")}\n"
            f"\tDataset: {self.get("Dataset")}\n"
            f"\t\tBatchSize: {self.get("BatchSize")}\n"
            f"\t\tTimeStep: {self.get("TimeStep")}\n"
            f"\t\tNumSteps: {self.get("NumSteps")}\n"
            f"\t\tNumLoadFile: {self.get("NumLoadFile")}\n"
            f"\t\tNumWorkers: {self.get("NumWorkers")}"
        )
