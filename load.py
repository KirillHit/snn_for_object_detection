from lightning.pytorch.cli import LightningCLI
from models import Yolo
from utils import PropheseeDataModule
import torch


def cli_main():
    """ LightningCLI(
        Yolo,
        PropheseeDataModule,
        parser_kwargs={
            "default_config_files": ["config/config.yaml", "config/logger.yaml"]
        },
        save_config_kwargs={"overwrite": True},
    ).model
    model.load_state_dict(torch.load("./nets/yolo_gen1.params", weights_only=True)) """

    awdawd = Yolo(2)
    print(awdawd.state_dict())


if __name__ == "__main__":
    cli_main()
