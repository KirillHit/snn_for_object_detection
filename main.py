from lightning.pytorch.cli import LightningCLI
from models import SODa, Yolo
from utils import PropheseeDataModule


def cli_main():
    LightningCLI(
        SODa,
        PropheseeDataModule,
        subclass_mode_model=True,
        parser_kwargs={
            "fit": {
                "default_config_files": [
                    "config/base.yaml",
                    "config/logger.yaml",
                    "config/fit.yaml",
                ]
            },
            "validate": {
                "default_config_files": [
                    "config/base.yaml",
                    "config/logger.yaml",
                    "config/test.yaml",
                ]
            },
            "test": {
                "default_config_files": [
                    "config/base.yaml",
                    "config/logger.yaml",
                    "config/test.yaml",
                ]
            },
        },
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
