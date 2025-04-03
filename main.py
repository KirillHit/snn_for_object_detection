from lightning.pytorch.cli import LightningCLI
from models import *
from utils import PropheseeDataModule


def cli_main():
    LightningCLI(
        SODa,
        PropheseeDataModule,
        subclass_mode_model=True,
        parser_kwargs={
            "fit": {
                "default_config_files": [
                    "config/config.yaml",
                    "config/logger.yaml",
                ]
            },
            "validate": {
                "default_config_files": [
                    "config/config.yaml",
                ]
            },
            "test": {
                "default_config_files": [
                    "config/config.yaml",
                ]
            },
            "predict": {
                "default_config_files": [
                    "config/config.yaml",
                ]
            },
        },
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
