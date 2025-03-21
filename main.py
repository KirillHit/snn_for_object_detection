from lightning.pytorch.cli import LightningCLI
from models import Yolo
from utils import PropheseeDataModule

def cli_main():
    LightningCLI(
        Yolo,
        PropheseeDataModule,
        parser_kwargs={"fit": {"default_config_files": ["config/config.yaml", "config/logger.yaml"]}},
    )


if __name__ == "__main__":
    cli_main()
