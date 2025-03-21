from lightning.pytorch.cli import LightningCLI
import utils.model_loader


def cli_main():
    model_loader = utils.model_loader.ModelLoader()
    data = model_loader.get_dataset()
    model = model_loader.get_model(data.get_labels())
    LightningCLI(model, data)


if __name__ == "__main__":
    cli_main()
