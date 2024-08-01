import utils
import engine
import models
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from torchview import draw_graph
import torch

import utils.devices


def ask_question(question, default="y"):
    valid = {"y": True, "n": False}

    while True:
        print(question + f" (Default - {default})")
        choice = input().lower()
        if default is not None and choice == "":
            if isinstance(default, int):
                return default
            return valid[default]
        elif choice in valid:
            return valid[choice]
        elif choice.isdigit():
            return int(choice)
        else:
            print("Please respond with 'y' or 'n'")


def ask_dataset(default: str = "b"):
    while True:
        print(f"Select dataset: b-bananas, h-hardhat (Default - {default})")
        choice = input().lower()
        if default is not None and choice == "":
            choice = default
        if choice == "b":
            return utils.BananasDataset(
                batch_size=16,
                resize=v2.Resize((256, 256)),
                normalize=v2.Normalize((0.23, 0.23, 0.23), (0.12, 0.12, 0.12)),
            ), "bananas"
        elif choice == "h":
            return utils.HardHatDataset(
                batch_size=16,
                resize=v2.Resize((256, 256)),
                normalize=v2.Normalize((0.23, 0.23, 0.23), (0.12, 0.12, 0.12)),
                save_tensor=True,
            ), "hardhat"
        else:
            print("Please respond with 'y' or 'n'")


if __name__ == "__main__":
    data, params_file = ask_dataset("b")
    model = models.SpikeYOLO(num_classes=1)
    model.to(utils.devices.gpu())
    trainer = engine.Trainer(num_gpus=1, display=True, every_n=4)
    trainer.prepare(model, data)
    plotter = utils.Plotter(threshold=0.001, rows=2, columns=4, labels=data.get_names())

    # model_graph = draw_graph(model, input_size=(8, 3, 256, 256), expand_nested=True, save_graph=True)

    if ask_question("Load parameters? [y/n]"):
        model.load_params(params_file)

    while True:
        num_epochs = ask_question("Start fit? [number of epochs/y/n]", default=0)
        if num_epochs is False:
            break
        try:
            trainer.fit(num_epochs)
            trainer.test_model(data, plotter, is_train=False)
            plt.show()
        except KeyboardInterrupt:
            print("Training was stopped!")

    if ask_question("Save parameters? [y/n]"):
        model.save_params(params_file)
