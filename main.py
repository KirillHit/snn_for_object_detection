import torch
from matplotlib import pyplot as plt
from torchview import draw_graph
from torchvision.transforms import v2

import engine
import models
import utils
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


def ask_dataset(default: str = "gf"):
    # print(f"Select dataset: gf - Gen1Fixed, m - 1Mpx (not supported yet) (Default - {default})")
    choice = ""  # input().lower() TODO
    if choice == "":
        choice = default
    if choice == "gf":
        return utils.Gen1Fixed(batch_size=2), "gen1"
    raise ValueError("Invalid dataset value!")


if __name__ == "__main__":
    data, params_file = ask_dataset()
    data.update_dataset("train")
    loader = data.get_dataset("train")
    print(loader[23])

    """ 
    model = models.SpikeYOLO(num_classes=1)
    model.to(utils.devices.gpu())
    trainer = engine.Trainer(num_gpus=1, display=True, every_n=4)
    trainer.prepare(model, data)
    plotter = utils.Plotter(
        threshold=0.001, rows=2, columns=4, labels=data.get_labels()
    )

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
    """
