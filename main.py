from matplotlib import pyplot as plt
from pynput import keyboard
from torch.nn.utils import parameters_to_vector as p2v

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
            print("Please respond with 'y', 'n' or number")


def ask_dataset(default: str = "gf"):
    # print(f"Select dataset: gf - Gen1, m - 1Mpx (not supported yet) (Default - {default})")
    choice = ""  # input().lower() TODO
    if choice == "":
        choice = default
    if choice == "gf":
        return utils.Gen1(
            batch_size=2, time_step=16, num_steps=256, num_load_file=16, num_workers=4
        ), "gen1"
    raise ValueError("Invalid dataset value!")


def on_press_construct(trainer: engine.Trainer):
    def on_press():
        print("[INFO]: Pause training")
        trainer.stop()
    return on_press


def generate_model(
    backbone_name: str, neck_name: str, batch_norm=False, init_weights=False
):
    match backbone_name[:3]:
        case "vgg":
            backbone = models.VGGBackbone(
                backbone_name[3:], batch_norm=batch_norm, init_weights=init_weights
            )
        case _:
            raise RuntimeError("Wrong backbone name")

    match neck_name[:3]:
        case "ssd":
            neck = models.SSDNeck(
                neck_name[3:],
                backbone.out_channels,
                batch_norm=batch_norm,
                init_weights=init_weights,
            )
        case _:
            raise RuntimeError("Wrong neck name")

    return models.SODa(backbone, neck, num_classes=2)


if __name__ == "__main__":
    data, dataset_name = ask_dataset()
    backbone_name = "vgg3"
    neck_name = "ssd3"
    params_file = f"{backbone_name}-{neck_name}-{dataset_name}"

    model = generate_model(backbone_name, neck_name, init_weights=True)
    print("Number of parameters: ", p2v(model.parameters()).numel())

    model.to(utils.devices.gpu())
    board = utils.ProgressBoard(
        yscale="log",
        xlabel="Batch idx",
        ylabel="Average loss",
        display=True,
        ylim=(1.2, 0.01),
        every_n=1,
    )
    trainer = engine.Trainer(board, num_gpus=1, epoch_size=60)
    trainer.prepare(model, data)

    # model_graph = draw_graph(model, input_size=(8, 3, 256, 256), expand_nested=True, save_graph=True)

    if ask_question("Load parameters? [y/n]", default="y"):
        model.load_params(params_file)

    key_listener = keyboard.GlobalHotKeys({'<ctrl>+q': on_press_construct(trainer)})
    key_listener.start()
    print("[INFO]: Press 'ctrl + q' to pause training!")

    plotter = utils.Plotter(
        threshold=0.01, labels=data.get_labels(), interval=data.time_step, columns=4
    )
    while True:
        num_epochs = ask_question("Start fit? [number of epochs/y/n]", default=0)
        if num_epochs is False:
            break
        try:
            if num_epochs:
                trainer.fit(num_epochs)
            trainer.test_model(plotter)
            plt.show()
        except KeyboardInterrupt:
            print("[INFO]: Training was stopped!")
            break
        except RuntimeError as exc:
            print("Error description: ", exc)
            print("[ERROR]: Training stopped due to error!")

    if ask_question("Save parameters? [y/n]"):
        model.save_params(params_file)
