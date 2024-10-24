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
    # print(f"Select dataset: gf - Gen1Fixed, m - 1Mpx (not supported yet) (Default - {default})")
    choice = ""  # input().lower() TODO
    if choice == "":
        choice = default
    if choice == "gf":
        return utils.Gen1Fixed(
            batch_size=16, time_step=64, num_steps=32, num_load_file=16, num_workers=4
        ), "gen1"
    raise ValueError("Invalid dataset value!")


if __name__ == "__main__":
    data, params_file = ask_dataset()
    model = models.SpikeYOLO(num_classes=2)
    model.to(utils.devices.gpu())
    board = utils.ProgressBoard(
        yscale="log",
        xlabel="Batch idx",
        ylabel="Average loss",
        display=False,
        ylim=(1.2, 0.01),
        every_n=10,
    )
    trainer = engine.Trainer(board, num_gpus=1, epoch_size=60)
    trainer.prepare(model, data)

    # model_graph = draw_graph(model, input_size=(8, 3, 256, 256), expand_nested=True, save_graph=True)

    if ask_question("Load parameters? [y/n]", default="y"):
        model.load_params(params_file)

    plotter = utils.Plotter(
        threshold=0.001, labels=data.get_labels(), interval=data.time_step, columns=4
    )

    num_epochs = ask_question("Start fit? [number of epochs/y/n]", default=0)
    try:
        trainer.fit(num_epochs)
    except KeyboardInterrupt:
        print("[INFO]: Training was stopped!")
    except RuntimeError as exc:
        print("Error description: ", exc)
        print("[ERROR]: Training stopped due to error!")

    model.save_params(params_file)
