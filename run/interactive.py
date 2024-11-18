from matplotlib import pyplot as plt
from pynput import keyboard
import engine
from utils.plotter import Plotter


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


def on_press_construct(trainer: engine.Trainer):
    def on_press():
        print("[INFO]: Pause training")
        trainer.stop()

    return on_press


def interactive_spin(
    model: engine.Module, trainer: engine.trainer, plotter: Plotter, params_file: str
):
    key_listener = keyboard.GlobalHotKeys({"<ctrl>+q": on_press_construct(trainer)})
    key_listener.start()
    print("[INFO]: Press 'ctrl + q' to pause training!")

    if ask_question("Load parameters? [y/n]", default="y"):
        model.load_params(params_file)

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

    key_listener.stop()
    key_listener.join()
