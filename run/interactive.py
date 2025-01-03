"""
Script for interactive learning

Shows a training graph and allows
you to pause training at any time and view the intermediate result.
"""

from matplotlib import pyplot as plt
from pynput import keyboard
import engine
from utils.plotter import Plotter
from tqdm import tqdm


def _ask_question(question, default="y") -> int | bool:
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


def _on_press_construct(trainer: engine.Trainer):
    def on_press():
        tqdm.write("[INFO]: Pause training")
        trainer.stop()

    return on_press


def interactive_spin(
    model: engine.Model, trainer: engine.Trainer, plotter: Plotter, params_file: str
):
    """Script for interactive learning
    
    Shows a training graph and allows you to pause training at any time 
    and view the intermediate result.

    :param model: Network model.
    :type model: Model
    :param trainer: Training tool.
    :type trainer: Trainer
    :param plotter: Display tool.
    :type plotter: Plotter
    :param params_file: Parameters file name. See :class:`engine.model.Model.load_params`.
    :type params_file: str
    """
    key_listener = keyboard.GlobalHotKeys({"<ctrl>+q": _on_press_construct(trainer)})
    key_listener.start()
    print("[INFO]: Press 'ctrl + q' to pause training!")

    if _ask_question("Load parameters? [y/n]", default="y"):
        model.load_params(params_file)

    while True:
        num_epochs = _ask_question("Start fit? [number of epochs/y/n]", default=0)
        if num_epochs is False:
            break
        try:
            if num_epochs:
                trainer.fit(num_epochs)
            plotter.display(*trainer.predict())
            plt.show()
        except KeyboardInterrupt:
            print("[INFO]: Training was stopped!")
            break
        except RuntimeError as exc:
            print("Error description: ", exc)
            print("[ERROR]: Training stopped due to error!")

    if _ask_question("Save parameters? [y/n]"):
        model.save_params(params_file)

    key_listener.stop()
    key_listener.join()
