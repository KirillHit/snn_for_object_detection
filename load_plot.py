from matplotlib import pyplot as plt
from pynput import keyboard

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


if __name__ == "__main__":
    board = utils.ProgressBoard(
        yscale="log",
        xlabel="Batch idx",
        ylabel="Average loss",
        display=True,
        ylim=(1.2, 0.01),
        every_n=10,
    )

    board.load_plot("20241025-101503")
    
    plt.show()

    while not ask_question("Exit? [y/n]"):
        pass
