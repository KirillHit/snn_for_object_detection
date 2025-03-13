"""
Script for background network training

Does not create windows and saves training progress to the ``log`` folder.
"""

import engine
import time


def train_spin(
    model: engine.Model,
    trainer: engine.Trainer,
    params_file: str,
    save_folder: str,
    load_parameters=True,
    num_train_rounds=-1,
    num_round_epochs=60,
):
    """Script for background network training

    Does not create windows and saves training progress to the ``log`` folder.

    :param model: Network model.
    :type model: engine.Model
    :param trainer: Training tool.
    :type trainer: engine.Trainer
    :param params_file: Parameters file name. See :class:`engine.model.Model.load_params`.
    :type params_file: str
    :param save_folder: Folder for saving parameters.
    :type save_folder: str
    :param load_parameters: If True loads parameters from a file, otherwise initializes
        the model again. Defaults to True.
    :type load_parameters: bool, optional
    :param num_train_rounds: Number of training rounds. If -1 the training will continue
        until the user stops the process. Defaults to -1.
    :type num_train_rounds: int, optional
    :param num_round_epochs: Number of epochs in one round. Defaults to 60.
    :type num_round_epochs: int, optional
    """
    start_time = time.strftime("%Y%m%d_%H%M%S")
    print(f"[INFO]: Start time: {start_time}")

    if load_parameters:
        model.load_params(params_file)

    idx = 1
    valid = True
    while valid and (num_train_rounds == -1 or idx <= num_train_rounds):
        print(f"[INFO]: Starting round {idx}  of {num_round_epochs} epoch")
        try:
            trainer.fit(num_round_epochs)
        except KeyboardInterrupt:
            print("[INFO]: Training was stopped!")
            valid = False
        except RuntimeError as exc:
            print("Error description: ", exc)
            print("[ERROR]: Training stopped due to error!")
            valid = False
        except Exception as exc:
            print("Error description: ", exc)
            print("[ERROR]: Training stopped due to unexpected error!")
            valid = False
        timestr = time.strftime("%Y%m%d_%H%M%S")
        model.save_params(f"{params_file}_round{idx}", f"archive/{save_folder}/{start_time}")
        trainer.board.save_plot(params_file, f"archive/{save_folder}/{start_time}")
        print(f"[INFO]: Round {idx} completed at {timestr} on batch number {trainer.train_batch_idx}")
        idx += 1
    print("[INFO]: Training complete")
