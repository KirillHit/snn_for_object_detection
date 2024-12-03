

import engine
import time


def train_spin(
    model: engine.Module,
    trainer: engine.Trainer,
    params_file: str,
    load_parameters=True,
    num_train_rounds=-1,
    num_round_epochs=60,
):
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
        except Exception as exc:
            print("Error description: ", exc)
            print("[ERROR]: Training stopped due to unexpected error!")
            valid = False
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print(f"[INFO]: Round {idx} fineshed at " + timestr)
        idx += 1
    print("[INFO]: Training complete")
