import engine
from tqdm import tqdm
from utils.evaluate import SODAeval


def eval_spin(
    model: engine.Module,
    trainer: engine.Trainer,
    eval: SODAeval,
    params_file: str,
    num_eval_rounds: int,
):
    model.load_params(params_file)
    for _ in tqdm(range(num_eval_rounds), leave=False, desc="[Eval]"):
        try:
            trainer.eval_model(eval)
        except KeyboardInterrupt:
            tqdm.write("[INFO]: Eval was stopped!")
            break
        except Exception as exc:
            tqdm.write("Error description: ", exc)
            tqdm.write("[ERROR]: Eval stopped due to unexpected error!")
            break
    eval.get_eval()
    print("[INFO]: Eval complete")
