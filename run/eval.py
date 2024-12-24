"""
Script for network evaluation
"""

import engine
from tqdm import tqdm
from utils.evaluate import SODAeval


def eval_spin(
    model: engine.Model,
    trainer: engine.Trainer,
    eval: SODAeval,
    params_file: str,
    num_eval_rounds: int,
):
    """Script for network evaluation

    :param model: Network model
    :type model: Model
    :param trainer: Training tool.
    :type trainer: Trainer
    :param eval: Evaluation tool.
    :type eval: SODAeval
    :param params_file: Parameters file name. See :class:`engine.model.Model.load_params`.
    :type params_file: str
    :param num_eval_rounds: Number of evaluation rounds. The resulting value is
        the average of all rounds.
    :type num_eval_rounds: int
    """
    model.load_params(params_file)
    for _ in tqdm(range(num_eval_rounds), leave=False, desc="[Eval]"):
        try:
            data, predictions, targets = trainer.predict()
            eval.add(targets, predictions[-1], data[-1])
        except KeyboardInterrupt:
            tqdm.write("[INFO]: Eval was stopped!")
            break
        except Exception as exc:
            tqdm.write("Error description: ", exc)
            tqdm.write("[ERROR]: Eval stopped due to unexpected error!")
            break
    eval.get_eval()
    print("[INFO]: Eval complete")
