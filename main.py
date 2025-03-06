import run
import utils.model_loader
import utils.devices

if __name__ == "__main__":
    model_loader = utils.model_loader.ModelLoader()
    data = model_loader.get_train_dataset()
    model = model_loader.get_model(data)
    model.to(utils.devices.gpu())
    trainer = model_loader.get_trainer()
    trainer.prepare(model, data)
    params_file = model_loader.get_params_file_name()

    match model_loader.get("Mode"):
        case 1:
            run.interactive_spin(
                model, trainer, model_loader.get_plotter(data), params_file
            )
        case 2:
            run.train_spin(
                model,
                trainer,
                params_file,
                model_loader.get("LoadParameters"),
                model_loader.get("NumTrainRounds"),
                model_loader.get("NumRoundEpoch"),
            )
        case 3:
            run.eval_spin(
                model,
                trainer,
                model_loader.get_evaluate(data),
                params_file,
                model_loader.get("NumEvalRounds"),
            )
        case 4:
            run.activity_test(
                model, trainer, model_loader.get_plotter(data), params_file
            )
        case _:
            raise RuntimeError("Wrong mode!")
