import utils
import utils.devices
import run

if __name__ == "__main__":
    model_loader = utils.ModelLoader()
    data = model_loader.get_train_dataset()
    model = model_loader.get_model()
    model.to(utils.devices.gpu())
    trainer = model_loader.get_trainer()
    trainer.prepare(model, data)
    params_file = model_loader.get_params_file_name()

    match model_loader.get("Mode"):
        case 1:
            run.interactive_spin(model, trainer, model_loader.get_plotter(data), params_file)
        case 2:
            run.train_spin(
                model,
                trainer,
                params_file,
                model_loader.get("LoadParameters"),
                model_loader.get("NumTrainRounds"),
                model_loader.get("NumRoundEpoch"),
            )
        case _:
            raise RuntimeError("Wrong mode!")
