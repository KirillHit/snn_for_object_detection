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
        case 3:
            data_loader = data.test_dataloader()
            dataloader_iter = iter(data_loader)
            tensors, targets = next(dataloader_iter)
            plotter = model_loader.get_plotter(data)
            plotter.display(tensors, None, targets)
        case _:
            raise RuntimeError("Wrong mode!")
