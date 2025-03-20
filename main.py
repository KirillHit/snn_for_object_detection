from lightning.pytorch.cli import LightningCLI
import utils.model_loader

def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)

if __name__ == "__main__":
    cli_main()
    
    model_loader = utils.model_loader.ModelLoader()
    data = model_loader.get_dataset()
    model = model_loader.get_model(data)
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
                model_loader.get("SaveFolder"),
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
