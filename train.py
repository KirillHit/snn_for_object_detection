import engine
import models
import utils
import utils.devices
import time


if __name__ == "__main__":
    data = utils.Gen1Fixed(
        batch_size=16, time_step=64, num_steps=32, num_load_file=16, num_workers=4
    )
    params_file = "gen1"
    model = models.SpikeYOLO(num_classes=2)
    model.to(utils.devices.gpu())
    board = utils.ProgressBoard(
        yscale="log",
        xlabel="Batch idx",
        ylabel="Average loss",
        display=False,
        ylim=(1.2, 0.01),
        every_n=1,
    )
    trainer = engine.Trainer(board, num_gpus=1, epoch_size=64)
    trainer.prepare(model, data)

    # model_graph = draw_graph(model, input_size=(8, 3, 256, 256), expand_nested=True, save_graph=True)

    model.load_params(params_file)

    plotter = utils.Plotter(
        threshold=0.001, labels=data.get_labels(), interval=data.time_step, columns=4
    )

    idx = 1
    valid = True
    while valid:
        num_epochs = 10
        print(f"[INFO]: Starting round {idx}  of {num_epochs} epoch")
        try:
            trainer.fit(num_epochs)
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
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model.save_params(params_file + timestr)
        print(f"[INFO]: Round {idx} fineshed at " + timestr)
        idx += 1
    board.save_plot()
    
    print("[INFO]: Training complete")
