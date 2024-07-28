import utils
import engine
import models
from matplotlib import pyplot as plt
import torchvision
from torchviz import make_dot

import utils.devices

def ask_question(question, default="y"):
    valid = {"y": True, "n":False}
    prompt = " [y/n] "

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'y' or 'n'")

if __name__ == "__main__":
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.23, 0.23, 0.23), (0.12, 0.12, 0.12))
        ]
    )
    data = utils.BananasDataset(batch_size=32, transform=transform)
    model = models.YOLO(num_classes=1)
    model.to(utils.devices.gpu())
    trainer = engine.Trainer(max_epochs=1, num_gpus=1, display=True, every_n=1)
    plotter = utils.Plotter(threshold=0.001, rows=2, columns=4, labels=data.get_names())
    
    #make_dot(model(torch.unsqueeze(data[0][0].to("cuda"), dim=0)), params=dict(model.named_parameters())).render("rnn_torchviz2", format="png")
    
    if (ask_question("Load parameters?")):
        model.load_params()

    while(ask_question("Start fit?")):
        trainer.fit(model, data)
        trainer.test_model(data, plotter, is_train=False)
        plt.show()

    if (ask_question("Save parameters?")):
        model.save_params()
