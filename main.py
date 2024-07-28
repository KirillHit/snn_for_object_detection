import utils
import engine
import models
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import time
from torchviz import make_dot

import utils.devices

if __name__ == "__main__":
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.23, 0.23, 0.23), (0.12, 0.12, 0.12))
        ]
    )
    data = utils.BananasDataset(batch_size=16, transform=transform)

    model = models.SpikeYOLO(num_classes=1)
    model.to(utils.devices.gpu())
    #make_dot(model(torch.unsqueeze(data[0][0].to("cuda"), dim=0)), params=dict(model.named_parameters())).render("rnn_torchviz2", format="png")
    model.load_params()

    trainer = engine.Trainer(max_epochs=0, num_gpus=1, display=True, every_n=1)
    trainer.fit(model, data)
    #model.save_params()
    
    plotter = utils.Plotter(threshold=0.001, rows=2, columns=4, labels=data.get_names())
    trainer.test_model(data, plotter, is_train=False)

    plt.ioff()
    plt.show()
