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
            #torchvision.transforms.Normalize((0.0, 0.0, 0.0), (0.004, 0.004, 0.004))
        ]
    )
    data = utils.BananasDataset(batch_size=32, transform=transform)

    model = models.YOLO(num_classes=1)
    model.to(utils.devices.gpu())
    #make_dot(model(torch.unsqueeze(data[0][0].to("cuda"), dim=0)), params=dict(model.named_parameters())).render("rnn_torchviz2", format="png")
    model.load_params()

    trainer = engine.Trainer(max_epochs=5, num_gpus=1, display=True, every_n=2)
    trainer.fit(model, data)
    model.save_params()
    
    plotter = utils.Plotter(threshold=0.1, rows=3, columns=6, labels=data.get_names())
    trainer.test_model(data, plotter, is_train=False)

    plt.ioff()
    plt.show()
