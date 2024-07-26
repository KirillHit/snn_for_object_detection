import utils
import engine
import models
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import time

import utils.devices

if __name__ == "__main__":
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.Normalize((0.0, 0.0, 0.0), (0.004, 0.004, 0.004))
        ]
    )
    data = utils.BananasDataset(batch_size=32, transform=transform)

    model = models.YOLO(num_classes=1)
    model.to(utils.devices.gpu())
    model.load_params()

    trainer = engine.Trainer(max_epochs=0, num_gpus=1, display=True, every_n=2)
    trainer.fit(model, data)

    #model.save_params()
    plotter = utils.Plotter(threshold=0.01, rows=2, columns=3)
    trainer.test_model(data, plotter, threshold=0.8)

    plt.ioff()
    plt.show()
