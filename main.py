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
        ]
    )
    data = utils.BananasDataset(batch_size=32, transform=transform)

    
    model = models.YOLO(num_classes=1)
    model.to(utils.devices.gpu())
    #model.load_params()
    trainer = engine.Trainer(max_epochs=10, num_gpus=1, display=True, every_n=2)
    trainer.fit(model, data)
    model.save_params()

    X = data[12][0]
    img = X.squeeze(0).permute(1, 2, 0).long()
    output = model.predict(X.to(utils.devices.gpu()))
    utils.display(img, output.cpu(), threshold=0.5)

    plt.ioff()
    plt.show()
