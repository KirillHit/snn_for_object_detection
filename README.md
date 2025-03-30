# SODa (Spike Object Detector)

This project implements a spiking neural network for detecting road objects using an event camera.

**Open libraries are used**: [PyTorch](https://github.com/pytorch/pytorch), [norse](https://github.com/norse/norse/tree/main), [Lightning](https://github.com/Lightning-AI/pytorch-lightning)

**Supported datasets**: [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/), [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/)

**Documentation**: [kirillhit.github.io/snn_for_object_detection](https://kirillhit.github.io/snn_for_object_detection/index.html) (in progress)

## Installation

1. Install git [lfs](https://git-lfs.com/) extension.

2. Clone this repository:

``` bash
git clone https://github.com/KirillHit/snn_for_object_detection.git --recurse-submodules
cd snn_for_object_detection
```

3. Create a virtual environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

``` bash
conda env create -f environment.yml
conda activate soda_env
```

4. Before starting, you need to download one of the [Gen1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) or [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) datasets, create a `data` folder in the project directory, and transfer data to it according to the template `/data/<"gen1" or "1mpx">/<"val" or "train" or "test">/<"*_bbox.npy" and "*_td.dat">`.

5. LightningCLI is used to work with the model. Launch examples:

``` bash
python main.py fit
python main.py validate
python main.py test
python main.py predict
```

The model is configured via yaml files in the `config` folder. You can edit them or create your own and pass them as an argument. You can read more about the parameters in the [api description](https://kirillhit.github.io/snn_for_object_detection/api.html).
