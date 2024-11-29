# Импульсная нейросеть для детекции объектов

Импульсная нейросеть для детекции объектов дорожной обстановки с использованием событийной камеры. Применяются открытые библиотеки [PyTorch](https://github.com/pytorch/pytorch) и [norse](https://github.com/norse/norse/tree/main). Поддерживаются датасеты [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) и [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).


## Запуск

Скачайте репозиторий:

``` bash
git clone https://github.com/KirillHit/spike_yolo.git
cd spike_yolo
```

Создайте виртуальное окружение с помощью [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

``` bash
conda env create -f environment.yml
conda activate spike_yolo_env
```

Запустите решение:

``` bash
python3 main.py
```

Далее выберете датасет и запустите обучение/тестирование.

## Промежуточные результаты

Пример работы сети на датасете Gen1:

![gen1_example](.images/gen1_example.gif)

Сеть основана на архитектуре SSD и имеет 900000 параметров.
