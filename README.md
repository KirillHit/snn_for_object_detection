# SODa (Spike Object Detector)

Импульсная нейросеть для детекции объектов дорожной обстановки с использованием событийной камеры. 

Применяются открытые библиотеки [PyTorch](https://github.com/pytorch/pytorch) и [norse](https://github.com/norse/norse/tree/main). Поддерживаются датасеты [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) и [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

Документация: [kirillhit.github.io/snn_for_object_detection](https://kirillhit.github.io/snn_for_object_detection/index.html)

## Запуск

Скачайте репозиторий:

``` bash
git clone https://github.com/KirillHit/snn_for_object_detection.git --recurse-submodules
cd snn_for_object_detection
```

Создайте виртуальное окружение с помощью [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

``` bash
conda env create -f environment.yml
conda activate soda_env
```

Перед запуском необходимо скачать один из наборов данных [Gen1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) или [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/), создать в директории проекта папку `data`, и перенести в нее данные в соответствии с шаблоном `/data/<"gen1" или "1mpx">/<"*_bbox.npy" и "*_td.dat">`

Для запуска доступно несколько [сценариев](https://kirillhit.github.io/snn_for_object_detection/pages/structure.html#startup-scripts). Выбор сценария, изменение размера пачки и настройка других параметров обучения осуществляется в [файле конфигурации](https://kirillhit.github.io/snn_for_object_detection/pages/config.html). Для тестов следует выбрать интерактивное обучение.

Далее запустите сценарий:

``` bash
python3 main.py
```

## Предварительные результаты

В данный момент проводятся эксперименты с различными архитектурами и методами обучения. Этот пример относиться к сети версии [0.4.1](https://github.com/KirillHit/snn_for_object_detection/tree/v0.4.1). Сеть основана на архитектуре YOLOv8 и имеет 3M параметров.

<p align="center">
<img src="https://raw.githubusercontent.com/KirillHit/snn_for_object_detection/main/.images/gen1_example.gif">
</p>

Для обучения использовались пачки из 5 примеров длительностью в 32 кадра. Временной шаг между кадрами составляет 16 мс, что примерно соответствует 60 fps. Удалось достигнуть точности 22.8 mAP@0.5. График обучения сети:

<p align="center">
<img src="https://raw.githubusercontent.com/KirillHit/snn_for_object_detection/main/.images/training_graph.png">
</p>

## Генерация моделей

Для ускорения прототипирования была реализована система генерации моделей, которая позволяет быстро проектировать и тестировать разные архитектуры и модули. 

Пример описания простой сверточной сети в коде:

``` python
def vgg_block(out_channels: int, kernel: int = 3):
    return Conv(out_channels, kernel), Norm(), LIF()

cfgs: ListGen = [
    *vgg_block(8), Pool("S"), *vgg_block(32), Pool("S"), *vgg_block(64), Pool("S")
]
```

Подробнее о генераторе моделей смотри [здесь](https://kirillhit.github.io/snn_for_object_detection/pages/generator.html).