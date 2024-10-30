# Spiking YOLO

Импульсная нейросеть для детекции объектов дорожной обстановки с использованием событийной камеры. Применяются открытые библиотеки [PyTorch](https://github.com/pytorch/pytorch) и [norse](https://github.com/norse/norse/tree/main).

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

## Использованные датасеты

### [GEN1 Automotive Detection Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

Набор данных записывался с помощью датчика PROPHESEE GEN1 с разрешением 304×240 пикселей, установленного на приборной панели автомобиля. Метки были получены с использованием камеры ATIS путем маркировки вручную.

Он содержит 39 часов езды по открытой дороге и различные сценарии вождения: в городе, на шоссе, в пригороде и сельской местности.

Доступны ограничивающие рамки для двух классов: пешеходов и автомобилей. (Грузовые автомобили и автобусы не имеют маркировки).

Планируется поддержка датасетов [1Mpx](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) и [DSEC](https://dsec.ifi.uzh.ch/)

## Промежуточные результаты

Пример работы сети на датасете Gen1.

![gen1_example](.images/gen1_example.gif)
