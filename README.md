# Spiking YOLO

Спайковая нейросеть для детекции объектов. Основана на библиотеке с открытым исходным кодом [norse](https://github.com/norse/norse/tree/main).

## Настройка окружения

Настройка окружения с помощью conda:

``` bash
conda create --name <name> --file requirements.txt -c pytorch -c nvidia -c conda-forge
```

На данный момент зависимости в conda для norse сломаны, поэтому его необходимо установить через pip:

``` bash
conda activate <name>
pip install norse
```

## Datasets

### Banana Detection

Содержит размеченные изображения бананов, наложенные на случайные картинки. Представлен в книге [d2l](https://d2l.ai/chapter_computer-vision/object-detection-dataset.html), можно скачать по [ссылке](http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip).
