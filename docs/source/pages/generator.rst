Генерация моделей
=================

Для упрощения прототипирования в проекте реализован генератор моделей для рекуррентных сетей. Инструменты для генерации реализованы в модуле :any:`models.generator`. Базовым классов для сгенерированных моделей является :class:`ModelGen <models.generator.ModelGen>`, в котором реализуется общий интерфейс и инструменты генерации. От него наследуются 3 модели, из которых класс :class:`SODa <models.soda.SODa>` собирает детектор:

====================================================    ===============
:class:`BackboneGen <models.generator.BackboneGen>`     Базовая часть сети. Выделяет признаки из изображения.
:class:`NeckGen <models.generator.NeckGen>`             Средняя часть сети. Генерирует карты признаков. Результатом работы сети являются 
                                                        несколько матриц разной формы.
:class:`HeadGen <models.generator.HeadGen>`             Последняя часть сети. Генерирует итоговые предсказания. Используется вместе 
                                                        с классом :class:`Head <models.generator.Head>`, который создаёт отдельный экземпляр модели головы для каждой карты признаков.
====================================================    ===============

Для генерации моделей используется объект :class:`BlockGen <models.generator.BlockGen>`, который создаёт модель из списка объектов :class:`LayerGen <models.module.generators.LayerGen>` для контурирования слоёв. Если в конфигурации имеется вложенный список, то :class:`BlockGen <models.generator.BlockGen>` вызывается рекурсивно. Также можно задать сложное поведение при обработке вложенных листов с помощью перегрузок для списка :external:class:`list`: :class:`Residual <models.module.generators.Residual>` и :class:`Dense <models.module.generators.Dense>`. Это позволяет строить остаточные и плотные нейросети.

.. code-block::
    :caption: Пример конфигурации свёрточной сети с остаточными связями

    def conv(out_channels: int, kernel: int = 3, stride: int = 1):
        return (
            Conv(out_channels, stride=stride, kernel_size=kernel),
            Norm(),
            LIF(),
        )

    def res_block(out_channels: int, kernel: int = 3):
        return (
            Conv(out_channels, 1),
            # Residual block. The values from all branches are added together
            Residual(
                [
                    [*conv(out_channels, kernel)],
                    [Conv(out_channels, 1)],
                ]
            ),
            Conv(out_channels, 1),
        )

    cfgs: ListGen = [
        *conv(64, 7, 2), *res_block(64, 5), *conv(128, 5, 2), *res_block(128)
    ]

.. note::
    `Пирамида признаков <https://arxiv.org/abs/1612.03144>`_ может быть представлена в списке конфигурации в виде остаточных или плотных сетей.

Собственная конфигурация
------------------------

Для добавления своей конфигурации необходимо создать класс наследник :class:`BaseConfig <models.generator.BaseConfig>`, в котором будут реализованы функции возвращающие конфигурации моделей для Backbone, Neck и Head. Пример конфигурации сети подобной YOLOv8 можно найти в :class:`Yolo <models.yolo.Yolo>`.

После создания собственной конфигурации её нужно добавить в словарь ``config_list`` в файле ``models/__init_.py``:

.. literalinclude:: ../../../models/__init__.py
    :language: python

После для выбора созданной модели достаточно указать её название из config_list в файле конфигурации (см. :doc:`config`).

Кастомные слои
--------------

Список конфигурации сети состоит из генераторов слоёв, которые наследуются от :class:`LayerGen <models.module.generators.LayerGen>`. Их задача сконструировать и вернуть новый :external:class:`модуль <torch.nn.Module>` для включения в сеть. Все поддерживаемые слои находятся в файле :any:`models.module.generators`. Чтобы добавить свой слой достаточно создать нового наследника :class:`LayerGen <models.module.generators.LayerGen>`, который будет конструировать требуемый модуль и рассчитывать количество каналов матрицы после его применения. Для примера смотри :class:`Conv <models.module.generators.Conv>`.