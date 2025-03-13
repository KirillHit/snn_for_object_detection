"""The board that plots data points in animation"""

from matplotlib import pyplot as plt
import collections
import json
import os
from typing import Union


class ProgressBoard:
    """The board that plots data points in animation"""

    def __init__(
        self,
        xlabel: str = None,
        ylabel: str = None,
        ylim: tuple[float, float] = (1.0, 0.1),
        xscale: str = "linear",
        yscale: str = "linear",
        ls: list[str] = ["-", "--", "-.", ":"],
        colors: list[str] = ["C0", "C1", "C2", "C3"],
        figsize: tuple[int, int] = (6, 6),
        display: bool = True,
        every_n: int = 1,
    ):
        self.ls, self.colors, self.display, self.every_n = ls, colors, display, every_n

        self.raw_points = collections.OrderedDict()
        self.data = collections.OrderedDict()
        self.lines = {}
        if not self.display:
            return
        plt.ion()
        subplot = plt.subplots(figsize=figsize)
        self.fig = subplot[0]
        self.axes: plt.Axes = subplot[1]
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        self.axes.set_ylim(top=ylim[0], bottom=ylim[1])

    def draw(self, x: Union[int, float], y: Union[int, float], label: str) -> None:
        """Add a new value to the chart

        :param x: Horizontal coordinate of a point
        :type x: Union[int, float]
        :param y: Vertical coordinate of the point
        :type y: Union[int, float]
        :param label: Name of the line to which to add a point
        :type label: str
        """
        Point = collections.namedtuple("Point", ["x", "y"])

        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        linep = self.data[label]

        points.append(Point(x, y))
        if len(points) != self.every_n:
            return

        mean = lambda x: sum(x) / len(x)
        linep.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()

        if not self.display:
            return

        if label not in self.lines:
            (line,) = self.axes.plot(
                linep[0].x,
                linep[0].y,
                linestyle=self.ls[len(self.lines) % 4],
                color=self.colors[len(self.lines) % 4],
            )
            self.lines[label] = line
            return

        self.lines[label].set_xdata([p.x for p in linep])
        self.lines[label].set_ydata([p.y for p in linep])
        left, right = self.axes.get_xlim()
        self.axes.set_xlim(
            left if linep[-1].x > left else linep[-1].x,
            right if linep[-1].x < right else linep[-1].x,
        )
        bottom, top = self.axes.get_ylim()
        self.axes.set_ylim(
            bottom if linep[-1].y > bottom else linep[-1].y,
            top if linep[-1].y < top else linep[-1].y,
        )
        self.axes.legend(self.lines.values(), self.lines.keys())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_loaded(self, data) -> None:
        mean = lambda x: sum(x) / len(x)
        Point = collections.namedtuple("Point", ["x", "y"])

        for label in data:
            points = []
            linep = []
            src_line = data[label]

            for x, y in src_line:
                points.append(Point(x, y))
                if len(points) != self.every_n:
                    continue
                linep.append(
                    Point(mean([p.x for p in points]), mean([p.y for p in points]))
                )
                points.clear()

            (line,) = self.axes.plot(
                linep[0][0],
                linep[0][1],
                linestyle=self.ls[len(self.lines) % 4],
                color=self.colors[len(self.lines) % 4],
            )
            line.set_xdata([p[0] for p in linep])
            line.set_ydata([p[1] for p in linep])
            self.lines[label] = line
        left, right = self.axes.get_xlim()
        self.axes.set_xlim(
            left if linep[-1][0] > left else linep[-1][0],
            right if linep[-1][0] < right else linep[-1][0],
        )
        bottom, top = self.axes.get_ylim()
        self.axes.set_ylim(
            bottom if linep[-1][1] > bottom else linep[-1][1],
            top if linep[-1][1] < top else linep[-1][1],
        )
        self.axes.legend(self.lines.values(), self.lines.keys())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self, file_name: str, folder: str = "") -> None:
        """Save the chart"""
        path = os.path.join("./", folder)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{file_name}.json"), "w") as f:
            f.write(json.dumps(self.data))
        print("[INFO]: Training logs saved")

    def load_plot(self, file_name: str, folder: str = "") -> None:
        """Loads a chart"""
        file_path = os.path.join("./", folder, f"{file_name}.json")
        if not os.path.exists(file_path):
            print("[ERROR]: The plot file does not exist. Check path: " + file_path)
            return
        with open(file_path, "r") as f:
            data = json.loads(f.read())
        self._draw_loaded(data)
        print("[INFO]: Training logs loaded")
