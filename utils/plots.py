from matplotlib import pyplot as plt
import collections


class ProgressBoard:
    """The board that plots data points in animation."""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        figsize=(6, 6),
        display=True,
    ):
        self.ls, self.colors, self.display = ls, colors, display
        if not self.display:
            return
        plt.ion()
        self.fig, self.axes = plt.subplots(figsize=figsize)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        if ylim is not None:
            self.axes.set_ylim(ylim)
        self.raw_points = collections.OrderedDict()
        self.data = collections.OrderedDict()
        self.lines = {}

    def draw(self, x, y, label, every_n=1):
        if not self.display:
            return

        Point = collections.namedtuple("Point", ["x", "y"])

        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        linep = self.data[label]

        points.append(Point(x, y))
        if len(points) != every_n:
            return

        mean = lambda x: sum(x) / len(x)
        linep.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()

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
