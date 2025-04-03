from lightning.pytorch.cli import LightningCLI
import lightning as L
from models import *
from utils import PropheseeDataModule
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from models.generator import StateStorage
from typing import List, Any, Optional, Dict, NamedTuple


class Statics:
    def __init__(self):
        self.layer_idx = 0

    def process(self, model: L.LightningModule):
        for m in model.children():
            if isinstance(m, StateStorage):
                state = self._unbatch_state(m.get_state())
                spikes = m.get_spikes()[:, 0].cpu()
                self._analyse_data(m, state, spikes)

    def _unbatch_state(self, states: NamedTuple) -> NamedTuple:
        state_dict = states._asdict()
        cls = states.__class__
        keys = list(state_dict.keys())
        output_dict = {}
        for key in keys:
            output_dict[key] = getattr(states, key)[:, 0].to(cpu())
        return cls(**output_dict)

    def _analyse_data(self, layer: StateStorage, state: NamedTuple, spikes: torch.Tensor):
        self.layer_idx += 1
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.suptitle(
            f"Statistics of layer {self.layer_idx} consisting of {type(layer.module).__name__} neurons"
        )
        self.axes: Dict[str, plt.Axes] = self.fig.subplot_mosaic(
            "AB;DC", per_subplot_kw={"A": {"projection": "3d"}}
        )
        keys = list(state._asdict().keys())
        self._plot_mean_states(
            state,
            *keys,
            title="Average values of potential and current",
        )
        self._plot_spikes(spikes)
        self._plot_density(spikes)
        self._plot_neurons_states(
            state,
            *keys,
        )

        plt.show()

    def _plot_mean_states(
        self,
        states: List[Any],
        *variables: str,
        title: Optional[str] = None,
        **kwargs,
    ):
        assert len(variables) > 0, "0 variables were given to render, we require at least 1"
        ax = self.axes["B"]
        ax.clear()
        for variable in variables:
            values: torch.Tensor = getattr(states, variable)
            values = values.flatten(1).mean(1)
            ax.plot(values, label=variable, **kwargs)
        ax.legend()
        ax.set_xlabel("time, ms")
        if title:
            ax.set_title(title)

    def _plot_spikes(self, spikes: torch.Tensor, slice=4):
        ax = self.axes["A"]

        t, c, x, y = spikes.shape

        def update(frame):
            ts = spikes[frame, ::slice, ::slice, ::slice]
            indexes = ts.nonzero(as_tuple=True)
            ax.clear()
            ax.scatter(indexes[1], indexes[2], indexes[0])
            ax.set_xbound(0, x // slice)
            ax.set_ybound(0, y // slice)
            ax.set_zbound(0, c // slice)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Cannel")
            ax.set_title(f"Visualization of spike activity for time {frame}")
            self.fig.canvas.draw()

        self.anim = animation.FuncAnimation(fig=self.fig, func=update, frames=t, interval=10)

    def _plot_density(self, spikes: torch.Tensor):
        ax = self.axes["C"]
        ax.set_title("The ratio of impulses at each step to the total number of neurons in layer")
        ax.set_xlabel("time, ms")

        _, c, x, y = spikes.shape
        volume = c * x * y
        value = []
        for ts in spikes:
            value.append(torch.count_nonzero(ts) / volume)
        ax.plot(value)

    def _plot_neurons_states(
        self,
        states: List[Any],
        *variables: str,
        title: Optional[str] = None,
        **kwargs,
    ):
        assert len(variables) > 0, "0 variables were given to render, we require at least 1"
        ax = self.axes["D"]
        ax_list = [ax] + [ax.twinx() for _ in range(len(variables) - 1)]
        ax.set_xlabel("time, ms")

        t, c, x, y = states[0].shape
        index = [torch.randint(c, (1,)), torch.randint(x, (1,)), torch.randint(y, (1,))]
        ax.set_title(f"State of a random neuron with index {index}")

        for variable, loc_ax, color in zip(
            variables,
            ax_list,
            list(mcolors.TABLEAU_COLORS),
        ):
            values: torch.Tensor = getattr(states, variable)
            values = values[:, *index]
            loc_ax.plot(values, label=variable, color=color, **kwargs)
            loc_ax.set_ylabel(variable, color=color)
        if title:
            ax.set_title(title)
        self.fig.tight_layout()


if __name__ == "__main__":
    cli = LightningCLI(
        SODa,
        PropheseeDataModule,
        subclass_mode_model=True,
        parser_kwargs={
            "default_config_files": [
                "config/config.yaml",
            ]
        },
        run=False,
    )
    model = cli.model
    trainer = cli.trainer
    stat = Statics()

    trainer.predict(model)
    stat.process(model)
