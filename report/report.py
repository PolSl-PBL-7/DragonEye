from typing import NamedTuple, List, Dict, Optional
from tensorflow.python.data.ops.dataset_ops import Dataset

from utils.report_utils import get_stochastic_dataset, get_figure_subplot_shape, get_dataset_len, min_max_scores
from report.plots import plotter

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import os


class ReportConfig(NamedTuple):
    name: str
    path: str
    plots: Optional[List[str]] = None
    figure_params: Optional[Dict] = None
    fps: Optional[int] = None


class Report:
    config: ReportConfig

    def __init__(self, config: ReportConfig):
        self.config = config

    def call(self):
        raise NotImplementedError("object of type report interface cannot be called, please implement your own call")


class VideoReport(Report):

    def __init__(self, config: ReportConfig):
        super().__init__(config)

    def __call__(self, dataset: Dataset, predictions: Dataset, scores: Dataset) -> None:
        
        dataset = get_stochastic_dataset(dataset)
        predictions = get_stochastic_dataset(predictions)
        scores = get_stochastic_dataset(scores, force=True)

        data = iter(tf.data.Dataset.zip((dataset, predictions, scores)))

        m, n = get_figure_subplot_shape(len(self.config.plots))
        min_scores, max_scores = min_max_scores(scores)
        plots = {plot_name: None for plot_name in self.config.plots}
        histories = {plot_name: None for plot_name in self.config.plots}
        fig, axes = plt.subplots(m, n, **self.config.figure_params)
        axes = axes.flatten()

        dataset_size = get_dataset_len(dataset)
        self.i = 0

        def animation_init():
            frame, pred, score = next(data)
            for ax, plot_name in zip(axes, self.config.plots):
                plot, history = plotter[plot_name](
                    ax=ax,
                    plot_name=plot_name,
                    frame=frame,
                    pred=pred,
                    score=score,
                    data_size=dataset_size,
                    max_score=max_scores[plot_name] if plot_name in max_scores.keys() else None,
                    min_score=min_scores[plot_name] if plot_name in min_scores.keys() else None,
                )
                plots[plot_name] = plot
                histories[plot_name] = history

            return tuple(plots.values())

        def animation_update(*args, **kwargs):
            frame, pred, score = next(data)
            for plot_name in self.config.plots:
                plots[plot_name], histories[plot_name] = plotter[plot_name](
                    frame=frame,
                    pred=pred,
                    score=score,
                    history=histories[plot_name],
                    plot=plots[plot_name],
                    plot_name=plot_name,
                )
            return tuple(plots.values())

        ani = animation.FuncAnimation(
            fig=fig,
            init_func=animation_init,
            func=animation_update,
            blit=True,
            interval=int(1000 / self.config.fps),
            frames=dataset_size - 1
        )

        path = (self.config.path + f'/{self.config.name}').split('/')
        for i in range(len(path) - 1):
            try:
                os.mkdir('/'.join(path[:i + 1]))
            except FileExistsError:
                pass

        ani.save(self.config.path + f'/{self.config.name}.mp4')

        return
