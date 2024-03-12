""""Combine multiple evaluation metrics results."""

import typing as tp
from pathlib import Path

import numpy as np

from metrics.evaluation_metrics import EvaluationMetrics


class EvaluationMetricsCombiner:
    def __init__(self, metrics: tp.List[EvaluationMetrics]):
        self.metrics = metrics
        self.all_results: tp.Dict[str, tp.Tuple[list, list]] = {}

    def combine(self):
        for metric in self.metrics:
            assert isinstance(metric, EvaluationMetrics)
            assert isinstance(metric.results, dict)
            assert metric.results is not None, "Results are empty"
            for metric_type in metric.results:
                if metric_type not in self.all_results:
                    self.all_results[metric_type] = ([], [])

                # special metrics
                if metric_type == "insync_per_video":
                    for video in metric.results[metric_type]:
                        self.all_results[metric_type][0].append(video)
                        self.all_results[metric_type][1].append(
                            int(metric.results[metric_type][video]["insync"])
                        )
                    continue

                # general metrics
                self.all_results[metric_type][0].append(
                    Path(metric.cfg.sample_directory).name
                )
                self.all_results[metric_type][1].append(metric.results[metric_type])
        return self.all_results

    def plot(
        self, plot_dir: tp.Union[str, Path, None]
    ) -> tp.Union[Path, tp.Dict[str, np.ndarray]]:
        """Plot metrics. Return path to the plot or the plot itself.

        Args:
            plot_dir (tp.Union[str, Path, None]): Path to plot directory. If None, return the plot itself.

        Returns:
            tp.Union[Path, tp.Dict[str, np.ndarray]]: Path to the plot or the plot itself.
        """
        assert (
            self.all_results is not None
        ), "All results are empty. Did you run combine()?"
        if type(plot_dir) == str:
            plot_dir = Path(plot_dir)
        save_plots_to_png = plot_dir is not None

        if save_plots_to_png:
            plot_dir.mkdir(exist_ok=True, parents=True)

        plots = {}
        for metric_type in self.all_results:
            if save_plots_to_png:
                plot_path = plot_dir / f"{metric_type}.png"
            else:
                plot_path = None
            plots[metric_type] = self.plot_metric(metric_type, plot_path=plot_path)
        return plots

    def plot_metric(
        self, metric_type: str, plot_path: tp.Union[Path, None] = None
    ) -> tp.Union[np.ndarray, Path]:
        """Plot a metric. Return the plot.

        Args:
            metric_type (str): Metric type.
            plot_path (tp.Union[str, Path, None], optional): Path to plot. If None, return the plot itself.

        Returns:
            tp.Union[np.ndarray, Path]: Plot or path to it.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots())
        ax.plot(self.all_results[metric_type][0], self.all_results[metric_type][1])
        ax.set_xlabel("Sample directory")
        ax.set_ylabel(metric_type)
        plt.xticks(rotation=45)
        if plot_path is not None:
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        else:
            return fig
