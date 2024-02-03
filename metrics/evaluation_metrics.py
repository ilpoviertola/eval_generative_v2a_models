import typing as tp
from dataclasses import asdict
import warnings
from pathlib import Path
from datetime import datetime
import yaml

from configs.evaluation_cfg import EvaluationCfg
from metrics.fad import calculate_fad
from metrics.kld import calculate_kld
from metrics.insync import calculate_insync
from eval_utils.file_utils import resample_dir_if_needed, rmdir_and_contents


class EvaluationMetrics:
    def __init__(self, cfg: EvaluationCfg) -> None:
        self.cfg = cfg
        self.results: tp.Dict[str, tp.Any] = {}

    def run_all(self, force_recalculate: bool = False) -> None:
        self.run_fad(force_recalculate)
        self.run_kld(force_recalculate)
        self.run_insync(force_recalculate)

    def run_insync(
        self, force_recalculate: bool = False
    ) -> tp.Tuple[float, tp.Dict[str, float]]:
        pipeline = self.cfg.pipeline
        if pipeline.insync is None:
            raise ValueError("No InSync configuration found in pipeline")
        if "insync" in self.results and not force_recalculate:
            print("InSync already calculated, skipping...")
            return self.results["insync"]

        score, score_per_video = calculate_insync(
            samples=self.cfg.sample_directory.as_posix(),
            verbose=self.cfg.verbose,
            **asdict(pipeline.insync),
        )
        self.results["insync"] = float(score)
        self.results["insync_per_video"] = score_per_video
        return float(score), score_per_video

    def run_kld(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.kld is None:
            raise ValueError("No KLD configuration found in pipeline")
        if "kld" in self.results and not force_recalculate:
            print("KLD already calculated, skipping...")
            return self.results["kld"]

        # note: no need to resample since KLD does it on the fly
        # filter warnings so user does not have to see them (unnecessary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = calculate_kld(
                audio_gts_dir=self.cfg.gt_directory.as_posix(),
                audio_samples_dir=self.cfg.sample_directory.as_posix(),
                verbose=self.cfg.verbose,
                **asdict(pipeline.kld),
            )
        self.results["kld"] = float(score)
        return float(score)

    def run_fad(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.fad is None:
            raise ValueError("No FAD configuration found in pipeline")
        if "fad" in self.results and not force_recalculate:
            print("FAD already calculated, skipping...")
            return self.results["fad"]

        if (
            pipeline.fad.embeddings_fn is not None
            and (self.cfg.sample_directory / pipeline.fad.embeddings_fn).exists()
        ):
            print(
                f"Embeddings found in sample directory ({(self.cfg.sample_directory / pipeline.fad.embeddings_fn).as_posix()})"
            )
            sample_dir = self.cfg.sample_directory
            resampled_samples = False
        else:
            sample_dir, resampled_samples = resample_dir_if_needed(
                self.cfg.sample_directory,
                pipeline.fad.sample_rate,
                self.cfg.sample_directory / f"resampled_to_{pipeline.fad.sample_rate}",
            )
        if (
            pipeline.fad.embeddings_fn is not None
            and (self.cfg.gt_directory / pipeline.fad.embeddings_fn).exists()
        ):
            print(
                f"Embeddings found in gt directory ({(self.cfg.gt_directory / pipeline.fad.embeddings_fn).as_posix()})"
            )
            gt_dir = self.cfg.gt_directory
            resampled_gt = False
        else:
            gt_dir, resampled_gt = resample_dir_if_needed(
                self.cfg.gt_directory,
                pipeline.fad.sample_rate,
                self.cfg.gt_directory / f"resampled_to_{pipeline.fad.sample_rate}",
            )
        score = calculate_fad(
            gts=gt_dir.as_posix(),
            samples=sample_dir.as_posix(),
            sample_embds_path=self.cfg.sample_directory.as_posix(),
            gt_embds_path=self.cfg.gt_directory.as_posix(),
            verbose=self.cfg.verbose,
            **asdict(pipeline.fad),
        )
        if resampled_samples:
            rmdir_and_contents(sample_dir, verbose=self.cfg.verbose)
        if resampled_gt:
            rmdir_and_contents(gt_dir, verbose=self.cfg.verbose)
        self.results["fad"] = float(score)
        return float(score)

    def export_results(
        self, output_path: tp.Optional[tp.Union[str, Path]] = None
    ) -> None:
        if output_path is None:
            assert (
                self.cfg.result_directory is not None
            ), "No directory specified where to export results"
            output_path = (
                self.cfg.result_directory
                / f"results_{self._get_current_timestamp()}.yaml"
            )

        if isinstance(output_path, str):
            output_path = Path(output_path)

        assert not output_path.exists(), f"Result file {output_path} already exists"

        if output_path.suffix != ".yaml":
            print("Warning: Changing output file suffix to .yaml")
            output_path = output_path.with_suffix(".yaml")

        with open(output_path, "w") as f:
            yaml.dump(self.results, f)

        print(f"Results exported to {output_path}")

    def print_results(self) -> None:
        print(f"GT directory: {self.cfg.gt_directory}")
        print(f"Sample directory: {self.cfg.sample_directory}")
        print(f"Result directory: {self.cfg.result_directory}")
        print(f"Results:")
        for metric, score in self.results.items():
            print(f"{metric}: {score}")

    @staticmethod
    def _get_current_timestamp() -> str:
        return datetime.now().strftime("%y-%m-%dT%H-%M-%S")
