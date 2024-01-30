import typing as tp
from dataclasses import asdict, fields
import warnings

from configs.evaluation_cfg import EvaluationCfg
from metrics.fad import calculate_fad
from metrics.kld import calculate_kld
from utils.file_utils import resample_dir_if_needed, rmdir_and_contents


class EvaluationMetrics:
    def __init__(self, cfg: EvaluationCfg) -> None:
        self.cfg = cfg
        self.results: tp.Dict[str, tp.Any] = {}

    def run_all(self) -> None:
        pipeline = self.cfg.pipeline
        scores = {}

        for field in fields(pipeline):
            if field.name.lower() == "fad":
                scores["fad"] = self.run_fad()
            elif field.name.lower() == "kld":
                scores["kld"] = self.run_kld()
            else:
                raise ValueError(f"Unknown metric {field.name.lower()}")

    def run_kld(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if not any(field.name == "kld" for field in fields(pipeline)):
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
        self.results["kld"] = score
        return score

    def run_fad(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if not any(field.name == "fad" for field in fields(pipeline)):
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
        self.results["fad"] = score
        return score
