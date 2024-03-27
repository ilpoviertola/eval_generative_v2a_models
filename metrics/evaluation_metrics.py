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
from metrics.avclip_score import calculate_avclip_score
from eval_utils.file_utils import (
    to_reencode,
    rmdir_and_contents,
    reencode_dir_if_needed,
    resample_dir_if_needed,
)


class EvaluationMetrics:
    def __init__(self, cfg: EvaluationCfg) -> None:
        self.cfg = cfg
        self.results: tp.Dict[str, tp.Any] = {}
        self.update_last_calculated_ts()

    def run_all(self, force_recalculate: bool = False) -> None:
        pipeline = self.cfg.pipeline
        if pipeline.fad is not None:
            self.run_fad(force_recalculate)
        if pipeline.kld is not None:
            self.run_kld(force_recalculate)
        if pipeline.insync is not None:
            self.run_insync(force_recalculate)
        if pipeline.avclip_score is not None:
            self.run_avclip_score(force_recalculate)

    def run_avclip_score(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.avclip_score is None:
            raise ValueError("No AVCLIP score configuration found in pipeline")
        if "avclip_score" in self.results and not force_recalculate:
            print("AVCLIP score already calculated, skipping...")
            return self.results["avclip_score"]

        self.update_last_calculated_ts()
        score = calculate_avclip_score(
            samples=self.cfg.sample_directory.as_posix(),
            verbose=self.cfg.verbose,
            **asdict(pipeline.avclip_score),
        )
        self.results["avclip_score"] = score
        return score

    def run_insync(
        self, force_recalculate: bool = False
    ) -> tp.Tuple[float, tp.Dict[str, tp.Dict[str, tp.Union[int, float, None]]]]:
        pipeline = self.cfg.pipeline
        if pipeline.insync is None:
            raise ValueError("No InSync configuration found in pipeline")
        if "insync" in self.results and not force_recalculate:
            print("InSync already calculated, skipping...")
            return self.results["insync"]

        self.update_last_calculated_ts()
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
        self.update_last_calculated_ts()
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
            if to_reencode(self.cfg.sample_directory):
                # sample_dir, resampled_samples = reencode_dir_if_needed(
                #     self.cfg.sample_directory,
                #     25,
                #     pipeline.fad.sample_rate,
                #     256,
                #     self.cfg.sample_directory
                #     / f"reencoded_to_{pipeline.fad.sample_rate}hz_25fps",
                # )
                sample_dir = self.cfg.sample_directory
                resampled_samples = False
            else:
                sample_dir, resampled_samples = resample_dir_if_needed(
                    self.cfg.sample_directory,
                    pipeline.fad.sample_rate,
                    self.cfg.sample_directory
                    / f"resampled_to_{pipeline.fad.sample_rate}hz",
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
            gt_dir, resampled_gt = reencode_dir_if_needed(
                self.cfg.gt_directory,
                25,
                pipeline.fad.sample_rate,
                256,
                self.cfg.gt_directory / f"resampled_to_{pipeline.fad.sample_rate}",
            )

        self.update_last_calculated_ts()
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
    ) -> Path:
        if output_path is None:
            assert (
                self.cfg.result_directory is not None
            ), "No directory specified where to export results"
            output_path = self.cfg.result_directory / f"results_{self.ts}.yaml"

        if isinstance(output_path, str):
            output_path = Path(output_path)

        if output_path.exists():
            print(f"Result file {output_path} already exists")
            print(
                "You must recalculate the results or specify a different output file."
            )
            return output_path

        if output_path.suffix != ".yaml":
            print("Warning: Changing output file suffix to .yaml")
            output_path = output_path.with_suffix(".yaml")

        with open(output_path, "w") as f:
            yaml.dump(self.results, f)

        print(f"Results exported to {output_path}")
        return output_path

    def read_results(self, path: tp.Union[str, Path]) -> tp.Dict[str, tp.Any]:
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r") as f:
            results = yaml.safe_load(f)

        self.results = results
        return results

    def print_results(self) -> None:
        print(f"GT directory: {self.cfg.gt_directory}")
        print(f"Sample directory: {self.cfg.sample_directory}")
        print(f"Result directory: {self.cfg.result_directory}")
        print(f"Results:")
        for metric, score in self.results.items():
            print(f"{metric}: {score}")

    def update_last_calculated_ts(self) -> None:
        self.ts = self._get_current_timestamp()

    @staticmethod
    def _get_current_timestamp() -> str:
        return datetime.now().strftime("%y-%m-%dT%H-%M-%S")
