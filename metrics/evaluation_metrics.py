import typing as tp
from dataclasses import asdict
import warnings
from pathlib import Path
from datetime import datetime
import yaml
import csv

from configs.evaluation_cfg import EvaluationCfg
from metrics.fad import calculate_fad
from metrics.kld import calculate_kld
from metrics.insync import calculate_insync
from metrics.avclip_score import calculate_avclip_score
from metrics.zcr import calculate_zcr
from metrics.rhythm_similarity import calculate_rhythm_similarity
from metrics.spectral_contrast_similarity import calculate_spectral_contrast_similarity
from metrics.imagebind_score import calculate_imagebind_score
from eval_utils.file_utils import (
    rmdir_and_contents,
    extract_audios_from_video_dir_if_needed,
    reencode_dir_if_needed,
)
from eval_utils.utils import dataclass_from_dict


def load_evaluation_metrics_from_file(path: tp.Union[str, Path]) -> "EvaluationMetrics":
    if isinstance(path, str):
        path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    cfg_dict = data["cfg"]
    cfg_dict["gt_directory"] = Path(cfg_dict["gt_directory"])
    cfg_dict["sample_directory"] = Path(cfg_dict["sample_directory"])
    cfg_dict["result_directory"] = (
        Path(cfg_dict["result_directory"])
        if cfg_dict["result_directory"] is not None
        else None
    )

    cfg = dataclass_from_dict(EvaluationCfg, cfg_dict)
    metrics = EvaluationMetrics(cfg, inited_from_file=True)
    metrics.results = data["results"]
    metrics.ts = data["timestamp"]
    return metrics


class EvaluationMetrics:
    def __init__(self, cfg: EvaluationCfg, inited_from_file: bool = False) -> None:
        self.cfg = cfg
        self.results: tp.Dict[str, tp.Any] = {}
        self.directory_info: tp.Dict[str, tp.Any] = {}
        self.metadata = self._read_metadata()
        if not inited_from_file:
            self.update_last_calculated_ts()
            self.clean_sample_directory()
            self.resolve_directories()

    def _read_metadata(self) -> tp.Union[None, tp.Dict[str, tp.Any]]:
        if self.cfg.metadata is not None:
            with open(self.cfg.metadata, "r") as f:
                reader = csv.reader(f)
                next(reader)
                metadata = {row[0]: row[1] for row in reader}
        else:
            metadata = None
        return metadata

    def clean_sample_directory(self) -> None:
        """Move some files that might be in the sample directory to a subdirectory"""
        for file in self.cfg.sample_directory.iterdir():
            if file.name.endswith("_original.mp4") or file.name.endswith(
                "_original.wav"
            ):
                new_dir = self.cfg.sample_directory / "original_files"
                new_dir.mkdir(exist_ok=True)
                file.rename(new_dir / file.name)

    def resolve_directories(self) -> None:
        """This function resamples the directories if needed according to the pipeline configuration

        Different variations of the data that are needed currently:
            - FAD: ONLY AUDIO (GT & generated), 16 000 Hz (not needed if embeddings are provided)
            - KLD: ONLY AUDIO (GT & generated), resamples on the fly (32 000 Hz) -> not resampled here
            - InSync: Only generated videos, 16 000 Hz, 25 fps, 256x256
            - AVCLIP: Only generated videos, 16 000 Hz, 25 fps, 256x256
        """
        pipeline = self.cfg.pipeline
        if (
            pipeline.insync is not None
            or pipeline.avclip_score is not None
            or pipeline.imagebind_score is not None
        ):
            print("Reencoding videos for InSync, AVCLIP and ImageBind scores...")
            vfps = self.get_config_value_across_pipelines(
                ["insync", "avclip_score", "imagebind_score"], "vfps", 25
            )
            afps = self.get_config_value_across_pipelines(
                ["insync", "avclip_score", "imagebind_score"], "afps"
            )
            input_size = self.get_config_value_across_pipelines(
                ["insync", "avclip_score", "imagebind_score"], "input_size", 224
            )

            # reencode the generated videos if needed
            if (
                self.cfg.sample_directory
                / f"reencoded_to_{vfps}vfps_{afps}afps_{input_size}size"
            ).exists():
                print("resampled directory already exists")
                generated_videos_path = (
                    self.cfg.sample_directory
                    / f"reencoded_to_{vfps}vfps_{afps}afps_{input_size}size"
                )
                resampled = True
            else:
                generated_videos_path, resampled = reencode_dir_if_needed(
                    self.cfg.sample_directory,
                    vfps,
                    afps,
                    input_size,
                    self.cfg.sample_directory
                    / f"reencoded_to_{vfps}vfps_{afps}afps_{input_size}size",
                )
            self.directory_info["generated_videos"] = {
                f"{vfps}vfps_{afps}afps_{input_size}size": (
                    generated_videos_path,
                    resampled,
                )
            }

        if pipeline.fad is not None:
            print("Extracting audio for FAD...")
            self.directory_info["fad"] = {}
            # check does embeddings exist for groundtruth
            if (
                pipeline.fad.embeddings_fn is not None
                and (self.cfg.gt_directory / pipeline.fad.embeddings_fn).exists()
            ):
                self.directory_info["fad"]["gt_embeddings"] = True
                print(
                    f"Embeddings found in gt directory ({(self.cfg.gt_directory / pipeline.fad.embeddings_fn).as_posix()})"
                )
            else:
                self.directory_info["fad"]["gt_embeddings"] = False
                # extract audio from GT videos and sample it to desired frequency
                if (
                    self.cfg.gt_directory / f"audio_{pipeline.fad.sample_rate}"
                ).exists():
                    print("audio directory already exists for GT")
                    gt_audios = (
                        self.cfg.gt_directory / f"audio_{pipeline.fad.sample_rate}"
                    )
                else:
                    gt_audios, _ = extract_audios_from_video_dir_if_needed(
                        self.cfg.gt_directory,
                        pipeline.fad.sample_rate,
                        True,
                        self.cfg.gt_directory / f"audio_{pipeline.fad.sample_rate}",
                    )
                self.directory_info["gt_audios"] = {
                    f"{pipeline.fad.sample_rate}afps": (gt_audios, True)
                }
            # check does embeddings exist for generated samples
            if (
                pipeline.fad.embeddings_fn is not None
                and (self.cfg.sample_directory / pipeline.fad.embeddings_fn).exists()
            ):
                self.directory_info["fad"]["generated_embeddings"] = True
                print(
                    f"Embeddings found in sample directory ({(self.cfg.sample_directory / pipeline.fad.embeddings_fn).as_posix()})"
                )
            else:
                self.directory_info["fad"]["generated_embeddings"] = False
                # extract audio from generated videos and sample it to desired frequency
                if (
                    self.cfg.sample_directory / f"audio_{pipeline.fad.sample_rate}"
                ).exists():
                    print("audio directory already exists for generated samples")
                    sample_audios = (
                        self.cfg.sample_directory / f"audio_{pipeline.fad.sample_rate}"
                    )
                else:
                    sample_audios, _ = extract_audios_from_video_dir_if_needed(
                        self.cfg.sample_directory,
                        pipeline.fad.sample_rate,
                        True,
                        self.cfg.sample_directory / f"audio_{pipeline.fad.sample_rate}",
                    )
                self.directory_info["sample_audios"] = {
                    f"{pipeline.fad.sample_rate}afps": (sample_audios, True)
                }

        if (
            pipeline.zcr is not None
            or pipeline.rhythm_similarity is not None
            or pipeline.spectral_contrast_similarity is not None
        ):
            print(
                "Extracting audio for ZCR, Rhythm Similarity or Spectral Contrast Similarity..."
            )
            # check does gt directory have audio already extracted
            if len(list(self.cfg.gt_directory.glob("*.wav"))) == len(
                list(self.cfg.gt_directory.glob("*.mp4"))
            ):
                print("audios already extracted for GT")
                gt_audios = self.cfg.gt_directory
            else:
                sr = self.get_config_value_across_pipelines(
                    ["zcr", "rhythm_similarity", "spectral_contrast_similarity"], "afps"
                )
                gt_audios, _ = extract_audios_from_video_dir_if_needed(
                    self.cfg.gt_directory,
                    sr,
                    True,
                    self.cfg.gt_directory,
                )
                self.directory_info["gt_audios"] = (
                    {f"{sr}afps": (gt_audios, False)}
                    if "gt_audios" not in self.directory_info
                    else {
                        **self.directory_info["gt_audios"],
                        f"{sr}afps": (gt_audios, False),
                    }
                )

            # copy ground truths of sample audios to a new directory
            sample_audio_gt_dir = gt_audios / "sample_subset"
            if sample_audio_gt_dir.exists():
                rmdir_and_contents(sample_audio_gt_dir, verbose=self.cfg.verbose)
            sample_audio_gt_dir.mkdir(exist_ok=True)
            for sample_audio in self.cfg.sample_directory.iterdir():
                fn = sample_audio.name
                if fn.endswith(".wav"):
                    assert (
                        gt_audios / fn
                    ).exists(), f"GT audio {gt_audios / fn} not found"
                    (sample_audio_gt_dir / fn).resolve().symlink_to(
                        self.cfg.gt_directory / fn
                    )
            self.directory_info["gts_for_sample_audios"] = (sample_audio_gt_dir, True)

    def remove_resampled_directories(self) -> None:
        """Remove the resampled directories if they were created"""
        if "sample_audios" in self.directory_info:
            for _, (sample_dir, resampled) in self.directory_info[
                "sample_audios"
            ].items():
                if resampled:
                    rmdir_and_contents(sample_dir, verbose=self.cfg.verbose)
        if "gt_audios" in self.directory_info:
            for _, (gt_dir, resampled) in self.directory_info["gt_audios"].items():
                if resampled:
                    rmdir_and_contents(gt_dir, verbose=self.cfg.verbose)
        if "generated_videos" in self.directory_info:
            for _, (gen_dir, resampled) in self.directory_info[
                "generated_videos"
            ].items():
                if resampled:
                    rmdir_and_contents(gen_dir, verbose=self.cfg.verbose)
        if "gts_for_sample_audios" in self.directory_info:
            gts_for_sample_audios_dir, resampled = self.directory_info[
                "gts_for_sample_audios"
            ]
            if resampled:
                rmdir_and_contents(gts_for_sample_audios_dir, verbose=self.cfg.verbose)

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
        if pipeline.zcr is not None:
            self.run_zcr(force_recalculate)
        if pipeline.rhythm_similarity is not None:
            self.run_rhythm_similarity(force_recalculate)
        if pipeline.spectral_contrast_similarity is not None:
            self.run_spectral_contrast_similarity(force_recalculate)
        if pipeline.imagebind_score is not None:
            self.run_imagebind_score(force_recalculate)

    def run_imagebind_score(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.imagebind_score is None:
            raise ValueError("No ImageBindScore configuration found in pipeline")
        if "imagebind_score" in self.results and not force_recalculate:
            print("ImageBindScore already calculated, skipping...")
            return self.results["imagebind_score"]

        self.update_last_calculated_ts()
        score = calculate_imagebind_score(
            self.cfg.sample_directory,
            pipeline.imagebind_score.device,
            pipeline.imagebind_score.get_diagonal_scores,
            pipeline.imagebind_score.afps,
            self.cfg.verbose,
        )
        self.results["imagebind_score"] = score
        return score

    def run_spectral_contrast_similarity(
        self, force_recalculate: bool = False
    ) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.spectral_contrast_similarity is None:
            raise ValueError(
                "No SpectralContrastSimilarity configuration found in pipeline"
            )
        if "spectral_contrast_similarity" in self.results and not force_recalculate:
            print("Spectral contrast similarity already calculated, skipping...")
            return self.results["spectral_contrast_similarity"]

        self.update_last_calculated_ts()
        score = calculate_spectral_contrast_similarity(
            self.cfg.sample_directory,
            self.directory_info["gts_for_sample_audios"][0],
            pipeline.rhythm_similarity.afps,
            self.cfg.verbose,
            self.metadata,
            pipeline.spectral_contrast_similarity.duration,
        )
        self.results["spectral_contrast_similarity"] = score
        return score

    def run_rhythm_similarity(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.rhythm_similarity is None:
            raise ValueError("No RhythmSimilarity configuration found in pipeline")
        if "rhythm_similarity" in self.results and not force_recalculate:
            print("Rhythm similarity already calculated, skipping...")
            return self.results["rhythm_similarity"]

        self.update_last_calculated_ts()
        score = calculate_rhythm_similarity(
            self.cfg.sample_directory,
            self.directory_info["gts_for_sample_audios"][0],
            pipeline.rhythm_similarity.afps,
            self.cfg.verbose,
            self.metadata,
            pipeline.rhythm_similarity.duration,
        )
        self.results["rhythm_similarity"] = score
        return score

    def run_zcr(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.zcr is None:
            raise ValueError("No ZCR configuration found in pipeline")
        if "zcr" in self.results and not force_recalculate:
            print("ZCR already calculated, skipping...")
            return self.results["zcr"]

        self.update_last_calculated_ts()
        score = calculate_zcr(
            self.cfg.sample_directory,
            self.directory_info["gts_for_sample_audios"][0],
            pipeline.zcr.afps,
            self.cfg.verbose,
            self.metadata,
            pipeline.zcr.duration,
        )
        self.results["zcr"] = score
        return score

    def run_avclip_score(self, force_recalculate: bool = False) -> float:
        pipeline = self.cfg.pipeline
        if pipeline.avclip_score is None:
            raise ValueError("No AVCLIP score configuration found in pipeline")
        if "avclip_score" in self.results and not force_recalculate:
            print("AVCLIP score already calculated, skipping...")
            return self.results["avclip_score"]

        self.update_last_calculated_ts()
        vfps = pipeline.avclip_score.vfps
        afps = pipeline.avclip_score.afps
        input_size = pipeline.avclip_score.input_size
        score = calculate_avclip_score(
            samples=self.directory_info["generated_videos"][
                f"{vfps}vfps_{afps}afps_{input_size}size"
            ][0].as_posix(),
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
        vfps = pipeline.insync.vfps
        afps = pipeline.insync.afps
        input_size = pipeline.insync.input_size
        score, score_per_video = calculate_insync(
            samples=self.directory_info["generated_videos"][
                f"{vfps}vfps_{afps}afps_{input_size}size"
            ][0].as_posix(),
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
                metadata=self.metadata,
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

        if self.directory_info["fad"]["gt_embeddings"]:
            gt_dir = None
            resampled_gt = False
        else:
            gt_dir, resampled_gt = self.directory_info["gt_audios"][
                f"{pipeline.fad.sample_rate}afps"
            ]
            gt_dir = gt_dir.as_posix()

        if self.directory_info["fad"]["generated_embeddings"]:
            sample_dir = None
            resampled_samples = False
        else:
            sample_dir, resampled_samples = self.directory_info["sample_audios"][
                f"{pipeline.fad.sample_rate}afps"
            ]
            sample_dir = sample_dir.as_posix()

        self.update_last_calculated_ts()
        score = calculate_fad(
            gt_audios=gt_dir,
            sample_audios=sample_dir,
            sample_embds_path=self.cfg.sample_directory.as_posix(),
            gt_embds_path=self.cfg.gt_directory.as_posix(),
            verbose=self.cfg.verbose,
            **asdict(pipeline.fad),
        )
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

        cfg = asdict(self.cfg)
        cfg["gt_directory"] = cfg["gt_directory"].as_posix()
        cfg["sample_directory"] = cfg["sample_directory"].as_posix()
        cfg["result_directory"] = (
            cfg["result_directory"].as_posix()
            if cfg["result_directory"] is not None
            else None
        )
        cfg["metadata"] = (
            cfg["metadata"].resolve().as_posix()
            if cfg["metadata"] is not None
            else None
        )

        exported_data = {
            "id": self.cfg.id,
            "timestamp": self.ts,
            "cfg": cfg,
            "results": self.results,
        }
        with open(output_path, "w") as f:
            yaml.dump(exported_data, f)

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
        print(f"ID: {self.cfg.id}")
        print(f"GT directory: {self.cfg.gt_directory}")
        print(f"Sample directory: {self.cfg.sample_directory}")
        print(f"Result directory: {self.cfg.result_directory}")
        print(f"Results:")
        for metric, score in self.results.items():
            print(f"{metric}: {score}")

    def update_last_calculated_ts(self) -> None:
        self.ts = self._get_current_timestamp()

    def get_config_value_across_pipelines(
        self, pipeline_keys: tp.List[str], key: str, default_value: tp.Any = None
    ) -> tp.Any:
        """Some configurations share same key/value pairs across different pipelines.
        This function helps to get the value of a key that is shared across multiple pipelines.

        In case the key is not found in any of the pipelines, a ValueError is raised or
        the default_value is returned.

        NOTE: The first pipeline in the list has the highest priority. It is assumed that the
        pipelines share the same key/value pairs.

        Args:
            pipeline_keys (tp.List[str]): List of pipeline names where to look for the key
            key (str): Key to look for
            default_value (tp.Any, optional): Default value to return if key is not found. Defaults to None.

        Raises:
            ValueError: Key not found

        Returns:
            tp.Any: Value of the key (or default value if not found)
        """
        for pipeline_key in pipeline_keys:
            if hasattr(self.cfg.pipeline, pipeline_key):
                pipeline = getattr(self.cfg.pipeline, pipeline_key)
                if hasattr(pipeline, key):
                    return getattr(pipeline, key)
        if default_value is not None:
            return default_value
        raise ValueError(f"Key {key} not found in any of the pipelines")

    @staticmethod
    def _get_current_timestamp() -> str:
        return datetime.now().strftime("%y-%m-%dT%H-%M-%S")
