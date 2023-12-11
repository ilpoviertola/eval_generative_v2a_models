from pathlib import Path
import json
from typing import Dict

from omegaconf import DictConfig
from frechet_audio_distance import FrechetAudioDistance

from utils.file_utils import resample_dir_if_needed, rmdir_and_contents


def calculate_fad(cfg: DictConfig) -> Dict[str, float]:
    """Calculate the Frechet Audio Distance."""
    fad = FrechetAudioDistance(
        model_name=cfg.model_name,
        sample_rate=cfg.sample_rate,
        use_pca=cfg.use_pca,
        use_activation=cfg.use_activation,
        verbose=cfg.verbose,
    )

    resampled_background_dir, bg_was_resampled = resample_dir_if_needed(
        Path(cfg.samples), cfg.sample_rate
    )
    resampled_eval_dir, eval_was_resampled = resample_dir_if_needed(
        Path(cfg.gts), cfg.sample_rate
    )

    score = fad.score(
        background_dir=resampled_background_dir.as_posix(),
        eval_dir=resampled_eval_dir.as_posix(),
        background_embds_path=cfg.background_embds_path,
        eval_embds_path=cfg.eval_embds_path,
        dtype=cfg.dtype,
    )

    if cfg.get("delete_resampled_dirs", True):
        if bg_was_resampled:
            rmdir_and_contents(resampled_background_dir)
        if eval_was_resampled:
            rmdir_and_contents(resampled_eval_dir)

    if cfg.get("save", True):
        with open(Path(cfg.samples) / "fad.json", "w") as f:
            json.dump({"FAD": score}, f, indent=4)

    if cfg.get("verbose", False):
        print("FAD:", score)

    return {"FAD": score}
