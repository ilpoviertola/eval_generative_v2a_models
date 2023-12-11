# Cross correlation between original and generated signals
from typing import Dict
from pathlib import Path
import json

import numpy as np
import torch
from scipy import signal
from omegaconf import DictConfig
from tqdm import tqdm

from utils.file_utils import (
    resample_dir_if_needed,
    rmdir_and_contents,
    AudioLoudnessNormalize,
)
from utils.dataset import AudioDataset


def xcorr(x, y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr


def calculate_xcorr(cfg: DictConfig) -> Dict[str, float]:
    """Calculate the Cross-Correlation Metric (XCorr)."""
    gt_audio_dir = Path(cfg.gts)
    gen_audio_dir = Path(cfg.samples)

    gt_audio_dir_resampled, resampled_gt = resample_dir_if_needed(
        gt_audio_dir, cfg.sample_rate
    )
    gen_audio_dir_resampled, resample_gen = resample_dir_if_needed(
        gen_audio_dir, cfg.sample_rate
    )

    dataset = AudioDataset(
        audio_samples_dir=gen_audio_dir_resampled,
        audio_gts_dir=gt_audio_dir_resampled,
        duration=cfg.duration,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )
    normalize = AudioLoudnessNormalize(-24.0)

    lags = 0
    count = 0
    for batch in tqdm(loader):
        gen_audios, gt_audios = batch["sample_audio"], batch["gt_audio"]
        assert (
            gen_audios.shape[1] == 1 and gt_audios.shape[1] == 1
        ), "Audio must be mono."

        # normalize audios
        gen_audios = normalize(gen_audios)
        gt_audios = normalize(gt_audios)

        # TODO: support batching
        for gen_audio, gt_audio in zip(gen_audios, gt_audios):
            # compute similarity
            lag, corr = xcorr(gen_audio.squeeze(0).numpy(), gt_audio.squeeze(0).numpy())
            lags += np.abs(np.abs(corr).argmax() - len(lag) // 2) / cfg.sample_rate
            count += 1

    if cfg.get("delete_resampled_dirs", True):
        if resampled_gt:
            rmdir_and_contents(gt_audio_dir_resampled)
        if resample_gen:
            rmdir_and_contents(gen_audio_dir_resampled)

    score = lags / count

    if cfg.get("verbose", False):
        print("XCORR:", score)

    if cfg.get("save", False):
        with open(Path(cfg.samples) / "xcorr.json", "w") as f:
            json.dump({"XCORR": score}, f, indent=4)

    return {"XCORR": score}
