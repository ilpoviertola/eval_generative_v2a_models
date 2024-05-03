from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa
from tqdm import tqdm


def calculate_spectral_contrast_similarity(
    samples_dir: Path,
    gt_dir: Path,
    sample_rate: int,
    verbose: bool = False,
    start_secs: Optional[Dict[str, float]] = None,
    duration: float = 2.56,
) -> float:
    sample_files = sorted(list(samples_dir.glob("*.wav")))
    gt_files = sorted(list(gt_dir.glob("*.wav")))
    if start_secs is None:
        start_secs = {}
    total_spectral_contrast_similarity = 0.0
    count = 0

    for sample_file, gt_file in tqdm(
        zip(sample_files, gt_files),
        desc="Calculating Spectral Contrast Similarity",
        total=len(sample_files),
    ):
        assert (
            sample_file.name == gt_file.name
        ), f"Sample and GT files do not match: {sample_file.name} != {gt_file.name}"
        sample, _ = librosa.load(sample_file, sr=sample_rate)
        gt, _ = librosa.load(
            gt_file,
            sr=sample_rate,
            offset=float(start_secs.get(gt_file.stem, 0)),
            duration=duration,
        )
        gt_spectral_contrast = librosa.feature.spectral_contrast(y=gt, sr=sample_rate)
        sample_spectral_contrast = librosa.feature.spectral_contrast(
            y=sample, sr=sample_rate
        )
        min_columns = min(
            gt_spectral_contrast.shape[1], sample_spectral_contrast.shape[1]
        )
        sample_spectral_contrast = sample_spectral_contrast[:, :min_columns]
        gt_spectral_contrast = gt_spectral_contrast[:, :min_columns]
        spectral_contrast_similarity = np.mean(
            np.abs(sample_spectral_contrast - gt_spectral_contrast)
        )
        normalized_spectral_contrast_similarity = spectral_contrast_similarity / np.max(
            [np.abs(gt_spectral_contrast), np.abs(sample_spectral_contrast)]
        )
        total_spectral_contrast_similarity += normalized_spectral_contrast_similarity
        count += 1

    if verbose:
        print(
            f"Spectral contrast similarity: {total_spectral_contrast_similarity / count}"
        )

    return float(total_spectral_contrast_similarity / count)
