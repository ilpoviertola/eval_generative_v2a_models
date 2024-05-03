from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa
from tqdm import tqdm


def calculate_rhythm_similarity(
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
    total_rhythm_similarity = 0.0
    count = 0

    for sample_file, gt_file in tqdm(
        zip(sample_files, gt_files),
        desc="Calculating Rhythm Similarity",
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

        min_length = len(gt)
        gt_onset_vector = np.zeros(min_length)
        gt_onsets = librosa.onset.onset_detect(y=gt, sr=sample_rate, units="time")
        gt_onsets = np.array(gt_onsets) * sample_rate
        gt_onsets = gt_onsets[gt_onsets < min_length]
        gt_onset_vector[gt_onsets.astype(int)] = 1

        min_length = min(min_length, len(sample))
        sample_onset_vector = np.zeros(min_length)
        sample_onsets = librosa.onset.onset_detect(
            y=sample[:min_length], sr=sample_rate, units="time"
        )
        sample_onsets = np.array(sample_onsets) * sample_rate
        sample_onsets = sample_onsets[sample_onsets < min_length]
        sample_onset_vector[sample_onsets.astype(int)] = 1

        rhythm_similarity = (
            np.corrcoef(
                gt_onset_vector[:min_length],
                sample_onset_vector[:min_length],
            )[0, 1]
            + 1
        ) / 2
        total_rhythm_similarity += (
            rhythm_similarity if not np.isnan(rhythm_similarity) else 0
        )
        count += 1

    if verbose:
        print(f"Rhythm similarity: {total_rhythm_similarity / count}")

    return float(total_rhythm_similarity / count)
