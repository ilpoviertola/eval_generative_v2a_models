from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa
from tqdm import tqdm


def calculate_zcr(
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
    total_zcr_similarity = 0.0
    count = 0

    for sample_file, gt_file in tqdm(
        zip(sample_files, gt_files), desc="Calculating ZCR", total=len(sample_files)
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
        gt_zcr = np.mean(np.abs(np.diff(np.sign(gt))) > 0)
        sample_zcr = np.mean(np.abs(np.diff(np.sign(sample))) > 0)
        zcr_similarity = 1 - np.abs(gt_zcr - sample_zcr)
        total_zcr_similarity += zcr_similarity
        count += 1

    if verbose:
        print(f"ZCR similarity: {total_zcr_similarity / count}")

    return total_zcr_similarity / count
