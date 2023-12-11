"""Peak Detection Metric (PDM)"""

from pathlib import Path

import torch
import torchmetrics
from omegaconf import DictConfig
from librosa.onset import onset_detect

from utils.file_utils import (
    resample_dir_if_needed,
    rmdir_and_contents,
    AudioLoudnessNormalize,
)
from utils.dataset import AudioDataset


class PeakDetectionMetric(torchmetrics.Metric):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.normalize = AudioLoudnessNormalize(target_loudness=0.0, sr=sample_rate)

        self.add_state("similarity", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gen_audios: torch.Tensor, gt_audios: torch.Tensor) -> None:
        """Update the similarity and count.

        Args:
            gen_audios (torch.Tensor): Generated audio [B, C, T].
            gt_audios (torch.Tensor): Ground truth audio [B, C, T].

        Returns:
            float: Similarity.
        """
        assert gen_audios.shape == gt_audios.shape, "Audio shapes must match."
        assert (
            gen_audios.shape[1] == 1 and gt_audios.shape[1] == 1
        ), "Audio must be mono."

        # normalize audios
        gen_audios = self.normalize(gen_audios)
        gt_audios = self.normalize(gt_audios)
        # BLAAAH, librosa does not support batching :(
        for gen_audio, gt_audio in zip(gen_audios, gt_audios):
            # picking peaks in an onset strength envelope
            gen_onset_strength = onset_detect(
                y=gen_audio.squeeze(0).numpy(), sr=self.sample_rate, normalize=False
            )
            gt_onset_strength = onset_detect(
                y=gt_audio.squeeze(0).numpy(), sr=self.sample_rate, normalize=False
            )

        # compute similarity
        # self.similarity += torchmetrics.functional.cosine_similarity(
        #     gen_peaks, gt_peaks
        # )
        self.count += torch.tensor(gen_audios.shape[0])

    def compute(self) -> float:
        """Compute the similarity.

        Returns:
            float: Similarity.
        """
        return self.similarity.item() / self.count.item()


def calculate_pdm(cfg: DictConfig):
    """Calculate the Peak Detection Metric (PDM)."""
    gt_audio_dir = Path(cfg.gt_audio_dir)
    gen_audio_dir = Path(cfg.gen_audio_dir)

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

    pdm = PeakDetectionMetric(sample_rate=cfg.sample_rate)

    for batch in loader:
        gen_audio, gt_audio = batch["sample_audio"], batch["gt_audio"]
        pdm(gen_audio, gt_audio)

    if cfg.get("delete_resampled_dirs", True):
        if resampled_gt:
            rmdir_and_contents(gt_audio_dir_resampled)
        if resample_gen:
            rmdir_and_contents(gen_audio_dir_resampled)
