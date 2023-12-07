from pathlib import Path
from typing import Optional

from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchaudio import load


class AudioDataset(Dataset):
    """PyTorch Dataset for audio files."""

    def __init__(
        self,
        audio_samples_dir: Path,
        audio_gts_dir: Path,
        duration: Optional[float] = 2.0,
    ):
        """Initialize AudioDataset.

        Args:
            audio_samples_dir (Path): Path to audio samples file dir.
            audio_gts_dir (Path): Path to GT audio files dir.
            duration (float, optional): Duration of audio files in seconds. Defaults to 2.0.
        """

        audio_samples = list(audio_samples_dir.glob("*.wav"))
        audio_gts = list(audio_gts_dir.glob("*.wav"))
        assert len(audio_samples) == len(
            audio_gts
        ), "Must have same number of samples and ground truths."

        self.audio_samples = sorted(audio_samples, key=lambda p: p.name)
        self.audio_gts = sorted(audio_gts, key=lambda p: p.name)
        self.duration = duration

    def __len__(self):
        """Return length of dataset."""
        return len(self.audio_gts)

    def __getitem__(self, idx):
        """Return item at index idx."""
        sample = self.audio_samples[idx]
        gt = self.audio_gts[idx]
        assert sample.stem == gt.stem, "Sample and ground truth must have same name."

        sample_audio, sample_audio_sr = load(sample)
        gt_audio, gt_audio_sr = load(gt)

        sample_audio = pad(
            sample_audio,
            (0, int(self.duration * sample_audio_sr) - sample_audio.shape[-1]),
            mode="constant",
            value=0,
        )
        gt_audio = pad(
            gt_audio,
            (0, int(self.duration * gt_audio_sr) - gt_audio.shape[-1]),
            mode="constant",
            value=0,
        )
        if gt_audio.shape[0] == 2:
            gt_audio = gt_audio.mean(dim=0, keepdim=True)
        return {
            "sample_audio": sample_audio,
            "gt_audio": gt_audio,
            "sample_audio_sr": sample_audio_sr,
            "gt_audio_sr": gt_audio_sr,
        }
