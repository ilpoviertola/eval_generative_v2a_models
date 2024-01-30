from pathlib import Path
from typing import Optional, Tuple
import shutil

import numpy as np
import julius
import torch
from tqdm import tqdm
from audiotools import Meter

from torchaudio import load, save
from torchaudio.transforms import Resample


class AudioLoudnessNormalize(torch.nn.Module):
    GAIN_FACTOR = np.log(10) / 20

    def __init__(self, target_loudness: float = 0.0, sr: int = 24000) -> None:
        super().__init__()
        self.target_loudness = target_loudness
        self.meter = Meter(sr)

    def forward(self, wav: torch.Tensor):
        loudness = self.meter.integrated_loudness(wav.permute(0, 2, 1))
        gain = self.target_loudness - loudness
        gain = torch.exp(gain * self.GAIN_FACTOR)
        return wav * gain[:, None, None]


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


def convert_audio(
    wav: torch.Tensor, from_rate: float, to_rate: float, to_channels: int
) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def resample_file_if_needed(
    file_path: Path, new_sample_rate: int, output_path: Optional[Path] = None
) -> Tuple[Path, bool]:
    """Resample an audio file if the sample rate is not the same as the desired sample rate.

    Args:
        file_path (Path): Path to audio file.
        new_sample_rate (int): Desired sample rate.
        output_path (Optional[Path]): Path to directory containing audio files.

    Returns:
        Tuple[Path, bool]: Path to (resampled) audio file and whether the file was resampled.
    """
    output_path = (
        output_path or file_path.parent / f"{file_path.stem}_{new_sample_rate}.wav"
    )
    assert output_path.suffix == ".wav", "Output file must be a .wav file."
    output_path.parent.mkdir(exist_ok=True, parents=True)

    waveform, sample_rate = load(file_path)
    if sample_rate == new_sample_rate:
        return file_path, False
    resample = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resample(waveform)
    save(output_path, resampled_waveform, new_sample_rate)
    return output_path, True


def resample_dir_if_needed(
    dir_path: Path, new_sample_rate: int, output_path: Optional[Path] = None
) -> Tuple[Path, bool]:
    """Resample a directory of audio files if the sample rate is not the same as the desired sample rate.

    Args:
        dir_path (Path): Path to directory containing audio files.
        new_sample_rate (int): Desired sample rate.
        output_path (Optional[Path]): Path to directory containing audio files.

    Returns:
        Tuple[Path, bool]: Path to (resampled) dir and whether the files were resampled.
    """
    resampled_dir = False
    output_path = output_path or dir_path.parent / f"{dir_path.name}_{new_sample_rate}"
    output_path.mkdir(exist_ok=True, parents=True)

    for file_path in tqdm(
        dir_path.glob("*.wav"),
        desc=f"Resampling directory {dir_path.name} to {new_sample_rate} Hz",
        total=len(list(dir_path.glob("*.wav"))),
    ):
        _, resampled = resample_file_if_needed(
            file_path, new_sample_rate, output_path / file_path.name
        )
        if resampled:
            resampled_dir = True

    if resampled_dir:
        return output_path, True

    rmdir_and_contents(output_path)  # remove empty directory
    return dir_path, False


def rmdir_and_contents(dir_path: Path, verbose: bool = False):
    """Remove a directory and its contents."""
    if verbose:
        print(f"Removing directory {dir_path} and its contents.")
    for file in dir_path.glob("*"):
        file.unlink()
    dir_path.rmdir()


def copy_files(source_dir: Path, destination_dir: Path, file_mask: str = "*.wav"):
    """Copy files from source directory to destination directory."""
    destination_dir.mkdir(exist_ok=True, parents=True)
    for file in source_dir.glob(file_mask):
        shutil.copy(file, destination_dir / file.name)
    return destination_dir
