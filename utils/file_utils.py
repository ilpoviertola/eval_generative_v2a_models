from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm
from torchaudio import load, save
from torchaudio.transforms import Resample


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

    for file_path in tqdm(dir_path.glob("*.wav")):
        _, resampled = resample_file_if_needed(
            file_path, new_sample_rate, output_path / file_path.name
        )
        if resampled:
            resampled_dir = True

    if resampled_dir:
        return output_path, True

    return dir_path, False


def rmdir_and_contents(dir_path: Path):
    """Remove a directory and its contents."""
    for file in dir_path.glob("*"):
        file.unlink()
    dir_path.rmdir()
