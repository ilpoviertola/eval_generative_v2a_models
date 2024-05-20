from typing import Optional
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance


def create_fad_audio_dir(fad_audio_dir: str) -> str:
    audios_path = Path(fad_audio_dir) / "fad_audios"
    audios_path.mkdir(exist_ok=False, parents=False)
    # create symbolic links to the files
    for audio in Path(fad_audio_dir).glob("*.wav"):
        (audios_path / audio.name).symlink_to(audio)
    return audios_path.as_posix()


def remove_fad_audio_dir(fad_audio_dir: str) -> None:
    # remove contents
    for audio in Path(fad_audio_dir).glob("*.wav"):
        audio.unlink()
    # remove empty dir
    Path(fad_audio_dir).rmdir()


def calculate_fad(
    sample_embds_path: str,
    gt_embds_path: str,
    embeddings_fn: str,
    gt_audios: Optional[str] = None,
    sample_audios: Optional[str] = None,
    model_name: str = "vggish",
    sample_rate: int = 16000,
    use_pca: bool = False,
    use_activation: bool = False,
    verbose: bool = False,
    dtype: str = "float32",
) -> float:
    """Calculate the Frechet Audio Distance."""
    # Copy audio files (if provided) to own directories since FAD just reads the whole dir
    if gt_audios is not None:
        gt_audios = create_fad_audio_dir(gt_audios)
    if sample_audios is not None:
        sample_audios = create_fad_audio_dir(sample_audios)

    fad = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
    )

    score = fad.score(
        background_dir=sample_audios,
        eval_dir=gt_audios,
        background_embds_path=sample_embds_path + "/" + embeddings_fn,
        eval_embds_path=gt_embds_path + "/" + embeddings_fn,
        dtype=dtype,
    )

    if gt_audios is not None:
        remove_fad_audio_dir(gt_audios)
    if sample_audios is not None:
        remove_fad_audio_dir(sample_audios)

    if verbose:
        print("FAD:", score)

    assert score != -1, "FAD calculation failed."
    return score
