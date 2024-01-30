from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance

from utils.file_utils import copy_files, rmdir_and_contents


def calculate_fad(
    gts: str,
    samples: str,
    sample_embds_path: str,
    gt_embds_path: str,
    embeddings_fn: str,
    model_name: str = "vggish",
    sample_rate: int = 16000,
    use_pca: bool = False,
    use_activation: bool = False,
    verbose: bool = False,
    dtype: str = "float32",
) -> float:
    """Calculate the Frechet Audio Distance."""
    fad = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
    )

    # since FrechetAudioDistance just tries to load every file in the directory,
    # we need to make sure that the directories only contain audio files
    audio_dir = copy_files(Path(samples), Path(samples) / "audio", file_mask="*.wav")

    score = fad.score(
        background_dir=audio_dir,
        eval_dir=gts,
        background_embds_path=sample_embds_path + "/" + embeddings_fn,
        eval_embds_path=gt_embds_path + "/" + embeddings_fn,
        dtype=dtype,
    )

    if verbose:
        print("FAD:", score)

    rmdir_and_contents(audio_dir)

    return score
