from typing import Optional

from frechet_audio_distance import FrechetAudioDistance


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

    if verbose:
        print("FAD:", score)

    assert score != -1, "FAD calculation failed."
    return score
