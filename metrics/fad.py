from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance

from eval_utils.file_utils import (
    copy_files,
    rmdir_and_contents,
    extract_audios_from_video_dir_if_needed,
)


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
    gt_audios, audio_dir = None, None
    if not Path(f"{gt_embds_path}/{embeddings_fn}").exists():
        gts, _ = extract_audios_from_video_dir_if_needed(
            Path(gts), samplerate=sample_rate
        )
        gt_audios = copy_files(Path(gts), Path(gts) / "audio", file_mask="*.wav")
    if not Path(f"{sample_embds_path}/{embeddings_fn}").exists():
        samples, _ = extract_audios_from_video_dir_if_needed(
            Path(samples), samplerate=sample_rate
        )
        audio_dir = copy_files(
            Path(samples), Path(samples) / "audio", file_mask="*.wav"
        )

    score = fad.score(
        background_dir=audio_dir.as_posix() if audio_dir else None,
        eval_dir=gt_audios.as_posix() if gt_audios else None,
        background_embds_path=sample_embds_path + "/" + embeddings_fn,
        eval_embds_path=gt_embds_path + "/" + embeddings_fn,
        dtype=dtype,
    )

    if verbose:
        print("FAD:", score)

    if audio_dir is not None:
        rmdir_and_contents(audio_dir)
    if gt_audios is not None:
        rmdir_and_contents(gt_audios)

    assert score != -1, "FAD calculation failed."
    return score
