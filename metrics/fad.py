from omegaconf import DictConfig
from frechet_audio_distance import FrechetAudioDistance


def calculate_fad(cfg: DictConfig):
    """Calculate the Frechet Audio Distance."""
    fad = FrechetAudioDistance(
        model_name=cfg.model_name,
        sample_rate=cfg.sample_rate,
        use_pca=cfg.use_pca,
        use_activation=cfg.use_activation,
        verbose=cfg.verbose,
    )
    fad.score(
        background_dir=cfg.background_dir,
        eval_dir=cfg.eval_dir,
        background_embds_path=cfg.background_embds_path,
        eval_embds_path=cfg.eval_embds_path,
        dtype=cfg.dtype,
    )
