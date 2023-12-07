"""Main gateway to the application."""
from omegaconf import OmegaConf, DictConfig

from metrics.fad import calculate_fad
from metrics.kld import calculate_kld


def get_cfg() -> DictConfig:
    """Get OmegaConf config.

    Returns:
        DictConfig: Parsed config.
    """
    args = OmegaConf.from_cli()
    cfg = OmegaConf.load(args.pop("config"))
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.merge(cfg, args)
    assert type(cfg) == DictConfig, "Config must be a DictConfig"
    return cfg


def main(cfg: DictConfig):
    """Run the application."""
    if cfg.metric == "fad":
        calculate_fad(cfg)
    elif cfg.metric == "kld":
        calculate_kld(cfg)


if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
