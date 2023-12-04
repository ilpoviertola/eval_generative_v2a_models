"""Main gateway to the application."""

from omegaconf import OmegaConf, DictConfig

from metrics.fad import calculate_fad


def get_cfg() -> DictConfig:
    """Return the config object."""
    args = OmegaConf.from_cli()
    cfg = OmegaConf.load(args.pop("config"))
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.merge(cfg, args)
    return cfg


def main(cfg: DictConfig):
    """Run the application."""
    if cfg.metric == "fad":
        assert cfg.get("fad", None) is not None, "Missing FAD config"
        calculate_fad(cfg)


if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
