"""Main gateway to the application."""
from omegaconf import OmegaConf, DictConfig

from metrics.fad import calculate_fad
from metrics.kld import calculate_kld
from metrics.pdm import calculate_pdm
from metrics.xcorr import calculate_xcorr
from metrics.latency import calculate_latency


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


def main(pipeline: DictConfig):
    """Run the application."""
    for cfg in pipeline.pipeline:
        if cfg.metric == "fad":
            calculate_fad(cfg.params)
        elif cfg.metric == "kld":
            calculate_kld(cfg.params)
        elif cfg.metric == "pdm":
            calculate_pdm(cfg.params)
        elif cfg.metric == "xcorr":
            calculate_xcorr(cfg.params)
        elif cfg.metric == "latency":
            calculate_latency(cfg.params)


if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
