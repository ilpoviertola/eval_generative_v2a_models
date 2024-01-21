"""Main gateway to the application."""
import json
from pathlib import Path
from datetime import datetime

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
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    all_scores = []
    all_scores.append(
        {
            "sample_dir": pipeline.get("samples", ""),
            "gt_dir": pipeline.get("audio_gts", ""),
        }
    )
    for cfg in pipeline.pipeline:
        if cfg.metric == "fad":
            score_item = calculate_fad(cfg.params)
        elif cfg.metric == "kld":
            score_item = calculate_kld(cfg.params)
        elif cfg.metric == "pdm":
            score_item = calculate_pdm(cfg.params)
        elif cfg.metric == "xcorr":
            score_item = calculate_xcorr(cfg.params)
        elif cfg.metric == "latency":
            score_item = calculate_latency(cfg.params)
        else:
            raise ValueError(f"Unknown metric: {cfg.metric}")

        all_scores.append(score_item)

    print(all_scores)

    save_path = Path(pipeline.get("score_save_path", "./")) / f"scores_{timestamp}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
