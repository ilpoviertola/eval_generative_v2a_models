import typing as tp
from dataclasses import asdict, fields

from configs.evaluation_cfg import EvaluationCfg
from metrics.fad import calculate_fad


class EvaluationMetrics:
    def __init__(self, cfg: EvaluationCfg) -> None:
        self.cfg = cfg
        self.results: tp.Dict[str, tp.Any] = {}

    def run(self) -> None:
        pipeline = self.cfg.pipeline
        scores = {}

        for field in fields(pipeline):
            if field.name.lower() == "fad":
                score = calculate_fad(
                    gts=self.cfg.gt_directory.as_posix(),
                    samples=self.cfg.sample_directory.as_posix(),
                    sample_embds_path=self.cfg.sample_directory.as_posix(),
                    gt_embds_path=self.cfg.gt_directory.as_posix(),
                    verbose=self.cfg.verbose,
                    **asdict(pipeline.fad),
                )
                scores["fad"] = score
            elif field.name.lower() == "kld":
                pass
            else:
                raise ValueError(f"Unknown metric {field.name.lower()}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(".")
    from utils.utils import dataclass_from_dict

    sample_directories = [
        Path(
            "/home/hdd/data/greatesthits/evaluation/24-01-21T19-41-37/generated_samples_24-01-22T15-17-01"
        ),
        Path(
            "/home/hdd/data/greatesthits/evaluation/24-01-10T11-25-02/generated_samples_24-01-21T19-25-28"
        ),
    ]

    pipeline = {
        "FAD": {"model_name": "encodec"},
        "KLD": {"pretrained_length": 10},
    }
    cfgs = []
    for sample_directory in sample_directories:
        cfgs.append(
            dataclass_from_dict(
                EvaluationCfg,
                {
                    "sample_directory": sample_directory,
                    "pipeline": pipeline,
                    "gt_directory": Path("/home/hdd/data/greatesthits/evaluation/GT"),
                },
            )
        )

    for cfg in cfgs:
        metrics = EvaluationMetrics(cfg)
        metrics.run()
