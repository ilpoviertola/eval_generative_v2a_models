import argparse
import typing as tp

from omegaconf import OmegaConf

from configs.evaluation_cfg import get_evaluation_config, EvaluationCfg
from metrics.evaluation_metrics import EvaluationMetrics
from metrics.evaluation_metrics_combiner import EvaluationMetricsCombiner


def get_args():
    parser = argparse.ArgumentParser(description="Run evaluations on audio samples.")
    parser.add_argument(
        "--pipeline_cfg",
        "-p",
        type=str,
        nargs="+",
        help="Path(s) to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--plot_dir",
        "-d",
        type=str,
        default=".",
        help="Directory where to save the evaluation plots.",
    )
    return parser.parse_args()


def print_pipeline_cfg(pipeline_cfg_file: tp.List[str]):
    for file in pipeline_cfg_file:
        pipeline_cfg = OmegaConf.load(file)
        print(OmegaConf.to_yaml(pipeline_cfg, resolve=True))


def get_calculated_evaluation_metrics(
    evaluation_cfg: EvaluationCfg, force_recalculate: bool = False
) -> "EvaluationMetrics":
    print(
        f"Evaluating ({evaluation_cfg.id}):", evaluation_cfg.sample_directory.as_posix()
    )
    evaluation_metrics = EvaluationMetrics(evaluation_cfg)
    assert type(evaluation_metrics) == EvaluationMetrics
    evaluation_metrics.run_all(force_recalculate)
    evaluation_metrics.export_results()
    print("Evaluation done\n")
    return evaluation_metrics


def main():
    args = get_args()
    pipeline_cfg_file = args.pipeline_cfg
    print(f"Running evaluations with pipeline configuration(s): {pipeline_cfg_file}")
    print_pipeline_cfg(pipeline_cfg_file)

    all_evaluation_cfgs = []
    all_evaluation_metrics = []
    for file in pipeline_cfg_file:
        eval_cfg = get_evaluation_config(file)
        all_evaluation_cfgs.append(eval_cfg)
        metrics = get_calculated_evaluation_metrics(eval_cfg)
        all_evaluation_metrics.append(metrics)

    evaluation_metrics_combiner = EvaluationMetricsCombiner(all_evaluation_metrics)
    evaluation_metrics_combiner.combine()
    evaluation_metrics_combiner.plot(args.plot_dir)


if __name__ == "__main__":
    main()
