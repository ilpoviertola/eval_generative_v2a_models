import sys
import pytest
from pathlib import Path


sys.path.append(".")
from configs.evaluation_cfg import EvaluationCfg, FADCfg, KLDCfg, PipelineCfg
from eval_utils.utils import dataclass_from_dict
from eval_utils.test_utils import sample_dirs, pipeline  # fixtures


def test_init_evaluation_cfg(sample_dirs, pipeline):
    cfgs = []
    for sample_directory in sample_dirs:
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

    assert len(cfgs) == 2
    for i in range(len(cfgs)):
        assert cfgs[i].sample_directory == sample_dirs[i]
        assert cfgs[i].result_directory == sample_dirs[i]
        assert type(cfgs[i].pipeline) == PipelineCfg
        assert type(cfgs[i].pipeline.fad) == FADCfg
        assert type(cfgs[i].pipeline.kld) == KLDCfg
        assert cfgs[i].pipeline.fad.model_name == pipeline["fad"]["model_name"]
        assert (
            cfgs[i].pipeline.kld.pretrained_length
            == pipeline["kld"]["pretrained_length"]
        )
        assert cfgs[i].gt_directory == Path("/home/hdd/data/greatesthits/evaluation/GT")
