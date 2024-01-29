import sys
import pytest

sys.path.append(".")
from configs.evaluation_cfg import EvaluationCfg
from utils.utils import dataclass_from_dict
from metrics.evaluation_metrics import EvaluationMetrics
from utils.test_utils import sample_dirs, pipeline  # fixtures


@pytest.fixture
def evaluation_cfg(sample_dirs, pipeline):
    return dataclass_from_dict(
        EvaluationCfg,
        {
            "sample_directory": sample_dirs[1],
            "pipeline": pipeline,
            "verbose": True,
        },
    )


def test_evaluation_metrics(evaluation_cfg):
    metrics = EvaluationMetrics(evaluation_cfg)
    metrics.run()
