# %% [markdown]
# # Evaluate generative multimodal audio models
#
# Use this notebook to evaluate the generative multimodal audio models. This notebook uses following metrics:
#
# - Frechet Audio Distance (FAD)
# - Kulback-Leibler Divergence (KLD)
# - ImageBind score (IB)
# - Synchronisation Error (SE)
#
# ## Setup
# User must have
#
# 1. Generated audio samples
# 1. GT audios for given videos that were used when generating the audio
#
# ## Configuration

# %%
from pathlib import Path

from utils.utils import dataclass_from_dict
from configs.evaluation_cfg import EvaluationCfg
from metrics.evaluation_metrics import EvaluationMetrics

# %%
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

# %%
for cfg in cfgs:
    print(cfg)
    print()
    metrics = EvaluationMetrics(cfg)
    metrics.run()
