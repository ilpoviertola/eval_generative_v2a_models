import pytest
from pathlib import Path


@pytest.fixture
def sample_dirs():
    return [
        Path(
            "/home/hdd/data/greatesthits/evaluation/23-12-22T09-12-38/generated_samples_24-01-05T11-55-13"
        ),
        Path(
            "/home/hdd/data/greatesthits/evaluation/23-12-20T00-45-15/generated_samples_23-12-20T09-17-40"
        ),
    ]


@pytest.fixture
def pipeline():
    return {
        "fad": {"model_name": "encodec"},
        "kld": {"pretrained_length": 10},
    }
