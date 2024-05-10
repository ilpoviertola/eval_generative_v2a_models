import pytest
from pathlib import Path


@pytest.fixture
def sample_dirs():
    return [
        Path(
            "/home/hdd/ilpo/checkpoints/synchronisonix/24-02-27T16-46-55/24-02-27T16-46-55/generated_samples_24-04-17T14-24-06"
        )
    ]


@pytest.fixture
def pipeline():
    return {
        "fad": {"model_name": "vggish"},
        "kld": {"pretrained_length": 10},
    }


@pytest.fixture
def gt_dir():
    return Path(
        "/home/hdd/ilpo/datasets/greatesthit/test_files-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random"
    )


@pytest.fixture
def cfg_file():
    return Path("configs/vgg_basic.yaml")
