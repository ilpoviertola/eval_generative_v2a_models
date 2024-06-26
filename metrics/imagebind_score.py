from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from submodules.ImageBind.imagebind.data import (
    load_and_transform_video_data,
    load_and_transform_audio_data,
)
from submodules.ImageBind.imagebind.models import imagebind_model
from submodules.ImageBind.imagebind.models.imagebind_model import ModalityType


BATCH_SIZE = 3


def calculate_imagebind_score(
    video_dir: Path,
    device: str,
    get_diagonal_scores: bool = True,
    afps: int = 16000,
    verbose: bool = False,
):
    # get videos
    all_videos = list(video_dir.glob("*.mp4"))

    # load model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    running_score = 0
    # run model inference
    for i in tqdm(
        range(0, len(all_videos), BATCH_SIZE), desc="Calculating ImageBind score"
    ):
        # load video and audio data
        video_data = load_and_transform_video_data(
            all_videos[i : i + BATCH_SIZE],
            device,
            sample_rate=afps,
        )
        audio_data = load_and_transform_audio_data(
            all_videos[i : i + BATCH_SIZE],
            device,
            sample_rate=afps,
        )
        inputs = {
            ModalityType.AUDIO: audio_data,
            ModalityType.VISION: video_data,
        }

        with torch.no_grad():
            embeddings = model(inputs)

        sim_scores = torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=1
        )
        sim_scores = sim_scores.cpu().numpy()

        if get_diagonal_scores:
            running_score += np.sum(sim_scores.diagonal())
        else:
            max_indices = np.argmax(sim_scores, axis=1)
            diag_indices = np.arange(sim_scores.shape[0])
            is_diag_max = max_indices == diag_indices
            running_score += np.sum(is_diag_max)

    score = running_score / len(all_videos)
    if verbose:
        print("ImageBind score:", score)
    return float(score)
