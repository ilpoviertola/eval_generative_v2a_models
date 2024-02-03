from typing import Dict, Tuple, Union
from pathlib import Path

from omegaconf import OmegaConf
import torch

from submodules.SparseSync.utils.utils import check_if_file_exists_else_download
from submodules.SparseSync.dataset.dataset_utils import get_video_and_audio
from submodules.SparseSync.dataset.transforms import make_class_grid
from submodules.SparseSync.scripts.example import (
    decode_single_video_prediction,
)
from submodules.SparseSync.scripts.train_utils import (
    get_model,
    get_transforms,
    prepare_inputs,
)

from eval_utils.file_utils import reencode_dir_if_needed, rmdir_and_contents


def calculate_insync(
    samples: str,
    exp_name: str,
    afps: int,
    vfps: int,
    input_size: int,
    device: str,
    ckpt_parent_path: str,
    verbose: bool = False,
) -> Tuple[float, Dict[str, Dict[str, Union[int, float]]]]:
    cfg_path = f"{ckpt_parent_path}/{exp_name}/cfg-{exp_name}.yaml"
    ckpt_path = f"{ckpt_parent_path}/{exp_name}/{exp_name}.pt"

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    model_cfg = OmegaConf.load(cfg_path)
    generated_videos_path = Path(samples)

    device = torch.device(device)

    # load the model
    _, model = get_model(model_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model"])

    model.eval()
    transforms = get_transforms(model_cfg)["test"]

    # reencode data if needed
    generated_videos_path, reencoded = reencode_dir_if_needed(
        generated_videos_path, vfps, afps, input_size
    )
    print("Reencoded" if reencoded else "No need to reencode")

    results: Dict[str, Dict[str, Union[int, float]]] = {}
    batch = []
    videos = list(generated_videos_path.glob("*.mp4"))
    insync_samples = 0
    for i, vid_path in enumerate(videos):
        vid_path_str = vid_path.as_posix()
        # load visual and audio streams
        # (Tv, 3, H, W) in [0, 255], (Ta, C) in [-1, 1]
        rgb, audio, meta = get_video_and_audio(vid_path_str, get_meta=True)
        item = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": vid_path_str,
            "split": "test",
            "targets": {
                # setting the start of the visual crop and the offset size.
                # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
                # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
                # track by `args.offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
                # start `offset_sec` earlier than the rgb track.
                # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (see `grid`)
                "v_start_i_sec": 0,
                "offset_sec": 0,
                # dummy values -- don't mind them
                "vggsound_target": 0,
                "vggsound_label": "PLACEHOLDER",
            },
        }
        # applying the transform
        item = transforms(item)
        batch.append(item)
        if len(batch) == 3 or i == len(videos) - 1:
            # prepare inputs for inference
            batch = torch.utils.data.default_collate(batch)
            aud, vid, targets = prepare_inputs(batch, device)

            # forward pass
            _, off_logits = model(vid, aud, targets)
            off_logits = off_logits.detach().cpu()
            off_probs = torch.softmax(off_logits, dim=-1).detach().cpu()

            for i in range(len(batch["path"])):
                vid_path = Path(batch["path"][i])
                top_class = off_probs[i].argmax()
                insync_samples += int(top_class.item())
                results[(vid_path.parent / vid_path.name).as_posix()] = {
                    "class": top_class.item(),
                    "prob": round(off_probs[i][top_class].item(), 3),
                }

            batch = []

    if reencoded:
        rmdir_and_contents(generated_videos_path)

    return insync_samples / len(results), results
