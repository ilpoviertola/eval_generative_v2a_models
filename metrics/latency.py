from typing import Dict
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf
import torchvision
import torch

from submodules.SparseSync.utils.utils import check_if_file_exists_else_download
from submodules.SparseSync.dataset.dataset_utils import get_video_and_audio
from submodules.SparseSync.dataset.transforms import make_class_grid, quantize_offset
from submodules.SparseSync.scripts.example import (
    reencode_video,
    decode_single_video_prediction,
)
from submodules.SparseSync.scripts.train_utils import (
    get_model,
    get_transforms,
    prepare_inputs,
)


def calculate_latency(cfg: DictConfig):
    cfg_path = f"./logs/sync_models/{cfg.exp_name}/cfg-{cfg.exp_name}.yaml"
    ckpt_path = f"./logs/sync_models/{cfg.exp_name}/{cfg.exp_name}.pt"

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    model_cfg = OmegaConf.load(cfg_path)
    generated_videos_path = Path(cfg.generated_videos)

    device = torch.device(cfg.device)

    # load the model
    _, model = get_model(model_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model"])

    model.eval()
    transforms = get_transforms(model_cfg)["test"]

    results: Dict[str, Dict[str, float]] = {"LATENCY": {}}
    for vid_path in generated_videos_path.glob("*.mp4"):
        # checking if the provided video has the correct frame rates
        vid_path = vid_path.as_posix()
        print(f"Using video: {vid_path}")
        v, a, vid_meta = torchvision.io.read_video(vid_path, pts_unit="sec")
        T, H, W, C = v.shape
        if (
            vid_meta["video_fps"] != cfg.vfps
            or vid_meta["audio_fps"] != cfg.afps
            or min(H, W) != cfg.input_size
        ):
            print(f'Reencoding. vfps: {vid_meta["video_fps"]} -> {cfg.vfps};', end=" ")
            print(f'afps: {vid_meta["audio_fps"]} -> {cfg.afps};', end=" ")
            print(f"{(H, W)} -> min(H, W)={cfg.input_size}")
            vid_path = reencode_video(vid_path, cfg.vfps, cfg.afps, cfg.input_size)
        else:
            print(
                f'No need to reencode. vfps: {vid_meta["video_fps"]}; afps: {vid_meta["audio_fps"]}; min(H, W)={cfg.input_size}'
            )

        # load visual and audio streams
        # (Tv, 3, H, W) in [0, 255], (Ta, C) in [-1, 1]
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)
        rgb = rgb.repeat(3, 1, 1, 1)[:125, ...]
        audio = audio.repeat(3)[:80000, ...]
        item = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": vid_path,
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
        max_off_sec = model_cfg.data.max_off_sec
        grid = make_class_grid(
            -max_off_sec,
            max_off_sec,
            model_cfg.model.params.transformer.params.num_offset_cls,
        )
        # applying the transform
        item = transforms(item)

        # prepare inputs for inference
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, device)

        # forward pass
        _, off_logits = model(vid, aud, targets)

        # simply prints the results of the prediction
        top_prob, top_class = decode_single_video_prediction(off_logits, grid, item)

        results["LATENCY"][vid_path.as_posix()] = {"class": top_class, "prob": top_prob}

    with open(Path(cfg.generated_videos) / "latency.json", "w") as f:
        json.dump(results, f)

    return results
