import typing as tp
from pathlib import Path

import torch
import numpy as np
import av
from torchvision.io import read_video
from torchvision.transforms import Compose, CenterCrop


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: tp.Optional[tp.Dict[str, tp.Any]] = None,
    audio_array: tp.Optional[torch.Tensor] = None,
    audio_fps: tp.Optional[float] = None,
    audio_codec: tp.Optional[str] = None,
    audio_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = a_stream.format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout
            )

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def transform_videos(
    file_paths: tp.List[Path],
    output_dir: Path,
    start_secs: tp.Optional[tp.List[float]] = None,
):
    transforms = Compose([CenterCrop(224)])
    start_secs = start_secs or [0] * len(file_paths)
    for i, file_path in enumerate(file_paths):
        output_path = output_dir / file_path.name
        if output_path.exists():
            continue

        # Load video
        video, audio, info = read_video(
            file_path.as_posix(),
            start_pts=start_secs[i],
            end_pts=start_secs[i] + 2.56,
            pts_unit="sec",
        )
        video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        video = transforms(video)
        video = video.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        write_video(
            output_path.as_posix(),
            video,
            fps=info["video_fps"],
            audio_array=audio,
            audio_fps=info["audio_fps"],
            audio_codec="aac",
            video_codec="h264",
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )


if __name__ == "__main__":
    # video_dir = "/home/hdd/ilpo/evaluation_data/specvqgan/2021-07-30T21-34-41_vggsound_transformer/samples_2024-04-19T18-59-19/sparse-videos"
    # video_files = [
    #     "-KqXcm-I2zY_87000_97000",
    #     "_2hETAEaX3c_106000_116000",
    #     "_jB-IM_77lI_0_10000",
    #     "-OJzsMV1G1A_61000_71000",
    #     "-Wsuo4VDwfE_30000_40000",
    #     "-ZSgg6jFUd8_688000_698000",
    #     "alBcRErUVVg_7000_17000",
    #     "AuwXsdruL7I_30000_40000",
    #     "BQPk4cmN__4_30000_40000",
    #     "HiM0cqYAV7Q_20000_30000",
    #     "QEE70zlVRoM_19000_29000",
    # ]

    # start_secs = [2.36, 4.6, 2.96, 4.76, 3.68, 3.2, 2.16, 0.84, 2.8, 1.0, 1.36]
    video_dir = "/home/hdd/ilpo/evaluation_data/specvqgan/2024-04-16T14-16-42_greatesthit_transformer/samples_2024-04-17T09-54-50/GreatestHits_test/videos/partition1"
    video_files = [
        "2015-09-28-14-21-29-422_denoised_167",
        "2015-09-29-13-27-53-50_denoised_422",
        "2015-09-29-15-17-35-790_denoised_290",
        "2015-09-29-15-44-54-1223_denoised_416",
        "2015-09-29-16-42-22-112_denoised_102",
        "2015-09-30-20-27-11-81_denoised_249",
        "2015-09-30-20-56-02-31_denoised_376",
        "2015-10-02-11-26-31-1_denoised_408",
        "2015-10-02-11-46-40-608_denoised_1144",
        "2015-03-29-16-56-48_denoised_907",
    ]
    file_paths = [Path(f"{video_dir}/{file}.mp4") for file in video_files]
    output_dir = Path(f"{video_dir}/transformed")
    output_dir.mkdir(exist_ok=True, parents=True)
    start_secs = [0.0] * len(file_paths)
    transform_videos(file_paths, output_dir, start_secs)
