import argparse
from multiprocessing import Pool
from moviepy.editor import AudioFileClip
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path):
    video_clip = AudioFileClip(str(video_path))
    video_clip.write_audiofile(str(audio_path))


def extract_audio_from_videos(video_paths: list, audio_paths: list):
    with Pool() as pool:
        pool.starmap(extract_audio, zip(video_paths, audio_paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from videos.")
    parser.add_argument("--input", required=True, help="Input directory path.")
    parser.add_argument("--output", required=True, help="Output directory path.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = list(input_dir.glob("*.mp4"))
    audio_paths = [output_dir / f"{video_path.stem}.wav" for video_path in video_paths]

    extract_audio_from_videos(video_paths, audio_paths)
