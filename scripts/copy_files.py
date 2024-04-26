import argparse
import shutil
from pathlib import Path


def copy_files(file_list: Path, source_dir: Path, dest_dir: Path):
    with open(file_list, "r") as f:
        filenames = [line.strip() for line in f]

    for filename in filenames:
        # files = source_dir.glob(f"{filename}*")
        files = [source_dir / filename]
        for file in files:
            if file.suffix == ".mp4":
                shutil.copy2(file, dest_dir / file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy files from source to destination."
    )
    parser.add_argument(
        "--file_list", required=True, help="File with list of filenames."
    )
    parser.add_argument("--source_dir", required=True, help="Source directory path.")
    parser.add_argument("--dest_dir", required=True, help="Destination directory path.")
    args = parser.parse_args()

    file_list = Path(args.file_list)
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copy_files(file_list, source_dir, dest_dir)
