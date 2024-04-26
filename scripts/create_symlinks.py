import argparse
from pathlib import Path


def match_all_gh_filenames(file_name: str, source_dir: Path):
    all_files = list(source_dir.glob(f"{file_name}_denoised*"))
    return [f.name for f in list(all_files)]


def create_symlinks(file_list: Path, source_dir: Path, dest_dir: Path):
    with open(file_list, "r") as f:
        filenames = [line.strip() + ".mp4" for line in f]

    for filename in filenames:
        source = source_dir / filename
        destination = dest_dir / filename
        destination.symlink_to(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create symbolic links from source to destination."
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

    create_symlinks(file_list, source_dir, dest_dir)
