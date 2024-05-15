"""
These classes provide abstractions over evaluated files.
It can get a bit messy when working with multiple different versions of same files.
"""

import typing as tp
from pathlib import Path
import hashlib
import concurrent
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from eval_utils.exceptions import ConfigurationError
from eval_utils.file_utils import check_is_file, save_audio_from_video, reencode_video


def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()


class EvaluationVideoDirectory:

    def __init__(self) -> None:
        # stores the different variations of the same file
        # variations have the same data but different samplerates, frame sizes, and/or encodings
        self.video_variations: tp.Dict[str, tp.List[EvaluationVideo]] = {}

    def __len__(self) -> int:
        return len(self.video_variations)

    def add_evaluation_videos(self, evaluation_videos: tp.List[tp.Type]) -> None:
        for evaluation_video in evaluation_videos:
            self.add_evaluation_video(evaluation_video)

    def add_evaluation_video(self, evaluation_video: tp.Type) -> None:
        assert isinstance(evaluation_video, EvaluationVideo)
        if evaluation_video.id not in self.video_variations:
            self.video_variations[evaluation_video.id] = []
        self.video_variations[evaluation_video.id].append(evaluation_video)

    def add_video_from_path(
        self,
        video_file_path: tp.Union[str, Path],
        vfps: int,
        afps: int,
        vcodec: str,
        acodec: str,
        is_ground_truth: bool,
        min_side: int = 256,
        extract_audio: bool = False,
        is_original_file: bool = False,
        audio_file_path: tp.Optional[tp.Union[str, Path]] = None,
        gt_evaluation_video_object: tp.Optional[tp.Type] = None,
        start_time: tp.Optional[float] = None,
        end_time: tp.Optional[float] = None,
        duration: tp.Optional[float] = None,
        id: tp.Optional[str] = None,
    ):
        # resolve video file path
        if isinstance(video_file_path, str):
            video_file_path = Path(video_file_path).resolve()

        id = hash_string(video_file_path.name) if not id else id
        try:
            evaluation_video = EvaluationVideo(
                video_file_path=video_file_path,
                vfps=vfps,
                afps=afps,
                vcodec=vcodec,
                acodec=acodec,
                min_side=min_side,
                is_ground_truth=is_ground_truth,
                extract_audio=extract_audio,
                audio_file_path=audio_file_path,
                gt_evaluation_video_object=gt_evaluation_video_object,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                id=id,
                is_original_file=is_original_file,
            )
        except ConfigurationError as e:
            raise e
        else:
            if id not in self.video_variations:
                self.video_variations[id] = []
            self.video_variations[id].append(evaluation_video)

    def remove_videos_by_name(
        self, name: str, delete_files: bool = False, force: bool = False
    ) -> None:
        id = hash_string(name)
        if id in self.video_variations:
            if delete_files:
                for video in self.video_variations[id]:
                    video.delete(force)
            del self.video_variations[id]
        else:
            print(f"Video(s) with name {name} not found in the directory.")

    def remove_videos_by_names(
        self, names: tp.List[str], delete_files: bool = False, force: bool = False
    ):
        for name in names:
            self.remove_videos_by_name(name, delete_files, force)

    def remove_all_videos(
        self, delete_files: bool = False, force: bool = False
    ) -> None:
        for id in list(self.video_variations.keys()):
            if delete_files:
                for video in self.video_variations[id]:
                    video.delete(force)
            del self.video_variations[id]

    def create_new_variatons(
        self,
        names: tp.Optional[tp.List[str]] = None,
        ids: tp.Optional[tp.List[str]] = None,
        vfps: int = 25,
        afps: int = 24000,
        vcodec: str = "h264",
        acodec: str = "aac",
        min_side: int = 256,
        for_ground_truth: bool = False,
        extract_audio: bool = False,
        parallel: bool = False,
    ):
        if not ids:
            ids = []
        if names:
            for name in names:
                ids.append(hash_string(name))
        if not ids:
            raise TypeError("Names or ids must be provided.")

        if parallel:
            variation_results = []
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self.create_new_variation,
                        None,
                        id,
                        vfps,
                        afps,
                        vcodec,
                        acodec,
                        min_side,
                        for_ground_truth,
                        extract_audio,
                        True,
                    ): id
                    for id in ids
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Creating new variations",
                ):
                    try:
                        result = future.result()
                        variation_results.append(result)
                    except Exception as exc:
                        print(f"Generated an exception: {exc}")

            # add the results to the directory
            # multiprocessing does not share class state between the processes
            # so adding the results to the directory is done in the main process
            for result in variation_results:
                self.add_evaluation_video(result)
        else:
            for id in tqdm(ids, desc="Creating new variations"):
                self.create_new_variation(
                    None,
                    id,
                    vfps,
                    afps,
                    vcodec,
                    acodec,
                    min_side,
                    for_ground_truth,
                    extract_audio,
                )

    def create_new_variation(
        self,
        name: tp.Optional[str] = None,
        id: tp.Optional[str] = None,
        vfps: int = 25,
        afps: int = 24000,
        vcodec: str = "h264",
        acodec: str = "aac",
        min_side: int = 256,
        for_ground_truth: bool = False,
        extract_audio: bool = False,
        parallel_exec: bool = False,
    ):
        if not id and not name:
            raise TypeError("Name or id must be provided.")
        if id and name:
            raise TypeError("Provide either name or id, not both.")
        id = hash_string(name) if not id else id
        if id not in self.video_variations:
            print(f"No original video found for the given ID ({id}) or name ({name}).")
            print("Add the original video through 'add_video_from_path'")
            print("or 'add_evaluation_video' method.")
            return

        # get the original video or the ground truth video if for_ground_truth is True
        original_evaluation_video = None
        for video in self.video_variations[id]:
            if video.is_original_file or (for_ground_truth and video.is_ground_truth):
                original_evaluation_video = video
                break
        if not original_evaluation_video:
            print(f"No original video found for the given ID ({id}) or name ({name}).")
            print("Make sure you configure the original video correctly.")
            return

        # create a new variation with the new samplerates, frame sizes, and/or encodings
        if (
            vfps == original_evaluation_video.vfps
            and afps == original_evaluation_video.afps
            and vcodec == original_evaluation_video.vcodec
            and acodec == original_evaluation_video.acodec
        ):
            print("No changes in the new variation.")
            return
        new_folder_name = (
            f"video_{vcodec}_{vfps}fps_{min_side}side_audio_{afps}hz_{acodec}"
        )
        new_path = original_evaluation_video.video_file_path.parent / new_folder_name
        new_path.mkdir(exist_ok=True)
        new_path = new_path / original_evaluation_video.video_file_path.name
        reencode_video(
            path=original_evaluation_video.video_file_path.as_posix(),
            vfps=vfps,
            afps=afps,
            min_side=min_side,
            acodec=acodec,
            vcodec=vcodec,
            new_path=new_path.as_posix(),
        )
        assert new_path.exists()

        if parallel_exec:
            # will be added to the directory in the main process
            return EvaluationVideo(
                video_file_path=new_path,
                vfps=vfps,
                afps=afps,
                vcodec=vcodec,
                acodec=acodec,
                min_side=min_side,
                is_ground_truth=for_ground_truth,
                extract_audio=extract_audio,
                is_original_file=False,
                gt_evaluation_video_object=(
                    original_evaluation_video.gt_evaluation_video_object
                    if not for_ground_truth
                    else None
                ),
                id=id,
            )
        else:
            # add to directory
            self.add_video_from_path(
                video_file_path=new_path,
                vfps=vfps,
                afps=afps,
                vcodec=vcodec,
                acodec=acodec,
                min_side=min_side,
                is_ground_truth=for_ground_truth,
                extract_audio=extract_audio,
                is_original_file=False,
                gt_evaluation_video_object=(
                    original_evaluation_video.gt_evaluation_video_object
                    if not for_ground_truth
                    else None
                ),
                id=id,
            )


class EvaluationVideo:

    def __init__(
        self,
        video_file_path: Path,
        vfps: int,
        afps: int,
        vcodec: str,
        acodec: str,
        is_ground_truth: bool,
        min_side: int = 256,
        extract_audio: bool = False,
        audio_file_path: tp.Optional[tp.Union[str, Path]] = None,
        gt_evaluation_video_object: tp.Optional[tp.Type] = None,
        start_time: tp.Optional[float] = None,
        end_time: tp.Optional[float] = None,
        duration: tp.Optional[float] = None,
        id: tp.Optional[str] = None,
        is_original_file: bool = False,
    ) -> None:
        if not check_is_file(video_file_path):
            raise ConfigurationError(f"{video_file_path} is not existing video file.")
        self.video_file_path = video_file_path

        # handle other variables
        # TODO: add checks for fps and codecs or read them from the video file
        self.vfps = vfps
        self.afps = afps
        self.vcodec = vcodec
        self.acodec = acodec
        self.min_side = min_side
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.is_ground_truth = is_ground_truth
        self.gt_evaluation_video_object = gt_evaluation_video_object
        self.is_original_file = is_original_file or is_ground_truth

        # handle audio
        self.audio_file_path: tp.Optional[Path] = None
        if not audio_file_path:
            if extract_audio:
                self.audio_file_path = save_audio_from_video(video_file_path, afps)
            else:
                self.audio_file_path = None
        else:
            if isinstance(audio_file_path, str):
                audio_file_path = Path(audio_file_path).resolve()
            if not check_is_file(audio_file_path):
                raise ConfigurationError(
                    f"{audio_file_path} is not existing audio file."
                )
            self.audio_file_path = audio_file_path

        # link file to other variations of the same file
        self.id = hash_string(self.video_file_path.name) if not id else id

    def set_gt_evaluation_video_object(self, evaluation_video: tp.Type) -> None:
        assert isinstance(evaluation_video, EvaluationVideo)
        assert evaluation_video.is_ground_truth
        self.gt_evaluation_video_object = evaluation_video

    def delete(self, force: bool = False):
        if self.is_ground_truth:
            # TODO: maybe delete audiofile?
            return
        if (self.is_original_file and force) or not self.is_original_file:
            self.video_file_path.unlink()
            if self.audio_file_path:
                self.audio_file_path.unlink()