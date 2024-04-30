from dataclasses import dataclass
from pathlib import Path
import typing as tp


@dataclass
class FADCfg:
    model_name: str = "vggish"
    sample_rate: int = 16_000
    use_pca: bool = False
    use_activation: bool = False
    dtype: str = "float32"
    embeddings_fn: tp.Optional[str] = None

    def __post_init__(self):
        # TODO: checking
        if self.embeddings_fn is None:
            self.embeddings_fn = f"{self.model_name}_embeddings.npy"


@dataclass
class KLDCfg:
    pretrained_length: int = 10
    batch_size: int = 10
    num_workers: int = 10
    duration: float = 2.56

    def __post_init__(self):
        # TODO: checking
        pass


@dataclass
class InSyncCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


# this is basically the same as InSyncCfg
@dataclass
class AVClipScoreCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


@dataclass
class ZCRCfg:
    afps: int = 24_000
    duration: float = 2.56


@dataclass
class RhythmSimilarityCfg:
    afps: int = 24_000
    duration: float = 2.56


@dataclass
class SpectralContrastSimilarityCfg:
    afps: int = 24_000
    duration: float = 2.56


@dataclass
class ImageBindScore:
    device: str = "cuda:0"
    afps: int = 16_000
    get_diagonal_scores: bool = True


@dataclass
class PipelineCfg:
    fad: FADCfg = None
    kld: KLDCfg = None
    insync: InSyncCfg = None
    avclip_score: AVClipScoreCfg = None
    zcr: ZCRCfg = None
    rhythm_similarity: RhythmSimilarityCfg = None
    spectral_contrast_similarity: SpectralContrastSimilarityCfg = None
    imagebind_score: ImageBindScore = None


@dataclass
class EvaluationCfg:
    # unique identifier for the evaluation
    id: str
    # directory containing evaluation data (.mp4 and .wav)
    sample_directory: Path
    # defined evaluation pipeline
    pipeline: PipelineCfg
    # ground truth data (.wav)
    gt_directory: Path
    # metadata
    metadata: tp.Optional[Path] = None
    # directories to save evaluation results
    result_directory: tp.Optional[Path] = None
    verbose: bool = False

    def __post_init__(self):
        assert self.sample_directory.is_dir(), "sample_directory not existing directory"

        if self.pipeline.fad is not None or self.pipeline.kld is not None:
            assert (
                self.gt_directory.is_dir()
            ), "gt_directory must be a Path object to an existing directory"

        if self.result_directory is None:
            self.result_directory = self.sample_directory
        assert (
            self.result_directory.is_dir()
        ), "result_directory must be a Path object to an existing directory"

        self._print_pipeline()

    def _print_pipeline(self):
        print("Evaluation pipeline:")
        print(f"id: {self.id}")
        print(f"sample_directory: {self.sample_directory}")
        print(f"gt_directory: {self.gt_directory}")
        print(f"metadata: {self.metadata}")
        print(f"result_directory: {self.result_directory}")
        print(self.pipeline)
        print(f"verbose: {self.verbose}")
        print()
