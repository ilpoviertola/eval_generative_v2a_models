from dataclasses import dataclass
from pathlib import Path
import typing as tp


@dataclass
class FADCfg:
    model_name: str = "encodec"
    sample_rate: int = 24000
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
    duration: float = 2.0

    def __post_init__(self):
        # TODO: checking
        pass


@dataclass
class PipelineCfg:
    fad: FADCfg
    kld: KLDCfg


@dataclass
class EvaluationCfg:
    # directories containing evaluation data (.mp4 and .wav)
    sample_directory: Path
    # defined evaluation pipeline
    pipeline: PipelineCfg
    # ground truth data (.wav)
    gt_directory: Path = Path("/home/hdd/data/greatesthits/evaluation/GT")
    # directories to save evaluation results
    result_directory: tp.Optional[Path] = None
    verbose: bool = False

    def __post_init__(self):
        assert self.sample_directory.is_dir(), "sample_directory not existing directory"

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
        print(f"sample_directory: {self.sample_directory}")
        print(f"gt_directory: {self.gt_directory}")
        print(f"result_directory: {self.result_directory}")
        print(self.pipeline)
        print(f"verbose: {self.verbose}")
