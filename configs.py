from dataclasses import dataclass
from typing import Optional, Union
import os
from enum import Enum

NUM_DATA_REPEAT = 30
CURRENT_DIR = os.path.dirname(__file__)


class TrainDataset(Enum):
    CSV = "csv"
    COCO = "coco"


@dataclass
class DataConfig:
    image_col: str
    caption_col: str

    pad_token: str

    train_split_ratio: float

    # Text to create vocab for tokenizer
    vocab_path: str
    # Path to data source. Local file for dataset=csv and huggingface path for dataset=coco
    data_path: str
    dataset: TrainDataset  # either 'csv' or 'coco'

    max_seq_length: Optional[int]
    padding: str = "longest"


class ProfilerMode(Enum):
    TRACE = "trace"
    MEMORY = "memory"


@dataclass
class ModelConfig:
    block_size: int
    num_blks: int
    head_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: int
    img_size: int
    patch_size: int
    image_embed_dim: int
    emb_dropout: int
    blk_dropout: int


@dataclass
class TrainerConfig:
    eval_interval: int
    eval_iters: int
    max_iters: Optional[int]
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    lr: float
    use_wandb: bool
    wandb_project: str
    snapshot_dir: str
    snapshot_load_strategy: Optional[Union[str, int]]
    compile: bool
    cuda_graph: bool
    use_amp: bool
    log_verbose: bool
    profile_results_dir: str
    profiler_mode: Optional[ProfilerMode]
    use_ddp: bool = False
