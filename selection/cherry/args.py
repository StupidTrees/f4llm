import io
import json
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class DataAnalysisArguments:
    data_path: str
    save_path: str
    model_name_or_path: str
    max_length: int = 512
    start_idx: int = 0
    end_idx: int = -1
    prompt: str = 'alpaca'
    mod: str = 'pre'
    quant: int = 32


@dataclass
class DataClusteringArguments:
    pt_data_path: str
    json_data_path: str
    json_save_path: str
    sent_type: int = 0
    ppl_type: int = 0
    cluster_method: str = 'kmeans'
    reduce_method: str = 'tsne'
    sample_num: int = 10
    kmeans_num_clusters: int = 100
    low_th: int = 1
    up_th: int = 99


@dataclass
class SiftingByIFDArguments:
    pt_data_path: str
    json_data_path: str
    json_save_path: str
    model_name_or_path: str
    max_length: int = 1024
    sample_rate: float = 0.1
    sample_number: int = 0
    prompt: str = 'alpaca'


@dataclass
class CherryModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class CherryDataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class CherrySFTArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA optimizer."})
    quant: int = field(default=32, metadata={"help": "Quantization bits."})


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
