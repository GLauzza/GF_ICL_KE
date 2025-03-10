from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class ModelEditHyperParams(HyperParams):
    insertion_type: str
    layer_to_modify: int
    n_tok_start: int
    n_tok_stop: int
    method: str
    strength: float
    treshold: float