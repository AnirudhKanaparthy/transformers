from enum import Enum
from dataclasses import dataclass
import time 


class DataSplit(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


@dataclass
class TransformerConfig:
    vocabulary_size: int
    embedding_dim: int
    context_length: int
    number_of_layers: int
    number_of_heads: int
    linear_layer_scale: int = 4
    dropout: float = 0.5
    device: str = 'cpu'


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    context_length: int
    maximum_iterations: int
    eval_intervals: int
    eval_iterations: int
    checkpoint_interval: int
    checkpoint_dir: str
    device: str = 'cpu'
    project_name: str = 'transformer_training'
    experiment_name: str = f'run_{int(time.time())}'
