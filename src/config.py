import dataclasses

from rltools.config import Config
from src.types_ import Layers


@dataclasses.dataclass
class CoderConfig(Config):
    # BYOL
    # https://github.com/deepmind/deepmind-research/blob/master/byol/configs/byol.py
    shift: int = 4
    byol_batch_size: int = 256
    byol_learning_rate: float = 1e-3
    byol_targets_update: int | float = 1e-2

    # DrQ
    # https://github.com/facebookresearch/drqv2/blob/main/cfgs/config.yaml
    gamma: float = .99
    num_actions: int = 20
    entropy_coef: float = 0.
    detach_encoder: bool = True
    drq_batch_size: int = 256
    drq_learning_rate: float = 1e-3
    drq_targets_update: int | float = 1e-2

    # Architecture
    activation: str = 'relu'
    normalization: str = 'none'

    cnn_emb_dim: int = 256
    cnn_depths: Layers = (3, 3, 3)
    cnn_kernels: Layers = (3, 3, 3)
    cnn_strides: Layers = (2, 2, 2)

    actor_layers: Layers = (256, 256)
    critic_layers: Layers = (256, 256)
    num_critic_heads: int = 2

    buffer_capacity: int = 1e6

    logdir: str = '/dev/null'
    seed: int = 0
