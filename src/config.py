import dataclasses

from rltools.config import Config
from src.types_ import Layers


@dataclasses.dataclass
class CoderConfig(Config):
    # BYOL
    shift: int = 4
    byol_batch_size: int = 256
    byol_learning_rate: float = 1e-3
    byol_targets_update: int | float = 1e-2

    # DrQ
    gamma: float = .99
    num_actions: int = 20
    entropy_coef: float | None = None
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
    actor_min_std: float = .1
    actor_max_std: float = .9
    num_critic_heads: int = 2
    critic_layers: Layers = (256, 256)

    buffer_capacity: int = 1e6

    logdir: str = '/dev/null'
    seed: int = 0
