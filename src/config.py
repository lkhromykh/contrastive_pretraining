import dataclasses

from rltools.config import Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class CoderConfig(Config):
    # BYOL
    # https://github.com/deepmind/deepmind-research/blob/master/byol/configs/byol.py
    shift: int = 5
    byol_batch_size: int = 32
    byol_learning_rate: float = 1e-3
    byol_targets_update: float = 5e-3
    byol_steps: int = 1000

    # DrQ-like
    # https://github.com/facebookresearch/drqv2/blob/main/cfgs/config.yaml
    gamma: float = .97
    lambda_: float = 1.
    disag_expl: float = 1.
    utd: int = 10
    use_projection: bool = False
    detach_encoder: bool = False
    drq_batch_size: int = 32
    demo_fraction: float = 0.5  # 2302.02948
    drq_learning_rate: float = 1e-3
    drq_targets_update: float = 1e-2
    log_every: int = 1
    pretrain_steps: int = 4

    # Architecture
    activation: str = 'elu'
    normalization: str = 'layer'

    emb_dim: int = 48
    projector_hid_dim: int = 256
    predictor_hid_dim: int = 256
    cnn_depths: Layers = (32, 32, 32, 32)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 2, 2, 2)
    critic_layers: Layers = (512,)
    ensemble_size: int = 3

    # Train common
    jit: bool = True
    replay_capacity: int = 10_000
    max_grad: float = 50.
    weight_decay: float = 0.

    logdir: str = 'logdir'
    task: str = 'ur_pick'
    time_limit: int = 16
    seed: int = 0
