import dataclasses

from rltools.config import Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class CoderConfig(Config):
    # BYOL
    # https://github.com/deepmind/deepmind-research/blob/master/byol/configs/byol.py
    shift: int = 4
    byol_batch_size: int = 256
    byol_learning_rate: float = 1e-3
    byol_targets_update: float = 5e-3
    byol_steps: int = 2000000
    # Or use supervised pretraining from the ImageNet instead.
    # Training hparams are shared with BYOL's one.
    supervised: bool = True
    mixup_lambda: float = 0.0

    # DrQ-like
    # https://github.com/facebookresearch/drqv2/blob/main/cfgs/config.yaml
    gamma: float = .96
    lambda_: float = 1.
    utd: int = 10
    detach_encoder: bool = False
    drq_batch_size: int = 32
    demo_fraction: float = 0.5  # 2302.02948
    drq_learning_rate: float = 1e-3
    drq_targets_update: float = 1e-2
    log_every: int = 5
    pretrain_steps: int = 4

    # Architecture
    activation: str = 'elu'
    normalization: str = 'layer'

    emb_dim: int = 64
    projector_hid_dim: int = 256
    predictor_hid_dim: int = 256
    cnn_depths: Layers = (32, 32, 32, 32)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 2, 1, 1)
    critic_layers: Layers = (512,)
    ensemble_size: int = 3

    # Train common
    jit: bool = True
    replay_capacity: int = 500
    max_grad: float = 50.
    weight_decay: float = 1e-6

    logdir: str = 'logdir'
    task: str = 'ur_pick'
    time_limit: int = 16
    seed: int = 1
