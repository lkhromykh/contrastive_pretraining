import dataclasses

from rltools.config import Config

Layers = tuple[int]

# TODO: LayerNorm, ensemble, symmetric sampling. 2302.02948


@dataclasses.dataclass
class CoderConfig(Config):
    # BYOL
    # https://github.com/deepmind/deepmind-research/blob/master/byol/configs/byol.py
    shift: int = 4
    byol_batch_size: int = 512
    byol_learning_rate: float = 1e-4
    byol_targets_update: float = 5e-3
    byol_steps: int = 100

    # DrQ
    # https://github.com/facebookresearch/drqv2/blob/main/cfgs/config.yaml
    gamma: float = .99
    utd: int = 10
    entropy_coef: float = 1e-3
    num_actions: int = 20
    detach_encoder: bool = False
    drq_batch_size: int = 256
    drq_learning_rate: float = 3e-4
    drq_targets_update: float = 5e-3
    log_every: int = 5
    eval_every: int = 200
    num_evaluations: int = 1
    pretrain_steps: int = 256

    # Architecture
    activation: str = 'relu'
    normalization: str = 'rms'

    cnn_emb_dim: int = 64
    cnn_depths: Layers = (32, 32, 32, 32)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 1, 1, 1)

    act_dim_nbins: int = 3
    actor_layers: Layers = (256, 256)
    critic_layers: Layers = (256, 256)
    ensemble_size: int = 10
    num_critics: int = 2

    # Train common
    buffer_capacity: int = 10_000
    max_grad: float = 50.
    weight_decay: float = 1e-5

    logdir: str = 'logdir/ur_pick_dense'
    task: str = 'ur_pick'
    seed: int = 0
