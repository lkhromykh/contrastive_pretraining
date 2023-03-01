import jax
import jax.numpy as jnp

import haiku as hk
import optax

from src.config import CoderConfig
from src.networks import CoderNetworks
from .augmentations import augmentation_fn
from src import types_ as types


def drq_loss(cfg: CoderConfig, networks):

    def actor_loss():
        ...

    def critic_loss():
        ...

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                o_tm1, a_tm1, r_t, disc_t, o_t
                ) -> jax.Array:
        sg = jax.lax.stop_gradient
        rngs = jax.random.split(rng, 4)
        o_tm1 = augmentation_fn(rngs[0], o_tm1, cfg.shift)
        o_t = augmentation_fn(rngs[1], o_t, cfg.shift)

        s_tm1 = networks.encoder(params, o_tm1)
        if cfg.detach_encoder:
            s_tm1 = sg(s_tm1)
        s_t = networks.encoder(target_params, o_t)
        dist_params_t = networks.actor(params, s_tm1)
        policy_t = networks.make_dist(*dist_params_t)

        q_tm1 = networks.critic(params, s_tm1)
        q_tm1 = jnp.min(q_tm1, -1)
        q_t = networks.critic(target_params, s_t)

        target_q_tm1 = r_t + cfg.gamma * disc_t * q_t
        critic_loss = optax.l2_loss(q_tm1[..., jnp.newaxis], target_q_tm1)




