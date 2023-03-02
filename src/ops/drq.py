import jax
import jax.numpy as jnp

import haiku as hk
import chex
import optax

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.training_state import TrainingState
from .augmentations import augmentation_fn
from src import types_ as types


def drq(cfg: CoderConfig, networks: CoderNetworks):

    def make_state(rng, params, obs):
        img = augmentation_fn(rng, obs[types.IMG_KEY], cfg.shift)
        img = networks.encoder(params, img)
        if cfg.detach_encoder:
            img = jax.lax.stop_gradient(img)
        return jnp.concatenate([img, obs[types.PROPRIO_KEY]])

    def critic_loss_fn(value, target_value):
        chex.assert_rank([value, target_value], [1, 0])
        target_value = jax.lax.stop_gradient(target_value)
        return jnp.square(value - target_value[jnp.newaxis])

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                o_tm1, a_tm1, r_t, disc_t, o_t
                ) -> jax.Array:
        chex.assert_rank([r_t, a_tm1], [0, 2])
        rngs = jax.random.split(rng, 3)

        s_tm1 = make_state(rngs[0], params, o_tm1)
        s_t = make_state(rngs[1], target_params, o_t)

        policy_t = networks.actor(params, s_t)
        policy_tm1 = networks.actor(params, s_tm1)
        entropy_tm1 = policy_tm1.entropy()

        a_t = policy_t.sample(seed=rngs[2], shape=(cfg.num_actions,))

        q_fn = jax.vmap(networks.critic, in_axes=(None, None, 0))
        q_t = q_fn(target_params, s_t, a_t)
        v_t = q_t.min().mean()
        q_tm1 = networks.critic(params, s_tm1, a_tm1)
        target_q_tm1 =\
            r_t + cfg.entropy_coef * entropy_tm1 +\
            cfg.gamma * disc_t * v_t
        critic_loss = critic_loss_fn(q_tm1, target_q_tm1)
        actor_loss = - v_t

        metrics = dict(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            entropy=entropy_tm1,
            reward=r_t,
            value=v_t
        )
        return critic_loss + actor_loss, metrics

    @chex.assert_max_traces(2)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        params = state.params
        target_params = state.target_params

        o_tm1, a_tm1, r_t, disc_t, o_t = map(
            batch.get,
            ('observations', 'actions', 'rewards', 'discounts', 'next_observations')
        )
        rngs = jax.random.split(state.rng, cfg.drq_batch_size + 1)
        in_axes = 2 * (None,) + 6 * (0,)
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=in_axes)

        out = grad_fn(
            params, target_params, rngs[:-1],
            o_tm1, a_tm1, r_t, disc_t, o_t
        )
        grads, metrics = jax.tree_util.tree_map(lambda t: jnp.mean(t, 0), out)

        state = state.update(grads)
        metrics.update(drq_grads_norm=optax.global_norm(grads))
        return state._replace(rng=rngs[-1]), metrics

    return step
