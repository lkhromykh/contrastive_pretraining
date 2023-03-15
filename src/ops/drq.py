from typing import Callable

import jax
import jax.numpy as jnp

import haiku as hk
import chex
import optax

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.training_state import TrainingState
from src.ops.augmentations import augmentation_fn
from src import types_ as types


def drq(cfg: CoderConfig, networks: CoderNetworks) -> Callable:

    def critic_loss_fn(value, target_value):
        chex.assert_rank([value, target_value], [1, 0])
        target_value = jax.lax.stop_gradient(target_value)
        return jnp.square(value - target_value).mean()

    def actor_loss_fn(policy, q_values, actions):
        chex.assert_rank([q_values, actions], [1, 3])
        q_values, actions = jax.lax.stop_gradient((q_values, actions))
        normalized_weights = jax.nn.softmax(q_values / cfg.entropy_coef)
        return -jnp.sum(normalized_weights * policy.log_prob(actions))

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                o_tm1, a_tm1, r_t, disc_t, o_t
                ) -> jax.Array:
        chex.assert_rank([r_t, a_tm1], [0, 2])
        rngs = jax.random.split(rng, 4)

        o_tm1[types.IMG_KEY] = augmentation_fn(
            rngs[0], o_tm1[types.IMG_KEY], cfg.shift)
        o_t[types.IMG_KEY] = augmentation_fn(
            rngs[1], o_t[types.IMG_KEY], cfg.shift)
        s_tm1 = networks.make_state(params, o_tm1)
        s_t = networks.make_state(target_params, o_t)

        policy_t = networks.actor(params, s_t)
        entropy_t = policy_t.entropy()
        a_t = policy_t.sample(seed=rngs[2], sample_shape=(cfg.num_actions,))

        critic_idxs = jax.random.choice(
            rngs[3], cfg.ensemble_size, (cfg.num_critics,), replace=False)
        q_fn = jax.vmap(networks.critic, in_axes=(None, None, 0))
        q_t = q_fn(target_params, s_t, a_t)
        q_t = jnp.maximum(0, q_t)
        v_t = q_t[critic_idxs].mean(0).min() + cfg.entropy_coef * entropy_t
        target_q_tm1 = r_t + cfg.gamma * disc_t * v_t

        q_tm1 = networks.critic(params, s_tm1, a_tm1)
        critic_loss = critic_loss_fn(q_tm1, target_q_tm1)
        actor_loss = actor_loss_fn(policy_t, q_t.mean(1), a_t)

        metrics = dict(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            entropy=entropy_t,
            reward=r_t,
            value=v_t
        )
        return critic_loss + actor_loss, metrics

    def _step(state: TrainingState,
              batch: types.Trajectory
              ) -> tuple[TrainingState, types.Metrics]:
        """Single batch step."""
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
        encoder_gn, _, actor_gn, critic_gn = map(
            optax.global_norm,
            networks.split_params(grads)
        )
        named_grads = {
            'encoder_grad_norm': encoder_gn,
            'actor_grad_norm': actor_gn,
            'critic_grad_norm': critic_gn
        }
        metrics.update(named_grads)
        return state._replace(rng=rngs[-1]), metrics

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        """Fusing multiple updates."""
        chex.assert_shape(batch['rewards'], (cfg.utd * cfg.drq_batch_size,))
        print('Tracing DrQ step.')
        rng, subkey = jax.random.split(state.rng)
        state = state._replace(rng=rng)
        idxs = jnp.arange(len(batch['actions']))
        metrics = []
        for group in jnp.split(idxs, cfg.utd):
            subbatch = jax.tree_util.tree_map(lambda x: x[group], batch)
            state, met = _step(state, subbatch)
            metrics.append(met)
        metrics = jax.tree_util.tree_map(lambda *t: jnp.stack(t), *metrics)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return state, metrics

    return step
