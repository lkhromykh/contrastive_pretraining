import jax
import jax.numpy as jnp

import haiku as hk
import chex
import optax

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.training_state import TrainingState
from src.ops.augmentations import augmentation_fn
from src.ops.tb_lambda import tree_backup
from src import types_ as types


def drq(cfg: CoderConfig, networks: CoderNetworks) -> types.StepFn:

    def select_actions(q_values, actions):
        gather = hk.BatchApply(jax.vmap(lambda src, idx: src[idx]))
        return gather(q_values, actions)

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                obs_t: types.Observation,
                a_t: types.Action,
                r_t: types.Array,
                disc_t: types.Array
                ) -> jax.Array:
        chex.assert_tree_shape_prefix(
            obs_t, (cfg.time_limit + 1, cfg.drq_batch_size))
        chex.assert_tree_shape_prefix(
            [r_t, a_t, disc_t], (cfg.time_limit, cfg.drq_batch_size))
        del rng  # TODO: actually introduce augmentations
        # q_t.shape = (..., num_actions, num_critics)
        q_t = networks.critic(params, obs_t)
        a_dash_t = q_t.mean(-1).argmax(-1)
        q_t = select_actions(q_t[:-1], a_t)
        target_q_t = networks.critic(target_params, obs_t)
        pi_t = (a_t == a_dash_t[:-1]).astype(q_t.dtype)
        v_tp1 = select_actions(target_q_t, a_dash_t)[1:]
        target_q_t = select_actions(target_q_t[:-1], a_t)

        in_axes = 5 * (1,) + (None,)
        target_fn = jax.vmap(tree_backup, in_axes=in_axes,
                             out_axes=1, axis_name='batch')
        in_axes = 2 * (-1,) + 4 * (None,)
        target_fn = jax.vmap(target_fn, in_axes=in_axes,
                             out_axes=-1, axis_name='q_ensemble')

        disc_t *= cfg.gamma
        target_q_t = target_fn(target_q_t, v_tp1,
                               r_t, disc_t,
                               pi_t, cfg.lambda_)
        target_q_t = target_q_t.min(-1, keepdims=True)
        critic_loss = jnp.square(q_t - target_q_t).mean()

        metrics = dict(
            critic_loss=critic_loss,
            reward=r_t.mean(),
            pi_t=pi_t.mean(),
            q_t=q_t.mean()
        )
        return critic_loss, metrics

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        """Single batch step."""
        params = state.params
        target_params = state.target_params
        rng, subkey = jax.random.split(state.rng)
        batch = jax.tree_util.tree_map(lambda t: jnp.swapaxes(t, 0, 1), batch)
        args = map(
            batch.get,
            ('observations', 'actions', 'rewards', 'discounts')
        )
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, target_params, subkey, *args)

        state = state.update(grad)
        encoder_gn, _, critic_gn = map(
            optax.global_norm,
            networks.split_params(grad)
        )
        metrics.update(encoder_grad_norm=encoder_gn, critic_grad_norm=critic_gn)
        return state.replace(rng=rng), metrics

    return step
