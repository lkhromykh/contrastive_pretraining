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
        select = hk.BatchApply(jax.vmap(lambda q, a: q[a]))
        return select(q_values, actions)

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
        chex.assert_shape(
            [r_t, a_t, disc_t], (cfg.time_limit, cfg.drq_batch_size))

        TIME_DIM, BATCH_DIM, ACT_DIM, QS_DIM = range(4)
        tc_aug = jax.vmap(  # time consistent augmentation
            augmentation_fn, in_axes=(None, TIME_DIM, None), out_axes=TIME_DIM)
        obs_t[types.IMG_KEY] = tc_aug(rng, obs_t[types.IMG_KEY], cfg.shift)
        q_t = networks.critic(params, obs_t)
        a_dash_t = q_t.mean(QS_DIM).argmax(ACT_DIM)
        q_t = select_actions(q_t[:-1], a_t)
        target_q_t = networks.critic(target_params, obs_t)
        adv_t = target_q_t.max(ACT_DIM) - target_q_t.min(ACT_DIM)
        q_std = target_q_t.std(QS_DIM)
        pi_t = (a_t == a_dash_t[:-1]).astype(q_t.dtype)
        v_tp1 = select_actions(target_q_t, a_dash_t)[1:]
        target_q_t = select_actions(target_q_t[:-1], a_t)
        sampled_q_std = target_q_t.std(QS_DIM)

        in_axes = 5 * (BATCH_DIM,) + (None,)
        target_fn = jax.vmap(tree_backup, in_axes=in_axes, out_axes=BATCH_DIM)
        in_axes = 2 * (QS_DIM,) + 4 * (None,)
        target_fn = jax.vmap(target_fn, in_axes=in_axes, out_axes=QS_DIM)

        disc_t *= cfg.gamma
        target_q_t = target_fn(target_q_t, v_tp1,
                               r_t, disc_t,
                               pi_t, cfg.lambda_)
        target_q_t = target_q_t.min(QS_DIM, keepdims=True)
        critic_loss = jnp.square(q_t - target_q_t).mean()

        metrics = dict(
            critic_loss=critic_loss,
            reward=r_t,
            pi=pi_t,
            value=v_tp1,
            advantage=adv_t,
            q_ensemble_std=q_std,
            q_ensemble_std_sampled=sampled_q_std
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
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
