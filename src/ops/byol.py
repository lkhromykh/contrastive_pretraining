import jax
import jax.numpy as jnp
import haiku as hk
import optax
import chex

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.training_state import TrainingState
from src import types_ as types
from src.ops.augmentations import augmentation_fn


def byol(cfg: CoderConfig, networks: CoderNetworks) -> types.StepFn:

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                obs: types.Observation
                ) -> tuple[jax.Array, types.Metrics]:
        rng1, rng2 = jax.random.split(rng)
        view = augmentation_fn(rng1, obs, cfg.shift)
        view_prime = augmentation_fn(rng2, obs, cfg.shift)

        def byol_fn(v, vp):
            y = networks.encoder(params, v)
            z = networks.projector(params, y)
            q = networks.predictor(params, z)
            yp = networks.encoder(target_params, vp)
            zp = networks.projector(target_params, yp)
            return optax.cosine_distance(q, zp).mean(), y

        loss, emb = byol_fn(view, view_prime)
        loss_prime, _ = byol_fn(view_prime, view)
        loss = loss + loss_prime
        return loss, dict(loss=loss, emb_std=emb.std(0).mean())

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        print('Tracing BYOL step.')
        params = state.params
        target_params = state.target_params
        observations = batch['observations']
        rng, subkey = jax.random.split(state.rng)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, target_params, subkey, observations)
        state = state.update(grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state.replace(rng=rng), metrics

    return step
