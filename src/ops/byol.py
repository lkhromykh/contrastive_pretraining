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
                img: jax.Array
                ) -> jax.Array:
        chex.assert_type(img, jnp.uint8)
        k1, k2 = jax.random.split(rng)
        view = augmentation_fn(k1, img, cfg.shift)
        view_prime = augmentation_fn(k2, img, cfg.shift)

        def byol_fn(v, vp):
            y = networks.encoder(params, v)
            z = networks.predictor(params, y)
            target_y = networks.encoder(target_params, vp)
            return optax.cosine_distance(z, target_y), y

        loss, projection = byol_fn(view, view_prime)
        loss_prime, _ = byol_fn(view_prime, view)
        return jnp.mean(loss + loss_prime), projection

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        print('Tracing BYOL step.')
        params = state.params
        target_params = state.target_params
        imgs = batch['observations'][types.IMG_KEY]
        rng, subkey = jax.random.split(state.rng)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, proj), grad = grad_fn(params, target_params, subkey, imgs)
        state = state.update(grad)
        metrics = dict(loss=loss,
                       grad_norm=optax.global_norm(grad),
                       proj_std=jnp.std(proj, 0).mean())
        return state.replace(rng=rng), metrics

    return step
