import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.training_state import TrainingState
from src import types_ as types
from .augmentations import augmentation_fn


def byol(config: CoderConfig, networks: CoderNetworks):

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                img: jax.Array
                ) -> jax.Array:
        chex.assert_rank(img, 3)
        chex.assert_type(img, float)

        k1, k2 = jax.random.split(rng)
        view = augmentation_fn(k1, img, config.shift)
        target_view = augmentation_fn(k2, img, config.shift)

        y = networks.encoder(params, view)
        target_y = networks.encoder(target_params, target_view)
        z = networks.predictor(params, y)
        loss = optax.cosine_distance(z, target_y)
        return jnp.mean(loss)

    def step(state: TrainingState, batch: types.Trajectory) -> TrainingState:
        imgs = batch["observations"]
        imgs = networks.preprocess(imgs)
        rngs = jax.random.split(state.rng, config.byol_batch_size + 1)
        in_axes = 2 * (None,) + 2 * (0,)
        grad_fn = jax.grad(loss_fn)
        grad_fn = jax.vmap(grad_fn, in_axes=in_axes)

        grads = grad_fn(state.params, state.target_params, rngs[:-1], imgs)
        grads = jax.tree_util.tree_map(lambda t: jnp.mean(t, 0), grads)
        state = state.update(grads, config.byol_targets_update)

        return state._replace(
            rng=rngs[-1]
        )

    return step


