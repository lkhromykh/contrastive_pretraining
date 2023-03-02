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


def byol(cfg: CoderConfig, networks: CoderNetworks):

    def loss_fn(params: hk.Params,
                target_params: hk.Params,
                rng: types.RNG,
                img: jax.Array
                ) -> jax.Array:
        chex.assert_rank(img, 3)
        chex.assert_type(img, jnp.uint8)

        k1, k2 = jax.random.split(rng)
        view = augmentation_fn(k1, img, cfg.shift)
        view_prime = augmentation_fn(k2, img, cfg.shift)

        def byol_fn(v, vp):
            y = networks.encoder(params, v)
            z = networks.predictor(params, y)
            target_y = networks.encoder(target_params, vp)
            return optax.cosine_distance(z, target_y)

        return byol_fn(view, view_prime) + byol_fn(view_prime, view)

    @chex.assert_max_traces(2)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        params = state.params
        target_params = state.target_params
        imgs = batch['observations'][types.IMG_KEY]
        rngs = jax.random.split(state.rng, cfg.byol_batch_size + 1)

        in_axes = 2 * (None,) + 2 * (0,)
        grad_fn = jax.value_and_grad(loss_fn)
        grad_fn = jax.vmap(grad_fn, in_axes=in_axes)
        out = grad_fn(params, target_params, rngs[:-1], imgs)
        loss, grads = jax.tree_util.tree_map(lambda t: jnp.mean(t, 0), out)

        state = state.update(grads)
        metrics = dict(byol_loss=loss, byol_grads_norm=optax.global_norm(grads))
        return state._replace(rng=rngs[-1]), metrics

    return step


