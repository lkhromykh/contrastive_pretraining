import jax
import jax.numpy as jnp
import chex
import haiku as hk
import dm_pix
import numpy as np

from src import types_ as types


def batched_random_hue(rng: chex.PRNGKey,
                       img: chex.Array,
                       max_delta: float
                       ) -> chex.Array:
    chex.assert_type(img, int)
    op = dm_pix.random_hue
    prefix = img.shape[:-3]
    if not prefix:
        return op(rng, img, max_delta)
    batch = np.prod(prefix, dtype=int)
    rngs = jax.random.split(rng, batch)
    rngs = rngs.reshape(prefix + (2,))
    op = jax.vmap(op, (0, 0, None))
    op = hk.BatchApply(op, len(prefix))
    return op(rngs, img, max_delta)


def augmentation_fn(rng: chex.PRNGKey,
                    obs: types.Observation,
                    *args,
                    ) -> types.Observation:
    # Inplace update differs for jit'ed and plain functions,
    # so here explicit copy is used.
    aobs = obs.copy()
    img = aobs[types.IMG_KEY]
    chex.assert_type(img, jnp.uint8)
    img = img.astype(jnp.float32) / 255.
    img = batched_random_hue(rng, img, *args)
    aobs[types.IMG_KEY] = img
    return aobs
