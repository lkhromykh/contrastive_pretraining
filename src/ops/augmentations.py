import jax.numpy as jnp

import chex
import dm_pix


def random_shift(rng: chex.PRNGKey,
                 img: chex.Array,
                 crop_size: int
                 ) -> chex.Array:
    """Crop HW dims preserving original shape with padding."""
    chex.assert_scalar_positive(crop_size)
    chex.assert_rank(img, 3)  # HWC

    shape = img.shape
    pad = (crop_size // 2, crop_size - crop_size // 2)
    pad_with = (pad, pad, (0, 0))
    img = jnp.pad(img, pad_with, mode='edge')
    return dm_pix.random_crop(rng, img, shape)


# May be any compose function
augmentation_fn = random_shift
