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


def supervised(cfg: CoderConfig, networks: CoderNetworks) -> types.StepFn:
    assert cfg.supervised, 'WrongLogic'

    def loss_fn(params: hk.Params,
                rng: types.RNG,
                img: chex.Array,
                label: chex.Array
                ) -> jax.Array:
        chex.assert_type([img, label], [jnp.uint8, int])
        view = augmentation_fn(rng, {types.IMG_KEY: img}, cfg.shift)
        emb = networks.backbone(params, view[types.IMG_KEY] / 255.)
        logits = networks.projector(params, emb)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
        acc = jnp.mean(logits.argmax(-1) == label)
        return loss, dict(acc=acc, loss=loss)

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: tuple[chex.Array, chex.Array]
             ) -> tuple[TrainingState, types.Metrics]:
        print('Tracing classifier step.')
        params = state.params
        rng, subkey = jax.random.split(state.rng)
        images, labels = batch

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, subkey, images, labels)
        state = state.update(grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state.replace(rng=rng), metrics

    return step
