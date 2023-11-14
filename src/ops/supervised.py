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
    assert cfg.supervised, 'LogicError'

    def loss_fn(params: hk.Params,
                rng: types.RNG,
                img: chex.Array,
                label: chex.Array
                ) -> tuple[jax.Array, types.Metrics]:
        # chex.assert_type([img, label], [jnp.uint8, {float, int}])
        view = augmentation_fn(rng, {types.IMG_KEY: img}, cfg.hue_max_delta)
        emb = networks.backbone(params, view[types.IMG_KEY] / 255.)
        logits = networks.projector(params, emb)
        _, top1 = jax.lax.top_k(logits, 1)
        _, top5 = jax.lax.top_k(logits, 5)
        ilabel = label.argmax(-1, keepdims=True)
        top1 = jnp.any(top1 == ilabel, -1).mean()
        top5 = jnp.any(top5 == ilabel, -1).mean()
        loss = optax.softmax_cross_entropy(logits, label).mean()
        return loss, dict(loss=loss, top1=top1, top5=top5)

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: tuple[chex.Array, chex.Array]
             ) -> tuple[TrainingState, types.Metrics]:
        print('Tracing classification step.')
        params = state.params
        rng, subkey = jax.random.split(state.rng)
        images, labels = batch

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, subkey, images, labels)
        state = state.update(grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state.replace(rng=rng), metrics

    return step
