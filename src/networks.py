from collections.abc import Callable
from typing import NamedTuple

import dm_env
from dm_env import specs

import jax
import jax.numpy as jnp
import chex
import haiku as hk
import tensorflow_probability.substrates.jax.distributions as tfd

from src import types_ as types
from src.config import CoderConfig

Array = jnp.ndarray


class MLP(hk.Module):

    def __init__(self,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 activate_final: bool = True,
                 name: str | None = None,
                 ) -> None:
        super().__init__(name)

        mlp = []
        for idx, layer in enumerate(layers):
            mlp.append(hk.Linear(layer))
            if idx != len(layers) - 1 or activate_final:
                mlp.append(_get_norm(norm))
                mlp.append(_get_act(act))

        self._mlp = hk.Sequential(mlp)

    def __call__(self, x: Array) -> Array:
        return self._mlp(x)


class Encoder(hk.Module):

    def __init__(self,
                 emb_dim: int,
                 depths: types.Layers,
                 kernels: types.Layers,
                 strides: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)

        self.emb_dim = emb_dim
        self.depths = depths
        self.kernels = kernels
        self.strides = strides
        self.act = act
        self.norm = norm
        
    def __call__(self, img: Array) -> Array:
        chex.assert_type(img, jnp.uint8)
        chex.assert_rank(img, 3)  # "HWC"

        x = img / 255.
        iter_ = zip(self.depths, self.kernels, self.strides)
        for d, k, s in iter_:
            x = hk.Conv2D(d, k, s, padding='valid')(x)
            x = _get_norm(self.norm)(x)
            x = _get_act(self.act)(x)

        emb = hk.Linear(self.emb_dim, name="projector")
        return emb(x.flatten())


class Actor(hk.Module):

    def __init__(self,
                 action_spec: dm_env.specs.BoundedArray,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        assert len(action_spec.shape) == 2,\
            "Intentionally Supports only discretized spaces."
        self.action_spec = action_spec
        self.layers = layers
        self.act = act
        self.norm = norm

    def __call__(self, state: Array) -> Array:
        chex.assert_rank(state, 1)
        chex.assert_type(state, float)

        state = MLP(self.layers, self.act, self.norm)(state)
        act_sh = self.action_spec.shape
        fc = hk.Linear(act_sh[0] * act_sh[1],
                       w_init=hk.initializers.TruncatedNormal(1e-3)
                       )
        logits = fc(state).reshape(act_sh)
        dist = tfd.OneHotCategorical(logits)
        return tfd.Independent(dist, 1)


class Critic(hk.Module):

    def __init__(self,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)

        self.layers = layers
        self.act = act
        self.norm = norm

    def __call__(self,
                 state: Array,
                 action: Array,
                 ) -> Array:
        chex.assert_rank([state, action], [1, 2])
        chex.assert_type([state, action], float)

        x = jnp.concatenate([state, action.flatten()], -1)
        x = MLP(self.layers, self.act, self.norm)(x)
        fc = hk.Linear(1, w_init=hk.initializers.TruncatedNormal(1e-2))
        return fc(x)


class CriticsEnsemble(hk.Module):

    def __init__(self,
                 num_critics: int,
                 *args,
                 name: str | None = None,
                 **kwargs
                 ) -> None:
        super().__init__(name)
        self.num_critics = num_critics
        self._factory = lambda n: Critic(*args, name=n, **kwargs)

    def __call__(self, *args, **kwargs):
        values = []
        for i in range(self.num_critics):
            critic = self._factory(f'critic_{i}')
            values.append(critic(*args, **kwargs))
        return jnp.concatenate(values, -1)


class CoderNetworks(NamedTuple):
    init: Callable
    encoder: Callable
    predictor: Callable
    actor: Callable
    critic: Callable
    split_params: Callable

    @classmethod
    def init(
            cls,
            cfg: CoderConfig,
            observation_spec: dm_env.specs.Array,
            action_spec: dm_env.specs.BoundedArray
    ) -> 'CoderNetworks':
        dummy_obs = jax.tree_map(
            lambda sp: sp.generate_value(),
            observation_spec
        )

        @hk.without_apply_rng
        @hk.multi_transform
        def model():
            encoder = Encoder(
                cfg.cnn_emb_dim,
                cfg.cnn_depths,
                cfg.cnn_kernels,
                cfg.cnn_strides,
                cfg.activation,
                cfg.normalization,
                name='encoder'
            )
            predictor = hk.Linear(cfg.cnn_emb_dim, name="predictor")
            actor = Actor(
                action_spec,
                cfg.actor_layers,
                cfg.activation,
                cfg.normalization,
                name='actor'
            )
            critic = CriticsEnsemble(
                cfg.num_critic_heads,
                cfg.critic_layers,
                cfg.activation,
                cfg.normalization,
                name='critic'
            )

            def init():
                img = encoder(dummy_obs[types.IMG_KEY])
                img = predictor(img)
                state = jnp.concatenate([img, dummy_obs[types.PROPRIO_KEY]])
                dist = actor(state)
                critic(state, dist.mean())

            return init, (encoder, predictor, actor, critic)

        def split_params(params: hk.Params) -> tuple[hk.Params]:
            modules = ('encoder', 'predictor', 'actor', 'critic')

            def split_fn(module, n, v) -> int:
                name = module.split('/')[0]
                return modules.index(name)

            return hk.data_structures.partition_n(
                split_fn, params, len(modules))

        init, apply = model
        return cls(
            init=init,
            encoder=apply[0],
            predictor=apply[1],
            actor=apply[2],
            critic=apply[3],
            split_params=split_params,
        )


def _get_act(act: str) -> Callable[[Array], Array]:
    if act == "none":
        return lambda x: x
    if hasattr(jax.lax, act):
        return getattr(jax.lax, act)
    if hasattr(jax.nn, act):
        return getattr(jax.nn, act)
    raise ValueError(act)


def _get_norm(norm: str) -> Callable[[Array], Array]:
    if norm == "none":
        return lambda x: x
    if norm == "layer":
        # investigate if scale should be created.
        return hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)
    raise ValueError(norm)
