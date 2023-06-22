from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import chex
import haiku as hk

from src import types_ as types
from src.config import CoderConfig

Array = chex.Array


class MLP(hk.Module):

    def __init__(self,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 activate_final: bool = True,
                 name: str | None = None,
                 ) -> None:
        super().__init__(name)
        self.layers = layers
        self.act = act
        self.norm = norm
        self.activate_final = activate_final

    def __call__(self, x: Array) -> Array:
        for idx, layer in enumerate(self.layers):
            x = hk.Linear(layer)(x)
            if idx != len(self.layers) - 1 or self.activate_final:
                x = _get_norm(self.norm)(x)
                x = _get_act(self.act)(x)
        return x


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

        prefix = img.shape[:-3]
        reshape = (np.prod(prefix, dtype=int),) + img.shape[-3:]
        x = jnp.reshape(img / 255., reshape)

        cnn_arch = zip(self.depths, self.kernels, self.strides)
        for depth, kernel, stride in cnn_arch:
            conv = hk.Conv2D(depth, kernel, stride,
                             w_init=hk.initializers.Orthogonal(),
                             padding='valid')
            x = conv(x)
            x = _get_norm(self.norm)(x)
            x = _get_act(self.act)(x)
        x = jnp.reshape(x, prefix + (-1,))
        emb = hk.Linear(self.emb_dim, name='projector')
        return emb(x)


class DQN(hk.Module):

    def __init__(self,
                 act_dim: int,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.act_dim = act_dim
        self.layers = layers
        self.act = act
        self.norm = norm

    def __call__(self, state: Array) -> Array:
        chex.assert_type(state, float)
        x = MLP(self.layers, self.act, self.norm)(state)
        w_init = hk.initializers.TruncatedNormal(stddev=1e-2)
        return hk.Linear(self.act_dim, w_init=w_init)(x)


class CriticsEnsemble(hk.Module):

    def __init__(self,
                 ensemble_size: int,
                 *args,
                 name: str | None = None,
                 **kwargs
                 ) -> None:
        super().__init__(name)
        self.ensemble_size = ensemble_size
        self._factory = lambda n: DQN(*args, name=n, **kwargs)

    def __call__(self, *args, **kwargs) -> Array:
        values = []
        for i in range(self.ensemble_size):
            critic = self._factory('critic_%d' % i)
            values.append(critic(*args, **kwargs))
        return jnp.stack(values, -1)


class CoderNetworks(NamedTuple):
    init: Callable
    encoder: Callable
    predictor: Callable
    critic: Callable
    act: Callable
    split_params: Callable

    @classmethod
    def make_networks(
            cls,
            cfg: CoderConfig,
            observation_spec: types.ObservationSpec,
            action_spec: types.ActionSpec
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
            predictor = hk.Linear(cfg.cnn_emb_dim, name='predictor')
            critic_ = CriticsEnsemble(
                cfg.ensemble_size,
                action_spec.num_values,
                cfg.critic_layers,
                cfg.activation,
                cfg.normalization,
                name='critic'
            )

            def critic(obs: types.Observation) -> types.Array:
                state = []
                for key, spec in sorted(observation_spec.items()):
                    match len(spec.shape), spec.dtype:
                        case 0 | 1, _:
                            feat = jnp.atleast_1d(obs[key])
                        case 3, jnp.uint8:
                            feat = encoder(obs[key])
                            if cfg.detach_encoder:
                                feat = jax.lax.stop_gradient(feat)
                        case _:
                            raise NotImplementedError(key, spec)
                    state.append(feat)
                state = jnp.concatenate(state, -1)
                return critic_(state)

            def act(obs: types.Observation) -> types.Action:
                q_values = critic(obs)
                q_mean = q_values.mean(-1)
                q_std = q_values.std(-1)
                score = q_mean + cfg.disag_expl * q_std
                return score.argmax(-1)

            def init():
                img = encoder(dummy_obs[types.IMG_KEY])
                predictor(img)
                critic(dummy_obs)

            return init, (encoder, predictor, critic, act)

        def split_params(params: hk.Params) -> tuple[hk.Params, ...]:
            modules = ('encoder', 'predictor', 'critic')

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
            critic=apply[2],
            act=apply[3],
            split_params=split_params,
        )


def _get_act(act: str) -> Callable[[Array], Array]:
    if act == 'none':
        return lambda x: x
    if hasattr(jax.lax, act):
        return getattr(jax.lax, act)
    if hasattr(jax.nn, act):
        return getattr(jax.nn, act)
    raise ValueError(act)


def _get_norm(norm: str) -> Callable[[Array], Array]:
    match norm:
        case 'none':
            return lambda x: x
        case 'layer':
            return hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )
        case 'rms':
            return hk.RMSNorm(
                axis=-1,
                create_scale=True
            )
        case _:
            raise ValueError(norm)
