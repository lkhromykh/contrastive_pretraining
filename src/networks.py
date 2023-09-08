from collections.abc import Callable
from typing import NamedTuple

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
                 activate_final: bool = False,
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


class CNN(hk.Module):

    def __init__(self,
                 depths: types.Layers,
                 kernels: types.Layers,
                 strides: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name=name)
        self.depths = depths
        self.kernels = kernels
        self.strides = strides
        self.act = act
        self.norm = norm

    def __call__(self, x: Array) -> Array:
        chex.assert_type(x, float)
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])
        cnn_arch = zip(self.depths, self.kernels, self.strides)
        for depth, kernel, stride in cnn_arch:
            conv = hk.Conv2D(depth, kernel, stride,
                             w_init=hk.initializers.Orthogonal(),
                             padding='valid')
            x = conv(x)
            x = _get_norm(self.norm)(x)
            x = _get_act(self.act)(x)
        return x.reshape(prefix + (-1,))


class Encoder(hk.Module):

    def __init__(self,
                 emb_dim: int,
                 backbone: hk.Module,
                 early_fusion: bool = True,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.emb_dim = emb_dim
        self.backbone = backbone
        self.early_fusion = early_fusion

    def __call__(self, obs: types.Observation) -> Array:
        img = obs[types.IMG_KEY]
        chex.assert_type(img, jnp.uint8)
        img = img.astype(jnp.float32) / 255.
        low_dim = [v for k, v in sorted(obs.items()) if k != types.IMG_KEY]
        low_dim = jnp.concatenate(low_dim, -1)

        if self.early_fusion:
            emb = self._early(img, low_dim)
        else:
            emb = self._late(img, low_dim)
        emb = hk.Linear(self.emb_dim)(emb)
        emb = _get_norm('layer')(emb)
        return jnp.tanh(emb)

    def _early(self, img: Array, low_dim: Array) -> Array:
        h, w = img.shape[-3:-1]
        x = jnp.expand_dims(low_dim, (-2, -3))
        x = jnp.tile(x, (h, w, 1))
        x = jnp.concatenate([img, x], -1)
        return self.backbone(x)

    def _late(self, img: Array, low_dim: Array) -> Array:
        emb = self.backbone(img)
        return jnp.concatenate([emb, low_dim], -1)


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
        mlp = MLP(self.layers, self.act, self.norm, activate_final=True)
        x = mlp(state)
        return hk.Linear(self.act_dim)(x)


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
    backbone: Callable
    encoder: Callable
    projector: Callable
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
            cnn = CNN(
                cfg.cnn_depths,
                cfg.cnn_kernels,
                cfg.cnn_strides,
                cfg.activation,
                cfg.normalization,
                name='backbone'
            )
            encoder = Encoder(
                cfg.emb_dim,
                cnn,
                not cfg.supervised,
                name='encoder'
            )
            if cfg.supervised:
                layers = (1000,)  # num_classes in the dataset
            else:
                layers = (cfg.projector_hid_dim, cfg.emb_dim)
            projector = MLP(
                layers,
                cfg.activation,
                cfg.normalization,
                name='projector'
            )
            predictor = MLP(
                (cfg.predictor_hid_dim, cfg.emb_dim),
                cfg.activation,
                cfg.normalization,
                name='predictor'
            )
            critic_ = CriticsEnsemble(
                cfg.ensemble_size,
                action_spec.num_values,
                cfg.critic_layers,
                cfg.activation,
                cfg.normalization,
                name='critic'
            )

            def critic(obs: types.Observation) -> types.Array:
                state = encoder(obs)
                if cfg.detach_encoder:
                    state = jax.lax.stop_gradient(state)
                return critic_(state)

            def act(obs: types.Observation) -> types.Action:
                q_values = critic(obs).mean(-1)
                return q_values.argmax(-1)

            def init():
                x = encoder(dummy_obs)
                if cfg.supervised:
                    img = dummy_obs[types.IMG_KEY] / 255.
                    projector(cnn(img))
                else:
                    projector(x)
                predictor(x)
                critic(dummy_obs)

            return init, (cnn, encoder, projector, predictor, critic, act)

        def split_params(params: hk.Params) -> tuple[hk.Params, ...]:
            modules = ('backbone', 'encoder', 'projector', 'predictor', 'critic')

            def split_fn(module, n, v) -> int:
                name = module.split('/')[0]
                return modules.index(name)

            return hk.data_structures.partition_n(
                split_fn, params, len(modules))

        init, apply = model
        return cls(
            init=init,
            backbone=apply[0],
            encoder=apply[1],
            projector=apply[2],
            predictor=apply[3],
            critic=apply[4],
            act=apply[5],
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
