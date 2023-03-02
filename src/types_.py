import collections.abc
from typing import TypedDict

import jax
import jax.numpy as jnp

import dm_env.specs

RNG = jax.random.PRNGKey
Array = jnp.ndarray

Action = Array
ActionSpecs = dm_env.specs.BoundedArray
Observation = collections.abc.MutableMapping[str, Array]
ObservationSpecs = collections.abc.Sequence[str, dm_env.specs.Array]

Policy = collections.abc.Callable[[Observation], Action]
Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, Array]

IMG_KEY = 'image'
PROPRIO_KEY = 'proprio'


class Trajectory(TypedDict, total=False):
    observations: list[Observation]
    actions: list[Action]
    reward: list[float]
    discounts: list[float]
    next_observations: list[Observation]
