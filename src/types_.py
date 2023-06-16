import collections.abc
from typing import TypedDict, Callable

import jax
import jax.numpy as jnp
import dm_env.specs

from src.training_state import TrainingState

IMG_KEY = 'image'
Array = jnp.ndarray
RNG = jax.random.PRNGKey

Action = Array
ActionSpec = dm_env.specs.DiscreteArray
Observation = collections.abc.MutableMapping[str, Array]
ObservationSpec = collections.abc.MutableMapping[str, dm_env.specs.Array]
Policy = Callable[[Observation], Action]


class Trajectory(TypedDict):
    observations: list[Observation]
    actions: list[Action]
    rewards: list[float]
    discounts: list[float]


Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, jnp.number]
StepFn = Callable[[TrainingState, Trajectory], tuple[TrainingState, Metrics]]
