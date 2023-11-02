import collections.abc
from typing import Any, Callable, TypedDict

import jax
import dm_env.specs

from src.training_state import TrainingState

IMG_KEY = 'realsense/image'
Array = RNG = jax.Array

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
Metrics = collections.abc.MutableMapping[str, Any]
StepFn = Callable[[TrainingState, Trajectory], tuple[TrainingState, Metrics]]
