import jax
import jax.numpy as jnp

from .types_ import Trajectory


class ReplayBuffer:

    def __init__(self,
                 capacity: int,
                 observation_spec,
                 action_spec,
                 mem='cpu'
                 ) -> None:
        new_shape = lambda sh: (capacity,) + sh
        self._observations = jnp.zeros(
            new_shape(observation_spec.shape),
            observation_spec.dtype
        )
        self._next_observations = jnp.zeros(
            new_shape(observation_spec.shape),
            observation_spec.dtype
        )
        self._actions = jnp.zeros(
            new_shape(action_spec.shape),
            action_spec.dtype
        )
        self._rewards = jnp.zeros((capacity, 1), jnp.float32)
        self._discounts = jnp.zeros((capacity, 1), jnp.float32)

        self.capacity = capacity
        self._cur_idx = 0

