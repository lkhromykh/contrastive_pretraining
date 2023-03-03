from collections.abc import Generator

import chex
import jax
import jax.numpy as jnp
import numpy as np

from src import types_ as types


class ReplayBuffer:
    """Not so sophisticated buffer to store transitions directly on GPU."""

    def __init__(self,
                 rng: types.RNG,
                 capacity: int,
                 backend: str = 'cpu'
                 ) -> None:
        self._rng = rng
        self.capacity = capacity
        self.backend = backend

        self._memory = None
        self._device = jax.devices(backend)[0]
        self._add = jax.jit(_replace, static_argnums=2, backend=backend)
        self._sample = jax.jit(_sample, static_argnums=2, backend=backend)
        self._len = 0
        self._idx = 0

    def add(self, tr: types.Trajectory) -> None:
        tr = jax.tree_map(jnp.stack, tr, is_leaf=lambda x: isinstance(x, list))
        if self._memory is None:
            self._allocate(tr)
        self._memory, self._idx = self._add(self._memory, tr, self._idx)
        self._len = self._idx if self._idx > self._len else self.capacity

    def as_generator(self, batch_size: int) -> Generator[types.Trajectory]:
        while True:
            self._rng, rng = jax.random.split(self._rng)
            yield self._sample(rng, self._memory, batch_size, self._len)

    def _allocate(self, tr: types.Trajectory) -> None:
        """Allocate memory with the same structure as a probe."""
        def to_device(x): return x if self.use_vram else jax.device_get(x)
        def empty(x): return jnp.zeros((self.capacity, ) + x.shape[1:], x.dtype)
        self._memory = jax.tree_util.tree_map(
            lambda x: to_device(empty(x)),
            tr
        )

    def __len__(self) -> int:
        return self._len

    def save(self, file) -> None:
        mem = jax.device_get(self._memory)
        np.savez(file, **mem)

    def load(self, file) -> None:
        self._memory = np.load(file)
        if self.use_vram:
            self._memory = jax.device_put(self._memory)


def tree_slice(tree, sl: slice, is_leaf=None):
    return jax.tree_util.tree_map(
        lambda t: t[sl],
        tree,
        is_leaf=is_leaf
    )


@chex.assert_max_traces(2)
def _replace(memory: types.Trajectory,
             data: types.Trajectory,
             start_idx: int
             ) -> tuple[types.Trajectory, int]:
    cap = len(memory['actions'])  # handle range
    l = len(data['actions'])
    last_idx = start_idx + l
    for k in data:
        memory[k] = memory[k].at[start_idx:last_idx].set(data[k])
    return memory, last_idx


@chex.assert_max_traces(2)
def _sample(rng: types.RNG,
            memory: types.Trajectory,
            batch_size: int,
            max_idx: int
            ) -> types.Trajectory:
    idx = jax.random.randint(rng, (batch_size,), 0, max_idx)
    return tree_slice(memory, idx)
