from collections.abc import Generator

import numpy as np
from jax.tree_util import tree_map

from src import types_ as types


# TODO: avoid memory duplication for obs/next_obs -- store structured traj instead.
class ReplayBuffer:
    """Not so sophisticated buffer to store transitions directly on GPU."""

    def __init__(self,
                 rng: np.random.Generator,
                 capacity: int,
                 ) -> None:
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self._rng = rng
        self.capacity = capacity

        self._memory = None
        self._idx = 0
        self._len = 0

    def add(self, tr: types.Trajectory) -> None:
        if self._memory is None:
            self._allocate(tr)
        # inplace nested memory update via tree_map?
        for k, v in tr.items():
            if isinstance(v, dict):
                for vk, vv in v.items():
                    self._memory[k][vk][self._idx] = vv
            else:
                self._memory[k][self._idx] = v
        self._idx = (self._idx + 1) % self.capacity
        self._len = max(self._idx, self._len)

    def as_generator(self, batch_size: int) -> Generator[types.Trajectory]:
        while True:
            idx = self._rng.integers(0, self._len, batch_size)
            batch = tree_slice(self._memory, idx)
            yield batch

    def _allocate(self, tr: types.Trajectory) -> None:
        """Allocate memory with the same structure as a probe."""
        self._memory = {}

        def empty(x):
            x = np.asanyarray(x)
            return np.zeros((self.capacity, ) + x.shape, x.dtype)
        for k, v in tr.items():
            self._memory[k] = tree_map(empty, v)

    def __len__(self) -> int:
        return self._len

    def save(self, file: str) -> None:
        np.savez(file, **self._memory)

    def load(self, file: str) -> None:
        assert self._memory is None
        data = np.load(file, allow_pickle=True)
        self._memory = {}
        for k, v in data.items():
            self._memory[k] = v


def tree_slice(tree_: 'T', sl: slice) -> 'T':
    return tree_map(lambda t: t[sl], tree_)
