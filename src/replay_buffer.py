from collections import deque
from collections.abc import Generator

import jax
import jax.numpy as jnp

from src import types_ as types


class ReplayBuffer:

    def __init__(self, rng: types.RNG, capacity: int, gpu: bool = True) -> None:
        self._rng = rng
        self._gpu = gpu
        self._memory = deque(maxlen=capacity)

    def add(self, tr: types.Trajectory) -> None:
        if not self._gpu:
            tr = jax.device_get(tr)
        for i in range(len(tr['actions'])):
            self._memory.append(tree_slice(tr, i))

    def as_generator(self, batch_size: int) -> Generator[types.Trajectory]:
        while True:
            self._rng, rng = jax.random.split(self._rng)
            idx = jax.random.randint(rng, (batch_size,), 0, len(self._memory))
            batch = [self._memory[i] for i in idx]
            batch = jax.tree_util.tree_map(
                lambda *t: jnp.stack(t),
                *batch
            )
            yield jax.device_put(batch)


def tree_slice(tree, sl: slice, is_leaf=None):
    return jax.tree_util.tree_map(
        lambda t: t[sl],
        tree,
        is_leaf=is_leaf
    )
