from collections.abc import Generator, Mapping, Iterator

import numpy as np
from jax.tree_util import tree_map
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

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

        self._memory = {}
        self._idx = 0
        self._len = 0

    def add(self,
            transition: Mapping[str, np.generic | Mapping[str, np.generic]]
            ) -> None:
        if not self._memory:
            self._allocate(transition)
        # inplace nested memory update via tree_map?
        for k, v in transition.items():
            if isinstance(v, Mapping):
                for vk, vv in v.items():
                    self._memory[k][vk][self._idx] = vv
            else:
                self._memory[k][self._idx] = v
        self._idx += 1
        self._len = max(self._idx, self._len)
        self._idx %= self.capacity

    def as_dataset(self, batch_size: int) -> tf.data.Dataset:
        gen = lambda: self._yield_batch(batch_size)
        signature = tree_map(
            lambda x: tf.TensorSpec(x.shape, x.dtype),
            next(gen())
        )
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=signature
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds.as_numpy_iterator()

    def _yield_batch(self, batch_size: int) -> Generator[types.Trajectory]:
        while True:
            idx = self._rng.integers(0, self._len, batch_size)
            batch = tree_slice(self._memory, idx)
            yield batch

    def _allocate(self, tr: types.Trajectory) -> None:
        """Allocate memory with the same structure as a probe."""
        def empty(x):
            x = np.asanyarray(x)
            return np.zeros((self.capacity, ) + x.shape, x.dtype)
        for k, v in tr.items():
            self._memory[k] = tree_map(empty, v)

    def __len__(self) -> int:
        return self._len

    def save(self, file: str) -> None:
        meta = (self._rng, self.capacity, self._idx, self._len)
        np.savez_compressed(file, meta=meta, **self._memory)

    @classmethod
    def load(cls, file: str) -> 'ReplayBuffer':
        data = np.load(file, allow_pickle=True)
        rng, capacity, idx, len_ = data['meta']
        replay = cls(rng, capacity)
        replay._idx = idx
        replay._len = len_
        for k, v in data.items():
            if k == 'meta':
                continue
            if v.dtype == object:
                v = v.item()
            replay._memory[k] = v
        return replay


def tree_slice(tree_: 'T', sl: slice, is_leaf=None) -> 'T':
    return tree_map(lambda t: t[sl], tree_, is_leaf=is_leaf)
