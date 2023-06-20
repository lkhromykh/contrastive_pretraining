import typing

import numpy as np
from jax import tree_util


class SpecsLike(typing.Protocol):
    shape: tuple[int, ...]
    dtype: type


T = typing.TypeVar('T')
Nested = typing.Union[T, 'Nested[T]']


class ReplayBuffer:

    def __init__(self,
                 rng: np.random.Generator | int,
                 capacity: int,
                 signature: Nested[SpecsLike]
                 ) -> None:
        self._rng = np.random.default_rng(rng)
        self.capacity = capacity
        self.signature = signature

        leaves, self._treedef = tree_util.tree_flatten(signature)
        self._num_leaves = len(leaves)
        self._memory = ReplayBuffer.tile_with(leaves, capacity, np.zeros)
        self._idx = 0
        self._len = 0

    def add(self, transition: Nested[np.ndarray]) -> None:
        leaves, struct = tree_util.tree_flatten(transition)
        assert struct == self._treedef,\
            f'Structures dont match: {struct}\n{self._treedef}'
        for i in range(self._num_leaves):
            self._memory[i][self._idx] = leaves[i]
        self._idx += 1
        self._len = max(self._len, self._idx)
        self._idx %= self.capacity

    def as_generator(self,
                     batch_size: int
                     ) -> typing.Generator[Nested, None, None]:
        while True:
            idx = self._rng.integers(0, self._len, batch_size)
            batch = tree_slice(self._memory, idx)
            yield self._treedef.unflatten(batch)

    def as_tfdataset(self, batch_size: int) -> 'tf.data.Dataset':
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            # Already initialized.
            pass

        output_signature = ReplayBuffer.tile_with(
            self.signature, batch_size, tf.TensorSpec)
        ds = tf.data.Dataset.from_generator(
            lambda: self.as_generator(batch_size),
            output_signature=output_signature
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds.as_numpy_iterator()

    def __len__(self) -> int:
        return self._len

    def save(self, path: str) -> None:
        public = (self.capacity, self.signature)
        private = (self._rng, self._idx, self._len)
        np.savez_compressed(path, *self._memory, public=public, private=private)

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        data = np.load(path, allow_pickle=True)
        rng, idx, len_ = data['private']
        replay = cls(rng, *data['public'])
        replay._memory = [v for k, v in data.items() if k.startswith('arr')]
        replay._idx = idx
        replay._len = len_
        return replay

    @staticmethod
    def tile_with(signature: Nested[SpecsLike],
                  reps: int | Nested[int],
                  constructor: typing.Type[SpecsLike] | None = None
                  ) -> Nested[SpecsLike]:
        if isinstance(reps, int):
            reps = tree_util.tree_map(lambda _: reps, signature)
        else:
            struct = tree_util.tree_structure
            assert struct(signature) == struct(reps)

        def tile_fn(sp, p):
            ctor = constructor or type(sp)
            return ctor(shape=(p,) + sp.shape, dtype=sp.dtype)
        return tree_util.tree_map(tile_fn, signature, reps)


def tree_slice(tree_: Nested[np.ndarray],
               sl: slice,
               is_leaf=None
               ) -> Nested[np.ndarray]:
    return tree_util.tree_map(lambda t: t[sl], tree_, is_leaf=is_leaf)
