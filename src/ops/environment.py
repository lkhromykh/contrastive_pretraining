from collections import defaultdict, deque

import numpy as np
import dm_env.specs

from src import types_ as types


class FrameStack:
    """Task specific frame stack wrapper."""

    def __init__(self, env: dm_env.Environment, frames_number: int = 3) -> None:
        self.env = env
        self.frames_number = frames_number
        self._deq = None

    def reset(self) -> dm_env.TimeStep:
        ts = self.env.reset()
        img = ts.observation[types.IMG_KEY]
        self._deq = deque(self.frames_number * [img], maxlen=self.frames_number)
        ts.observation[types.IMG_KEY] = np.concatenate(self._deq, -1)
        return ts

    def step(self, action: types.Action) -> dm_env.TimeStep:
        ts = self.env.step(action)
        self._deq.append(ts.observation[types.IMG_KEY])
        ts.observation[types.IMG_KEY] = np.concatenate(self._deq, -1)
        return ts

    def observation_spec(self) -> types.ObservationSpecs:
        spec = self.env.observation_spec().copy()
        img_spec = spec[types.IMG_KEY]
        new_shape = \
            img_spec.shape[:-1] + (self.frames_number * img_spec.shape[-1],)
        spec[types.IMG_KEY] = img_spec.replace(shape=new_shape)
        return spec

    def __getattr__(self, item):
        return getattr(self.env, item)


def environment_loop(env: dm_env.Environment,
                     policy: types.Policy,
                     ) -> types.Trajectory:
    ts = env.reset()
    trajectory = defaultdict(list)
    while not ts.last():
        obs = ts.observation
        action = policy(obs)
        ts = env.step(action)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount)

    trajectory['next_observations'] = \
        trajectory['observations'][1:] + [ts.observation]
    return trajectory


def assert_valid_env(env: dm_env.Environment) -> None:
    """Check compatibility with the networks."""
    spec = env.observation_spec()
    assert types.IMG_KEY in spec, f'Image is missing from obs spec: {spec}'
    img = spec[types.IMG_KEY]
    assert len(img.shape) == 3 and img.dtype == np.uint8, \
        f'Invalid image spec: {img}'

    act_spec = env.action_spec()
    assert isinstance(act_spec, dm_env.specs.BoundedArray), \
        f'Unbounded action space: {act_spec}'
    assert len(act_spec.shape) == 2, f'Discretized space required{act_spec}'

    assert hasattr(env, 'environment_specs'), \
        'rltools.dmc_wrappers provided container for dm_env specs is missing.'
