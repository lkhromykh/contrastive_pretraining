from collections import defaultdict

import numpy as np
import dm_env.specs

from src import types_ as types


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
