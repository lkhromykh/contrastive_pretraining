from collections import defaultdict

import numpy as np
import dm_env.specs
from jax.tree_util import tree_map

from src import types_ as types


def nested_stack(traj: types.Trajectory) -> types.Trajectory:
    def stack(xs): return tree_map(lambda *x: np.stack(x), *xs)
    traj = tree_map(stack, traj, is_leaf=lambda x: isinstance(x, list))
    return dict(traj)


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
    trajectory['observations'].append(ts.observation)
    return nested_stack(trajectory)


def assert_valid_env(env: dm_env.Environment) -> None:
    """Check compatibility with the networks."""
    spec = env.observation_spec()
    assert types.IMG_KEY in spec, f'Image is missing from the obs spec: {spec}'
    img = spec[types.IMG_KEY]
    assert len(img.shape) == 3 and img.dtype == np.uint8, \
        f'Invalid image spec: {img}'

    act_spec = env.action_spec()
    assert isinstance(act_spec, dm_env.specs.DiscreteArray), \
        f'Not a discrete action space: {act_spec}'

    assert hasattr(env, 'environment_specs'), \
        'rltools.dmc_wrappers like container for the specs is missing.'
