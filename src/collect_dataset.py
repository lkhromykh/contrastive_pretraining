"""Standalone file that is intended for a robot."""
import os
import pickle
from collections import defaultdict

from ur_env.remote import RemoteEnvClient
from ur_env.teleop import Gamepad
from rltools import dmc_wrappers

address = ('', 5555)
env = RemoteEnvClient(address)
env = dmc_wrappers.ActionRescale(env)
gamepad = Gamepad('/dev/input/event20')


def environment_loop(env_, policy):
    ts = env_.reset()
    trajectory = defaultdict(list)
    while not ts.last():
        obs = ts.observation
        action = policy(obs)
        ts = env_.step(action)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount)

    trajectory['next_observations'] = \
        trajectory['observations'][1:] + [ts.observation]
    return trajectory


idx = 0
while True:
    com = input()
    if com == 'add':
        tr = environment_loop(env, lambda _: gamepad.read_input())
        print('Save this [y/N]?')
        if input() == 'y':
            path = f'traj{idx}'
            assert not os.path.exists(path)
            with open(path, 'wb') as f:
                pickle.dump(tr, f)
            idx += 1
    if com == 'break':
        break

env = dmc_wrappers.DiscreteActionWrapper(env, 11)
with open('specs.pkl', 'wb') as f:
    pickle.dump(env.environment_specs, f)
env.close()
