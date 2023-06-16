"""Standalone file that is intended for a robot."""
import os
import pickle
from collections import defaultdict

from ur_env.remote import RemoteEnvClient
from ur_env.teleop import Gamepad
from rltools import dmc_wrappers

DIR = 'raw_demos'


def environment_loop(env, policy):
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
    return trajectory


if __name__ == '__main__':
    # address = ('', 5555)
    # env = RemoteEnvClient(address)
    # gamepad = Gamepad('/dev/input/event20')
    # def policy(_): return gamepad.read_input()
    from src.test_env import Platforms
    env = Platforms(0, 5, 5)
    env = dmc_wrappers.base.Wrapper(env)
    def policy(_): return int(input())

    idx = len(os.listdir(DIR))
    com = 1
    while com:
        tr = environment_loop(env, policy)
        print('Save this [y/N]?')
        if input() == 'y':
            path = os.path.join(DIR, f'traj{idx}')
            assert not os.path.exists(path)
            with open(path, 'wb') as f:
                pickle.dump(tr, f)
            idx += 1
        print('Continue?')
        com = input()

    with open('specs.pkl', 'wb') as f:
        pickle.dump(env.environment_specs, f)
    env.close()
