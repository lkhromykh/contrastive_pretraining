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


def test():
    from src.test_env import Platforms
    env = Platforms(1, 1, 10)
    def policy(obs): return obs['nodes'].argmax(-1)
    return env, policy


def train():
    address = ('', 5555)
    env = RemoteEnvClient(address)
    gamepad = Gamepad('/dev/input/event20')
    def policy(_): return gamepad.read_input()
    return env, policy


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    env, policy = test()
    env = dmc_wrappers.base.Wrapper(env)
    idx = 0
    num_episodes = 40
    preexist = len(os.listdir(DIR))
    while idx < num_episodes:
        tr = environment_loop(env, policy)
        print('Save this? [y/N]')
        if True or input() == 'y':
            path = os.path.join(DIR, f'traj{preexist+idx}')
            with open(path, 'wb') as f:
                pickle.dump(tr, f)
            idx += 1

    with open('specs.pkl', 'wb') as f:
        pickle.dump(env.environment_specs, f)
    env.close()
