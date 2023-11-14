"""Standalone file that is intended for a robot."""
import os
import pickle
from collections import defaultdict

from rltools import dmc_wrappers

DIR = 'raw_demos'
TOTAL_DEMOS = 50


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


def train():
    from ur_env.remote import RemoteEnvClient
    from ur_env.teleop.xboxc import Gamepad
    evkey = Gamepad.EV_KEY
    evabs = Gamepad.EV_ABS
    mapping = {
        evkey.BTN_A: 4,
        evkey.BTN_Y: 5,
        evabs.ABS_HAT0X: lambda v: 1 if v > 0 else 0,
        evabs.ABS_HAT0Y: lambda v: 3 if v > 0 else 2,
        evkey.BTN_TR: 8,
        evkey.BTN_TL: 9,
        evabs.ABS_MISC: lambda v: 6 if v > 0 else 7,
    }
    address = ('', 5555)
    env_ = RemoteEnvClient(address)
    gamepad = Gamepad(mapping, device='/dev/input/event10')
    def policy_(_): return gamepad.read_input()
    return env_, policy_


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    env, policy = train()
    env = dmc_wrappers.base.Wrapper(env)
    idx = len(os.listdir(DIR))
    while idx < TOTAL_DEMOS:
        print('Episode: ', idx)
        tr = environment_loop(env, policy)
        print('Save this? [y/N]')
        if input() == 'y':
            path = os.path.join(DIR, f'traj{idx}')
            with open(path, 'wb') as f:
                pickle.dump(tr, f)
            idx += 1

    with open('specs.pkl', 'wb') as f:
        pickle.dump(env.environment_specs, f)
    env.close()
