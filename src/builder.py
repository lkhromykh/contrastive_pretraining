import os

from dm_control import suite
from dm_control.suite.wrappers import pixels

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from rltools.loggers import TFSummaryLogger
from rltools import dmc_wrappers

from src.config import CoderConfig
from src.networks import CoderNetworks
from src.ops import drq, byol, environment
from src.training_state import TrainingState
from src.replay_buffer import ReplayBuffer
from src import types_ as types


class Builder:

    def __init__(self, cfg: CoderConfig) -> None:
        self.cfg = cfg
        self._rngseq = hk.PRNGSequence(jax.random.PRNGKey(cfg.seed))
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
            cfg.save(cfg.logdir + '/config.yaml')

    def run_byol(self):
        c = self.cfg
        np_rng = next(self._rngseq)[0].item()
        replay = ReplayBuffer(np_rng, c.buffer_capacity)
        replay.load(c.logdir + '/offline_buffer.npz')
        gen = replay.as_generator(c.byol_batch_size)

        optim = optax.adam(c.byol_learning_rate)
        state = TrainingState
        raise NotImplementedError

    def run_drq(self):
        c = self.cfg
        env = self.make_env()
        networks = self.make_networks(env)
        params = networks.init(next(self._rngseq))

        replay = self.make_replay_buffer()
        ds = replay.as_generator(c.drq_batch_size * c.utd)
        optim = optax.adam(c.drq_learning_rate)
        state = TrainingState.init(next(self._rngseq),
                                   params,
                                   optim,
                                   c.drq_targets_update)
        act = jax.jit(networks.act)
        step = jax.jit(drq(c, networks))
        logger = TFSummaryLogger(c.logdir, 'train', step_key='step')

        ts = env.reset()
        interactions = 0
        while interactions != 1e7:
            if ts.last():
                ts = env.reset()
            obs = ts.observation
            action = act(state.params, next(self._rngseq), obs, True)
            ts = env.step(action)
            interactions += 1
            replay.add({
                'observations': obs,
                'actions': action,
                'rewards': ts.reward,
                'discounts': ts.discount,
                'next_observations': ts.observation
            })
            if len(replay) > 2e3:
                batch = jax.device_put(next(ds))
                state, metrics = step(state, batch)
                metrics.update(step=state.step.item())
                logger.write(metrics)

            if interactions % 10000 == 0:
                def policy(obs_):
                    return act(state.params, next(self._rngseq), obs_, False)
                trajectory = environment.environment_loop(env, policy)
                print(f"Eval reward {sum(trajectory['rewards'])} on step {interactions}")

    def make_replay_buffer(self, load=None):
        np_rng = next(self._rngseq)[0].item()
        replay = ReplayBuffer(np_rng, self.cfg.buffer_capacity)
        if load is not None:
            replay.load(self.cfg.logdir + load)
        return replay

    def make_env(self):
        env = suite.load('cartpole', 'balance')
        env = pixels.Wrapper(env,
                             pixels_only=False,
                             render_kwargs={'width': 84, 'height': 84},
                             observation_key=types.IMG_KEY
                             )
        env = dmc_wrappers.ActionRescale(env)
        env = dmc_wrappers.DiscreteActionWrapper(env, 11)
        environment.assert_valid_env(env)
        return env

    def make_networks(self, env=None):
        env = env or self.make_env()
        networks = CoderNetworks.make_networks(
            self.cfg,
            env.observation_spec(),
            env.action_spec()
        )
        return networks
