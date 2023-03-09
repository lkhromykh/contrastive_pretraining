import os
import pickle

import dm_env
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

    CONFIG = 'config.yaml'
    REPLAY = 'replay.npz'
    ENCODER = 'encoder.pkl'
    AGENT = 'agent.pkl'

    def __init__(self, cfg: CoderConfig) -> None:
        self.cfg = cfg
        self._rngseq = hk.PRNGSequence(jax.random.PRNGKey(cfg.seed))
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
            cfg_path = self._exp_path(Builder.CONFIG)
            cfg.save(cfg_path)

    def run_replay_collection(self):
        """Prepare expert's demonstrations."""

    def run_byol(self):
        """Visual pretraining."""
        c = self.cfg
        np_rng = next(self._rngseq)[0].item()
        replay = ReplayBuffer(np_rng, c.buffer_capacity)
        replay.load(c.logdir + '/offline_buffer.npz')
        gen = replay.as_generator(c.byol_batch_size)

        optim = optax.adam(c.byol_learning_rate)
        state = TrainingState
        raise NotImplementedError

    def run_drq(self):
        """RL on top of prefilled buffer and trained visual net."""
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
        replay_path = self._exp_path(Builder.REPLAY)
        agent_path = self._exp_path(Builder.AGENT)

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

            if interactions % 20000 == 0:
                def policy(obs_):
                    return act(state.params, next(self._rngseq), obs_, False)
                trajectory = environment.environment_loop(env, policy)
                print(f'Eval reward {sum(trajectory["rewards"])}'
                      f'on step {interactions}')

                replay.save(replay_path)
                with open(agent_path, 'wb') as f:
                    # TODO: avoid saving tx but save a whole state.
                    pickle.dump(jax.device_get(state.params), f)

    def make_replay_buffer(self, load: str | None = None) -> ReplayBuffer:
        np_rng = next(self._rngseq)[0].item()
        replay = ReplayBuffer(np_rng, self.cfg.buffer_capacity)
        if load is not None:
            replay.load(self._exp_path(load))
        return replay

    def make_env(self) -> dm_env.Environment:
        env = suite.load('walker', 'walk')
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs={'width': 84, 'height': 84},
                             observation_key=types.IMG_KEY
                             )
        env = dmc_wrappers.ActionRepeat(env, 2)
        env = dmc_wrappers.ActionRescale(env)
        env = dmc_wrappers.DiscreteActionWrapper(env, 11)
        environment.assert_valid_env(env)
        return env

    def make_networks(self,
                      env: dm_env.Environment | None = None
                      ) -> CoderNetworks:
        env = env or self.make_env()
        networks = CoderNetworks.make_networks(
            self.cfg,
            env.observation_spec(),
            env.action_spec()
        )
        return networks

    def _exp_path(self, path: str) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        return os.path.join(logdir, path)
