import os
import pickle
import time
from typing import NamedTuple

import dm_env

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
from src.replay_buffer import ReplayBuffer, tree_slice
from src import types_ as types


class Runner:

    CONFIG = 'config.yaml'
    REPLAY = 'replay.npz'
    ENCODER = 'encoder.pkl'
    AGENT = 'agent.pkl'

    class Status(NamedTuple):

        path_exists: bool = False
        replay_exists: bool = False
        encoder_exists: bool = False
        agent_exists: bool = False

        @classmethod
        def infer(cls, logdir: str) -> 'Status':
            if not os.path.exists(logdir):
                return cls()
            statuses = map(
                lambda p: os.path.exists(logdir + '/' + p),
                (Runner.REPLAY, Runner.ENCODER, Runner.AGENT)
            )
            return cls(True, *statuses)

    def __init__(self, cfg: CoderConfig) -> None:
        self.cfg = cfg
        self._rngseq = hk.PRNGSequence(jax.random.PRNGKey(cfg.seed))
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
            cfg_path = self._exp_path(Runner.CONFIG)
            cfg.save(cfg_path)

    def run_replay_collection(self, policy: types.Policy) -> None:
        """Prepare expert's demonstrations."""
        env = self.make_env()
        replay = self.make_replay_buffer()
        com = None

        while com != 'fin':
            match input():
                case 'add':
                    trajectory = environment.environment_loop(env, policy)
                    for i in range(len(trajectory['actions'])):
                        replay.add(tree_slice(trajectory, i))
                case 'fin':
                    print(f'Total transitions: {len(replay)}')
                    replay.save(self._exp_path(Runner.REPLAY))
                    return
                case _:
                    continue

    def run_byol(self):
        """Visual pretraining."""
        status = Runner.Status.infer(self._exp_path())
        assert status.replay_exists, 'Nothing to pretrain from.'
        assert not status.encoder_exists, 'Already exists.'

        start = time.time()
        c = self.cfg
        replay = self.make_replay_buffer(Runner.REPLAY)
        ds = replay.as_dataset(c.byol_batch_size)

        networks = self.make_networks()
        params = networks.init(next(self._rngseq))
        optim = optax.adam(c.byol_learning_rate)
        optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)
        state = TrainingState.init(next(self._rngseq),
                                   params,
                                   optim,
                                   c.byol_targets_update)

        step = jax.jit(byol(c, networks))
        logger = TFSummaryLogger(self._exp_path(), 'byol', 'step')

        for t in range(c.byol_steps):
            batch = jax.device_put(next(ds))
            state, metrics = step(state, batch)
            metrics.update(step=t, time=time.time() - start)
            logger.write(metrics)

        with open(self._exp_path(Runner.ENCODER), 'wb') as f:
            params = jax.device_get(state.params)
            pickle.dump(params, f)

    def run_drq(self):
        """RL on top of prefilled buffer and trained visual net."""
        status = Runner.Status.infer(self._exp_path())
        assert status.replay_exists and status.encoder_exists
        assert not status.agent_exists, 'Already exists.'

        start = time.time()
        c = self.cfg
        env = self.make_env()
        networks = self.make_networks(env)
        with open(self._exp_path(Runner.ENCODER), 'rb') as f:
            params = pickle.load(f)
        params = jax.device_put(params)

        replay = self.make_replay_buffer(Runner.REPLAY)
        ds = replay.as_dataset(c.drq_batch_size * c.utd)
        optim = optax.adam(c.drq_learning_rate)
        optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)
        state = TrainingState.init(next(self._rngseq),
                                   params,
                                   optim,
                                   c.drq_targets_update)
        act = jax.jit(networks.act)
        step = jax.jit(drq(c, networks))

        logger = TFSummaryLogger(self._exp_path(), 'drq', step_key='step')
        replay_path = self._exp_path(Runner.REPLAY)
        agent_path = self._exp_path(Runner.AGENT)

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
                metrics.update(step=state.step.item(), time=time.time() - start)
                logger.write(metrics)

            if interactions % c.drq_eval_every == 0:
                def policy(obs_):
                    return act(state.params, next(self._rngseq), obs_, False)
                trajectory = environment.environment_loop(env, policy)
                logger.write({
                    'step': interactions,
                    'eval_return': sum(trajectory['rewards'])
                })

                replay.save(replay_path)
                with open(agent_path, 'wb') as f:
                    pickle.dump(jax.device_get(state.params), f)

    def make_replay_buffer(self, load: str | None = None) -> ReplayBuffer:
        if load is not None:
            return ReplayBuffer.load(load)
        np_rng = next(self._rngseq)[0].item()
        replay = ReplayBuffer(np_rng, self.cfg.buffer_capacity)
        return replay

    def make_env(self) -> dm_env.Environment:
        match self.cfg.task.split('_'):
            case 'dmc', domain, task:
                from dm_control import suite
                from dm_control.suite.wrappers import pixels
                env = suite.load(domain, task)
                render_kwargs = {'camera_id': 0, 'width': 84, 'height': 84}
                env = pixels.Wrapper(
                    env,
                    pixels_only=False,
                    render_kwargs=render_kwargs,
                    observation_key=types.IMG_KEY
                )
            case 'ur', _:
                from ur_env.remote import RemoteEnvClient
                address = ('')
                env = RemoteEnvClient(address)
            case _:
                raise ValueError(self.cfg.task)

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

    def _exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        return os.path.join(logdir, path)
