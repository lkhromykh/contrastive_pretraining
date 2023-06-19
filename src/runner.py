import os
import pickle
import cloudpickle
import time

import dm_env
import numpy as np

import jax
import optax
import haiku as hk

from rltools.loggers import TFSummaryLogger
from rltools import dmc_wrappers

from src import ops
from src.config import CoderConfig
from src.networks import CoderNetworks
from src.replay_buffer import ReplayBuffer
from src.training_state import TrainingState
from src import types_ as types


class Runner:
    CONFIG = 'config.yaml'
    SPECS = 'specs.pkl'
    DEMO = 'demo.npz'
    ENCODER = 'encoder.cpkl'
    REPLAY = 'replay.npz'
    AGENT = 'agent.cpkl'

    class Status:

        def __init__(self, logdir: str) -> None:
            def exs(p): return os.path.exists(os.path.join(logdir, p))
            self.config_exists = exs(Runner.CONFIG)
            self.specs_exists = exs(Runner.SPECS)
            self.demo_exists = exs(Runner.DEMO)
            self.encoder_exists = exs(Runner.ENCODER)
            self.replay_exists = exs(Runner.REPLAY)
            self.agent_exists = exs(Runner.AGENT)

    def __init__(self, cfg: CoderConfig) -> None:
        self.cfg = cfg
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        cfg_path = self.exp_path(Runner.CONFIG)
        cfg.save(cfg_path)
        # lazy prng, better pass as an argument to the methods
        rng = jax.random.PRNGKey(cfg.seed)
        self._rngseq = hk.PRNGSequence(rng)

    def run_replay_collection(self) -> None:
        """Prepare expert's demonstrations."""
        status = Runner.Status(self.exp_path())
        if status.demo_exists and status.specs_exists:
            print('Demos exist.')
            return
        print('Preparing replay.')
        replay = self.make_replay_buffer()
        idx = 0
        while os.path.exists(path := self.exp_path(f'raw_demos/traj{idx}')):
            trajectory: types.Trajectory = pickle.load(open(path, 'rb'))
            trajectory = ops.nested_stack(trajectory)
            replay.add(trajectory)
            idx += 1
        print('Total episodes: ', idx)
        replay.save(self.exp_path(Runner.DEMO))

    def run_byol(self):
        """Visual pretraining."""
        status = Runner.Status(self.exp_path())
        if status.encoder_exists:
            print('Encoder exists.')
            return
        assert status.specs_exists and status.demo_exists, \
            'Nothing to pretrain from.'

        print('Pretraining.')
        start = time.time()
        c = self.cfg
        replay = self.make_replay_buffer(load=self.exp_path(Runner.DEMO))
        ds = replay.as_tfdataset(c.byol_batch_size)
        networks = self.make_networks()
        params = networks.init(next(self._rngseq))
        optim = optax.adamw(c.byol_learning_rate, weight_decay=c.weight_decay)
        optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)
        state = TrainingState.init(next(self._rngseq),
                                   params,
                                   optim,
                                   c.byol_targets_update)
        step = ops.byol(c, networks)
        if c.jit:
            step = jax.jit(step)
        logger = TFSummaryLogger(self.exp_path(), 'byol', 'step')

        for t in range(c.byol_steps):
            batch = jax.device_put(next(ds))
            state, metrics = step(state, batch)
            metrics.update(step=t, time=time.time() - start)
            logger.write(metrics)

        with open(self.exp_path(Runner.ENCODER), 'wb') as f:
            state = jax.device_get(state)
            cloudpickle.dump(state, f)
        jax.clear_backends()

    def run_drq(self):
        status = Runner.Status(self.exp_path())

        print('Interacting.')
        start = time.time()
        c = self.cfg
        env = self.make_env()
        networks = self.make_networks()

        # Load the most recent weights if any.
        def load(path): return cloudpickle.load(open(self.exp_path(path), 'rb'))
        if status.agent_exists:
            state = load(Runner.AGENT)
        else:
            if status.encoder_exists:
                params = load(Runner.ENCODER).params
            else:
                params = networks.init(next(self._rngseq))
            params = jax.device_put(params)
            optim = optax.adamw(c.drq_learning_rate, weight_decay=c.weight_decay)
            optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)
            state = TrainingState.init(next(self._rngseq),
                                       params,
                                       optim,
                                       c.drq_targets_update)
        state = jax.device_put(state)
        step = ops.drq(c, networks)
        act = networks.act
        if c.jit:
            act = jax.jit(act)
            step = jax.jit(step)

        logger = TFSummaryLogger(self.exp_path(), 'drq', step_key='step')
        replay_path = self.exp_path(Runner.REPLAY)
        agent_path = self.exp_path(Runner.AGENT)

        # Poll for an existing replay buffers.
        if status.replay_exists:
            replay = self.make_replay_buffer(load=replay_path)
        else:
            replay = self.make_replay_buffer()
        if status.demo_exists:
            demo = self.make_replay_buffer(load=self.exp_path(Runner.DEMO))
        else:
            demo = replay

        agent_ds = None
        interactions = len(replay)
        while True:
            traj = ops.environment_loop(env, lambda obs: act(state.params, obs))
            interactions += c.time_limit
            replay.add(traj)
            if len(replay) < c.pretrain_steps:
                continue
            if agent_ds is None:
                half_batch = c.drq_batch_size // 2
                agent_ds = replay.as_tfdataset(half_batch)
                demo_ds = demo.as_tfdataset(half_batch)
                print('Training.')
            for _ in range(c.utd):
                agent_batch = next(agent_ds)
                demo_batch = next(demo_ds)
                batch = jax.tree_util.tree_map(
                    lambda t1, t2: np.concatenate([t1, t2]),
                    agent_batch, demo_batch
                )
                batch = jax.device_put(batch)
                state, metrics = step(state, batch)
            if interactions % c.log_every == 0:
                metrics.update(step=state.step, time=time.time() - start)
                logger.write(metrics)
                replay.save(replay_path)
                with open(agent_path, 'wb') as f:
                    cloudpickle.dump(jax.device_get(state), f)

    def make_specs(self) -> dmc_wrappers.EnvironmentSpecs:
        with open(self.exp_path(Runner.SPECS), 'rb') as f:
            return pickle.load(f)

    def make_replay_buffer(self, *, load: str | None = None) -> ReplayBuffer:
        if load is not None:
            return ReplayBuffer.load(load)
        env_specs = self.make_specs()
        signature = {
            'actions': env_specs.action_spec,
            'rewards': env_specs.reward_spec,
            'discounts': env_specs.discount_spec,
        }
        tl = self.cfg.time_limit

        def time_major(sp, times=tl):
            return np.zeros((times,) + sp.shape, dtype=sp.dtype)
        tree_map = jax.tree_util.tree_map
        signature = tree_map(time_major, signature)
        signature['observations'] = tree_map(
            lambda x: time_major(x, tl + 1),
            env_specs.observation_spec
        )
        np_rng = next(self._rngseq)[0].item()
        return ReplayBuffer(np_rng, self.cfg.replay_capacity, signature)

    def make_env(self) -> dm_env.Environment:
        match self.cfg.task.split('_'):
            case ['test']:
                from src.test_env import Platforms
                env = Platforms(0, self.cfg.time_limit, 5)
            case 'ur', _:
                from ur_env.remote import RemoteEnvClient
                address = None
                env = RemoteEnvClient(address)
            case _:
                raise ValueError(self.cfg.task)

        env = dmc_wrappers.base.Wrapper(env)
        ops.assert_valid_env(env)
        return env

    def make_networks(self) -> CoderNetworks:
        env_specs = self.make_specs()
        return CoderNetworks.make_networks(
            self.cfg,
            env_specs.observation_spec,
            env_specs.action_spec,
        )

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)

    def _load(self, path: str):
        with open(self.exp_path(path), 'rb') as f:
            return pickle.load(f)
