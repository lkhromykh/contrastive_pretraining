import os
import time
from typing import Any, NamedTuple

import numpy as np
import cloudpickle
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
    RAW_DIR = 'raw_demos/'
    DEMO = 'demo.npz'
    ENCODER = 'encoder.cpkl'
    REPLAY = 'replay.npz'
    AGENT = 'agent.cpkl'

    class Status(NamedTuple):
        config_exists: bool
        specs_exists: bool
        raw_demos_exists: bool
        demo_exists: bool
        encoder_exists: bool
        replay_exists: bool
        agent_exists: bool

    def __init__(self, cfg: CoderConfig) -> None:
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        self.cfg = cfg
        cfg_path = self.exp_path(Runner.CONFIG)
        cfg.save(cfg_path)
        # lazy prng, better pass as an argument.
        rng = jax.random.PRNGKey(cfg.seed)
        self._rngseq = hk.PRNGSequence(rng)

    def run_replay_collection(self) -> None:
        """Prepare expert's demonstrations."""
        status = self.get_status()
        if status.demo_exists:
            print('Demos exist.')
            return
        assert status.specs_exists, 'No specs found.'
        assert status.raw_demos_exists, 'No raw demos found.'
        print('Preparing replay.')
        replay = self.make_replay_buffer()
        counter = 0
        path = self.exp_path(Runner.RAW_DIR)
        for demo in os.scandir(path):
            trajectory: types.Trajectory = self._open(demo)
            trajectory = ops.nested_stack(trajectory)
            replay.add(trajectory)
            counter += 1
        print('Total number of episodes: ', counter)
        replay.save(self.exp_path(Runner.DEMO))

    def run_byol(self):
        """Visual pretraining."""
        status = self.get_status()
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

        state = jax.device_get(state)
        self._write(state, Runner.ENCODER)
        jax.clear_backends()

    def run_drq(self):
        status = self.get_status()

        print('Interacting.')
        start = time.time()
        c = self.cfg
        env = self.make_env()
        networks = self.make_networks()

        # Load the most recent weights if any.
        if status.agent_exists:
            state: TrainingState = self._open(Runner.AGENT)
        else:
            if status.encoder_exists:
                params: hk.Params = self._open(Runner.ENCODER).params
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
        num_episodes = len(replay)
        while True:
            traj = ops.environment_loop(env, lambda obs: act(state.params, obs))
            num_episodes += 1
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
            if num_episodes % c.log_every == 0:
                metrics.update(step=num_episodes * c.time_limit,
                               time=time.time() - start,
                               grad_step=state.step)
                logger.write(metrics)
                replay.save(replay_path)
                self._write(jax.device_get(state), Runner.AGENT)

    def make_specs(self) -> dmc_wrappers.EnvironmentSpecs:
        if os.path.exists(self.exp_path(Runner.SPECS)):
            return self._open(Runner.SPECS)
        return self.make_env().environment_specs

    def make_replay_buffer(self, *, load: str | None = None) -> ReplayBuffer:
        # Here replay stores a whole episode rather than a single transition.
        if load is not None:
            return ReplayBuffer.load(load)
        env_specs = self.make_specs()
        signature = {
            'observations': env_specs.observation_spec,
            'actions': env_specs.action_spec,
            'rewards': env_specs.reward_spec,
            'discounts': env_specs.discount_spec,
        }
        tl = self.cfg.time_limit
        tm = jax.tree_util.tree_map
        signature = tm(lambda sp: sp.generate_value(), signature)
        reps = tm(lambda _: tl, signature)
        reps['observations'] = tm(lambda _: tl + 1, reps['observations'])
        signature = ReplayBuffer.tile_with(signature, reps, np.zeros)
        np_rng = next(self._rngseq)[0].item()
        return ReplayBuffer(np_rng, self.cfg.replay_capacity, signature)

    def make_env(self) -> dmc_wrappers.base.Wrapper:
        match self.cfg.task.split('_'):
            case ['test']:
                from src.test_env import Platforms
                env = Platforms(0, self.cfg.time_limit, 10)
            case 'ur', _:
                from ur_env.remote import RemoteEnvClient
                address = None
                env = RemoteEnvClient(address)
            case _:
                raise ValueError(self.cfg.task)

        env = dmc_wrappers.base.Wrapper(env)
        ops.assert_compliance(env)
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

    def get_status(self) -> 'Runner.Status':
        def exs(p): return os.path.exists(self.exp_path(p))
        return Runner.Status(
            config_exists=exs(Runner.CONFIG),
            specs_exists=exs(Runner.SPECS),
            raw_demos_exists=exs(Runner.RAW_DIR),
            demo_exists=exs(Runner.DEMO),
            encoder_exists=exs(Runner.ENCODER),
            replay_exists=exs(Runner.REPLAY),
            agent_exists=exs(Runner.AGENT),
        )

    def _open(self, path: str) -> Any:
        path = self.exp_path(path)
        with open(path, 'rb') as f:
            obj = cloudpickle.load(f)
        return obj

    def _write(self, obj: Any, path: str) -> None:
        path = self.exp_path(path)
        with open(path, 'wb') as f:
            cloudpickle.dump(obj, f)
