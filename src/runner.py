import os
import pickle
import time
from typing import NamedTuple

import numpy as np

import jax
import haiku as hk
import optax
import rltools.dmc_wrappers.base

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
    SPECS = 'specs.pkl'
    DEMO = 'demo.npz'
    ENCODER = 'encoder.pkl'
    REPLAY = 'replay.npz'
    AGENT = 'agent.pkl'

    class Status(NamedTuple):
        specs_exists: bool = False
        demo_exists: bool = False
        encoder_exists: bool = False
        replay_exists: bool = False
        agent_exists: bool = False

        @classmethod
        def infer(cls, logdir: str) -> 'Status':
            if not os.path.exists(logdir):
                return cls()
            statuses = map(
                lambda p: os.path.exists(logdir + '/' + p),
                (Runner.SPECS, Runner.DEMO, Runner.ENCODER,
                 Runner.REPLAY, Runner.AGENT)
            )
            return cls(*statuses)

    def __init__(self, cfg: CoderConfig) -> None:
        self.cfg = cfg
        self._rngseq = hk.PRNGSequence(jax.random.PRNGKey(cfg.seed))
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        cfg_path = self._exp_path(Runner.CONFIG)
        cfg.save(cfg_path)

    def run_replay_collection(self) -> None:
        """Prepare expert's demonstrations."""
        # Instead of collection we parse existing one.
        status = Runner.Status.infer(self._exp_path())
        if status.demo_exists:
            return

        print('Preparing replay.')
        env_specs = pickle.load(open(self._exp_path(Runner.SPECS), 'rb'))
        act_sp = env_specs.action_spec
        replay = self.make_replay_buffer()

        def action_fn(action):
            """Discretize loaded actions."""
            action = action.astype(np.int32)
            disc = np.zeros(act_sp.shape, act_sp.dtype)
            disc[range(action.size), action] = (action != 0)
            return disc

        idx = 0
        while os.path.exists(path := self._exp_path(f'raw_demos/traj{idx}')):
            trajectory: types.Trajectory = pickle.load(open(path, 'rb'))
            actions = trajectory['actions']
            trajectory['actions'] = list(map(action_fn, actions))

            for i in range(len(actions)):
                replay.add(
                    tree_slice(
                        trajectory, i,
                        is_leaf=lambda t: isinstance(t, list)
                    )
                )
            idx += 1

        print('Total episodes: ', idx)
        print('Total steps: ', len(replay))
        replay.save(self._exp_path(Runner.DEMO))

    def run_byol(self):
        """Visual pretraining."""
        status = Runner.Status.infer(self._exp_path())
        if status.encoder_exists:
            return
        assert status.specs_exists and status.demo_exists, \
            'Nothing to pretrain from.'

        print('Pretraining.')
        start = time.time()
        c = self.cfg
        replay = self.make_replay_buffer(self._exp_path(Runner.DEMO))
        env_specs = pickle.load(open(self._exp_path(Runner.SPECS), 'rb'))
        ds = replay.as_dataset(c.byol_batch_size)

        networks = self.make_networks(env_specs)
        params = networks.init(next(self._rngseq))
        optim = optax.adamw(c.byol_learning_rate, weight_decay=c.weight_decay)
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
            print(metrics)

        with open(self._exp_path(Runner.ENCODER), 'wb') as f:
            params = jax.device_get(state.params)
            pickle.dump(params, f)
        jax.clear_backends()

    def run_drq(self):
        """RL on top of prefilled buffer and trained visual net."""
        status = Runner.Status.infer(self._exp_path())

        print('Interacting.')
        start = time.time()
        c = self.cfg
        env = self.make_env()
        networks = self.make_networks(env.environment_specs)

        # Load most recent weights.
        if status.agent_exists:
            with open(self._exp_path(Runner.AGENT), 'rb') as f:
                params = pickle.load(f)
        elif status.encoder_exists:
            with open(self._exp_path(Runner.ENCODER), 'rb') as f:
                params = pickle.load(f)
        else:
            params = networks.init(next(self._rngseq))
        params = jax.device_put(params)

        optim = optax.adamw(c.drq_learning_rate, weight_decay=c.weight_decay)
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

        # Search for existing replay buffers.
        if status.replay_exists:
            replay = self.make_replay_buffer(replay_path)
        else:
            replay = self.make_replay_buffer()
        if status.demo_exists:
            demo = self.make_replay_buffer(self._exp_path(Runner.DEMO))
        else:
            demo = replay
        half_batch = c.drq_batch_size * c.utd // 2
        agent_ds = demo_ds = None

        ts = env.reset()
        interactions = 0
        while True:
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
            if len(replay) < half_batch:
                continue
            if agent_ds is None:
                agent_ds = replay.as_dataset(half_batch)
                demo_ds = demo.as_dataset(half_batch)
                print('Training.')
            agent_batch = next(agent_ds)
            demo_batch = next(demo_ds)
            batch = jax.tree_util.tree_map(
                lambda t1, t2: np.concatenate([t1, t2]),
                agent_batch, demo_batch
            )
            batch = jax.device_put(batch)
            state, metrics = step(state, batch)
            metrics.update(step=state.step.item(), time=time.time() - start)
            logger.write(metrics)

            if interactions % c.drq_eval_every == 0:
                def policy(obs_):
                    return act(state.params, next(self._rngseq), obs_, False)
                trajectory = environment.environment_loop(env, policy)
                ts = env.reset()
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

    def make_env(self) -> rltools.dmc_wrappers.base.Wrapper:
        match self.cfg.task.split('_'):
            case 'dmc', domain, task:
                from dm_control import suite
                from dm_control.suite.wrappers import pixels
                env = suite.load(domain, task)
                render_kwargs = {'camera_id': 0, 'width': 84, 'height': 84}
                env = pixels.Wrapper(
                    env,
                    pixels_only=True,
                    render_kwargs=render_kwargs,
                    observation_key=types.IMG_KEY
                )
                env = dmc_wrappers.ActionRepeat(env, 2)
                env = environment.FrameStack(env, 3)
            case 'ur', _:
                from ur_env.remote import RemoteEnvClient
                address = ('10.201.2.136', 5555)
                env = RemoteEnvClient(address)
            case _:
                raise ValueError(self.cfg.task)

        env = dmc_wrappers.ActionRescale(env)
        env = dmc_wrappers.DiscreteActionWrapper(env, self.cfg.act_dim_nbins)
        environment.assert_valid_env(env)
        return env

    def make_networks(
            self,
            env_specs: dmc_wrappers.base.EnvironmentSpecs | None = None
    ) -> CoderNetworks:
        env_specs = env_specs or self.make_env().environment_specs
        networks = CoderNetworks.make_networks(
            self.cfg,
            env_specs.observation_spec,
            env_specs.action_spec,
        )
        return networks

    def _exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        return os.path.join(logdir, path)
