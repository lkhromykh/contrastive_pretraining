from enum import IntEnum
from collections import OrderedDict

import numpy as np
from dm_env import specs

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.manipulation.shared import workspaces


MODEL_XML = """
<mujoco model="particle">

    <option gravity="0 0 0" timestep=".05"/>

    <worldbody>
        <light name="light" pos = "0 0 2"
               directional="true" castshadow="false"/>
        <geom name="floor" size = "3 3 .2" type="plane" rgba="1 1 1 1"/>
        <camera name="fixed" pos="0 0 2" zaxis="0 0 1"/>
        <body name="particle" pos="0 0 .1" mocap="true">
            <geom name="particle" type="sphere"
                  size=".03" mass="1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

SCENE_LIM = .2
CTRL_LIMIT = .05
THRESHOLD = 2 * CTRL_LIMIT
HEIGHT_OFFSET = .1
_WIDTH, _HEIGHT = IMG_SIZE = (84, 84)

DEFAULT_BBOX = workspaces.BoundingBox(
    lower=np.array([-SCENE_LIM, -SCENE_LIM, HEIGHT_OFFSET]),
    upper=np.array([SCENE_LIM, SCENE_LIM, HEIGHT_OFFSET])
)


class Particle(composer.Entity):

    def _build(self):
        self._mjcf_model = mjcf.from_xml_string(MODEL_XML)
        self.particle = self.mjcf_model.find('body', 'particle')
        self.camera = self.mjcf_model.find('camera', 'fixed')

    def _build_observables(self):
        return ParticleObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model


class ParticleObservables(composer.Observables):

    @composer.observable
    def pos(self):
        return observable.MJCFFeature('xpos', self._entity.particle)


class DiscreteActions(IntEnum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    @staticmethod
    def as_array(action: int, dtype=None) -> np.ndarray:
        idx, val = np.divmod(action, 2)
        np_action = np.zeros(len(DiscreteActions) // 2, dtype=dtype)
        np_action[idx] = -1 if val else 1
        return np_action


class ParticleReach(composer.Task):

    def __init__(self,
                 scene_bbox: workspaces.BoundingBox = DEFAULT_BBOX,
                 ) -> None:
        self._root_entity = Particle()
        self._particle = self._root_entity.particle

        self._bbox = scene_bbox
        self._target_site = workspaces.add_target_site(
            self.root_entity.mjcf_model.worldbody,
            radius=THRESHOLD,
            visible=True, rgba="0 0 1 0.3",
            name="target_site"
        )
        workspaces.add_bbox_site(
            self.root_entity.mjcf_model.worldbody,
            lower=scene_bbox.lower,
            upper=scene_bbox.upper,
            visible=False
        )
        self._rng_fn = lambda rng: rng.uniform(
            scene_bbox.lower, scene_bbox.upper)
        self._task_observables = OrderedDict()

        def goal_pos(physics):
            return physics.bind(self._target_site).xpos

        # self._task_observables['goal_pos'] = observable.Generic(goal_pos)
        self._task_observables['image'] = observable.MJCFCamera(
            self._root_entity.camera, width=_WIDTH, height=_HEIGHT)

        self.root_entity.observables.enable_all()
        for obs in self._task_observables.values():
            obs.enabled = True

    def initialize_episode(self, physics, random_state):
        # Sample new goal.
        target_pos = self._rng_fn(random_state)
        physics.bind(self._target_site).pos = target_pos
        # Sample initial pos.
        self._set_pos(physics, self._rng_fn(random_state))
        physics.forward()

    def before_step(self, physics, action, random_state):
        del random_state
        pos = self._get_pos(physics)
        action = DiscreteActions.as_array(action)
        pos += np.concatenate([CTRL_LIMIT*action, [0]])
        self._set_pos(physics, pos)

    def get_reward(self, physics):
        pos = self._get_pos(physics)
        target_site = physics.bind(self._target_site)
        dist = np.linalg.norm(target_site.xpos - pos)
        return float(dist < THRESHOLD)

    def action_spec(self, physics):
        return specs.DiscreteArray(len(DiscreteActions), np.int32)

    def _set_pos(self, physics, pos):
        particle = physics.bind(self._particle)
        particle.mocap_pos = np.clip(
            pos,
            a_min=self._bbox.lower,
            a_max=self._bbox.upper
        )

    def _get_pos(self, physics):
        particle = physics.bind(self._particle)
        return particle.mocap_pos

    @property
    def root_entity(self):
        return self._root_entity

    @property
    def task_observables(self):
        return self._task_observables


class ParticleEnv(composer.Environment):
    def __init__(self,
                 task=ParticleReach(),
                 time_limit=float('inf'),
                 random_state=None,
                 **kwargs
                 ):
        super().__init__(task, time_limit, random_state,
                         strip_singleton_obs_buffer_dim=True, **kwargs)
