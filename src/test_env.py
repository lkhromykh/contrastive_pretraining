import numpy as np
import dm_env.specs


class Platforms(dm_env.Environment):
    """Follow the path of greater numbers to receive a reward."""

    OBS_KEY = 'nodes'
    IMG_KEY = 'image'
    IMG_SHAPE = (32, 32, 3)

    def __init__(self,
                 rng: int | np.random.Generator,
                 depth: int,
                 width: int
                 ) -> None:
        self._rng = np.random.default_rng(rng)
        self.depth = depth
        self.width = width

    def reset(self) -> dm_env.TimeStep:
        self._t = 0
        self._tree = self._rng.random((self.depth + 1, self.width), np.float32)
        return dm_env.restart(self._get_obs())

    def step(self, action: int) -> dm_env.TimeStep:
        reward = float(action == self._tree[self._t].argmax())
        self._t += 1
        obs = self._get_obs()
        if self._t == self.depth:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)

    def action_spec(self) -> dm_env.specs.DiscreteArray:
        return dm_env.specs.DiscreteArray(self.width)

    def observation_spec(self) -> dict[str, dm_env.specs.Array]:
        return {
            self.OBS_KEY: dm_env.specs.Array((self.width,), np.float32),
            self.IMG_KEY: dm_env.specs.Array(self.IMG_SHAPE, np.uint8)
                }

    def _get_obs(self):
        return {
            self.OBS_KEY: self._tree[self._t],
            self.IMG_KEY: np.ones(self.IMG_SHAPE, np.uint8)
                }
