from jax import numpy as jp
from envs.robots.unitree_env import UnitreeEnv


class GO2Env(UnitreeEnv):
    def __init__(self, cfg):
        super().__init__(cfg, model_path='unitree_go2/scene.xml')
        # set up robot properties
        self._setup()

    def _setup(self):
        """
        Defines GO2 specific properties, like the foot radius, joint limits, and default position.
        """
        super()._setup()

        self._foot_radius = 0.023
        self.max_z = 0.43
        self.min_z = 0.2

        self.torque_limits = jp.array([23.7, 23.7, 35.55] * 4)

        self.default_pos = jp.array(
            [0, 0, 0.32, 1, 0, 0, 0, # base coord + quat
             -0.1, 0.8, -1.5, #FR
             0.1, 0.8, -1.5,  #FL
             -0.1, 1.0, -1.5, #RR
             0.1, 1.0, -1.5]  #RL
        )

        # Specify Gains for PD controller for each joint
        self.p_gains = jp.array([130., 130., 130.] * 4)
        self.d_gains = jp.array([1., 1., 2.] * 4)

        # position limits
        lower_limits = jp.array([-1.0472, -1.5708, -2.7227]*2 + [-1.0472, -0.5236, -2.7227]*2)
        upper_limits = jp.array([1.0472, 3.4907, -0.83776]*2 + [1.0472, 4.5379, -0.83776]*2)
        m = (lower_limits + upper_limits) / 2
        r = upper_limits - lower_limits
        self.lower_limits = m - 0.5 * r * self.soft_limits
        self.upper_limits = m + 0.5 * r * self.soft_limits
