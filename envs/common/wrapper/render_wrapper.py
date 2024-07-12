from typing import Optional
import mujoco
from envs.common.wrapper.torch_wrapper import TorchWrapper
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import jax

class RenderWrapper:
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self,
                 env: TorchWrapper,
                 render_mode: Optional[str] = None,
                 width: int = 480,
                 height: int = 480,
                 camera_id: Optional[int] = None,
                 camera_name: Optional[str] = None,
                 default_camera_config: Optional[dict] = None,
                 ):
        """Creates a gym Env to one that outputs PyTorch tensors."""
        self._env = env
        self.model: mujoco.MjModel = env.model
        self.data: mujoco.MjData = env.data
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.render_mode = render_mode
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.observation_size = self._env.observation_size
        self.action_size = self._env.action_size

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config
        )

    def reset(self, initial_xy: jax.Array, manual_control: bool = False):
        return self._env.reset(initial_xy, manual_control)

    def step(self, action):
        obs, priviledged_obs, reward, done, info = self._env.step(action)

        if self.render_mode == "human":
            data = self._env.state.pipeline_state
            self.data.qpos = data.qpos[0]
            self.data.qvel = data.qvel[0]
            mujoco.mj_forward(self.model, self.data)

            self.render()
        return obs, priviledged_obs,reward, done, info

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )
