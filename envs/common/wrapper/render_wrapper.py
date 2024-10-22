from typing import Optional
import mujoco
from envs.common.wrapper.torch_wrapper import TorchWrapper
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import jax
import numpy as np
import torch 

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
        self.single_obs_size = self._env.single_observation_size
        self.privileged_observation_size = self._env.privileged_observation_size
        self.action_size = self._env.action_size

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config
        )
        # Visualization context and scene for markers
        self.vopt = mujoco.MjvOption()  # Initialize visualization options
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)  # Scene to hold geoms
        self.cam = mujoco.MjvCamera()  # Camera setup

    def reset(self, initial_xy=jax.numpy.array([0.0,0.0]), manual_cmd=jax.numpy.array([0.0,0.0]), traj = jax.numpy.zeros((500,3))):
        return self._env.reset(initial_xy, manual_cmd, traj)

    def step(self, action, env_id=0):
        obs, privileged_obs, reward, done, info, metrics = self._env.step(action)

        
        if self.render_mode == "human":
            data = self._env.state.pipeline_state
            self.data.qpos = data.qpos[env_id]
            self.data.qvel = data.qvel[env_id]
            mujoco.mj_forward(self.model, self.data)

            #self.add_markers(info, env_id)
            self.render()
            
            
        return obs, privileged_obs,reward, done, info, metrics

    def add_markers(self, state_info, env_id):    
        
        foot_height = state_info['des_foot_height'][env_id]
        foot_pos = state_info['foot_pos'][env_id]


        if self.mujoco_renderer.viewer is not None:
            for j in range(20):
                cmd = state_info['command'][env_id]
                offset = cmd[:2]*0.02*j
                position = torch.cat((foot_pos[:,:2]+offset, torch.unsqueeze(foot_height[:,j],1)), 1)
                position = position.detach().cpu().numpy()
                for i in range(foot_height.shape[0]):
                    size = 0.01
                    if (i == 0) or (i ==2):
                        rgba = np.array([1, 0, 0, 1])
                    else:
                        rgba = np.array([0, 1, 0, 1])
                    print(f"position: {position[i]}")
                    if (j%2 == 0) and (i==0):
                        self.mujoco_renderer.viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,                                               
                                                            pos=position[i],                                               
                                                            size=np.array([size, size, size]),                                               
                                                            rgba=np.array([0, 1, 0, 1])
                                                            )
                                                              

    def render(self):

        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )
