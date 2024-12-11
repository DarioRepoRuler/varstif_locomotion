from typing import Optional
import mujoco
from envs.common.wrapper.torch_wrapper import TorchWrapper
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import jax
import numpy as np
import torch 
import os
import imageio
from typing import Optional, List

class RenderWrapper:
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self,
                 env: TorchWrapper,
                 render_mode: Optional[str] = None,
                 width: int = 640,
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
        
        self.recording = False
        self.recorded_frames: List[np.ndarray] = []
        self.video_path = None
        self.fps = 30
        from threading import Lock
        self.render_lock = Lock()
    
    def start_video_recording(self, video_path: str, fps: int = 30):
        """
        Start recording a video.

        Args:
            video_path (str): Path to save the video file.
            fps (int): Frames per second for the video (default: 30).
        """
        self.recording = True
        self.recorded_frames = []
        self.video_path = video_path
        self.fps = fps
        print(f"Started video recording to {video_path} at {fps} FPS.")

    def stop_video_recording(self):
        """
        Stops recording and saves the video.
        """
        if not self.recording:
            print("No active recording to stop.")
            return

        if not self.video_path:
            print("Error: No valid video path specified. Please call `start_video_recording` first.")
            return

        # Ensure the directory for the video exists
        video_dir = os.path.dirname(self.video_path)
        print(f"Saving video to {self.video_path}")
        if video_dir:  # Only create directories if a path exists
            os.makedirs(video_dir, exist_ok=True)
        # Save the video using imageio
        imageio.mimwrite(self.video_path, self.recorded_frames, fps=self.fps, format="mp4")
        print(f"Video saved to {self.video_path}")

        # Reset recording state
        self.recording = False
        self.recorded_frames = []
        
    def reset(self, initial_xy=jax.numpy.array([0.0,0.0]), manual_cmd=jax.numpy.array([0.0,0.0]), traj = jax.numpy.zeros((500,3))):
        return self._env.reset(initial_xy, manual_cmd, traj)

    def step(self, action, env_id=0):
        obs, privileged_obs, reward, done, info, metrics = self._env.step(action)        
        if self.render_mode == "rgb_array":
            data = self._env.state.pipeline_state
            self.data.qpos = data.qpos[env_id]
            self.data.qvel = data.qvel[env_id]
            mujoco.mj_forward(self.model, self.data)

            #self.add_markers(info, env_id)
            frame = self.render()
            #print(f"Frame: {frame}")
            # If recording, save the frame
            if self.recording and frame is not None:
                self.recorded_frames.append(frame)
                
        elif self.render_mode == "human":
            data = self._env.state.pipeline_state
            self.data.qpos = data.qpos[env_id]
            self.data.qvel = data.qvel[env_id]
            mujoco.mj_forward(self.model, self.data)

            #self.add_height_marker(info, env_id)
            #self.add_height_markers(info, env_id)
            self.render()
            
            
        return obs, privileged_obs,reward, done, info, metrics    

    def add_height_markers(self, state_info, env_id):    
        # Retrieve relevant information
        height_scan = state_info['height_scan1'][env_id].detach().cpu().numpy()  # Height scan data
        # foot_pos = state_info['foot_pos'][env_id].cpu().numpy()  # Position of the foot

        size = 0.01
        if self.mujoco_renderer.viewer is not None and state_info['step'][env_id].detach().cpu().numpy()>=2:
            self.mujoco_renderer.viewer._markers.clear()
            print(f"Height scan from env: {height_scan}")
            for i in range(height_scan.shape[0]):
                print(f"Height scan: {height_scan[i]}")
                self.mujoco_renderer.viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,                                               
                                                                pos=np.array(height_scan[i]),                                               
                                                                size=np.array([size, size, size]),                                               
                                                                rgba=np.array([0, 1, 0, 1])
                                                                )
    
    
    def add_height_marker(self, state_info, env_id):    
        # Retrieve relevant information
        height_scan = state_info['height_scan'][env_id].detach().cpu().numpy()  # Height scan data
        print(f"Height scan from env: {height_scan}")
        # foot_pos = state_info['foot_pos'][env_id].cpu().numpy()  # Position of the foot

        size = 0.01
        if self.mujoco_renderer.viewer is not None:
            self.mujoco_renderer.viewer._markers.clear()
            print(f"Height scan from env: {height_scan}")
            self.mujoco_renderer.viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,                                               
                                                                pos=np.array(height_scan),                                               
                                                                size=np.array([size, size, size]),                                               
                                                                rgba=np.array([0, 1, 0, 1])
                                                                )
        
        
        # if self.mujoco_renderer.viewer is not None:
        #     for j in range(20):
        #         cmd = state_info['command'][env_id]
        #         offset = cmd[:2]*0.02*j
        #         position = torch.cat((foot_pos[:,:2]+offset, torch.unsqueeze(foot_height[:,j],1)), 1)
        #         position = position.detach().cpu().numpy()
        #         for i in range(foot_height.shape[0]):
        #             size = 0.01
        #             if (i == 0) or (i ==2):
        #                 rgba = np.array([1, 0, 0, 1])
        #             else:
        #                 rgba = np.array([0, 1, 0, 1])
        #             print(f"position: {position[i]}")
        #             if (j%2 == 0) and (i==0):
        #                 self.mujoco_renderer.viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,                                               
        #                                                     pos=position[i],                                               
        #                                                     size=np.array([size, size, size]),                                               
        #                                                     rgba=np.array([0, 1, 0, 1])
        #                                                     )
    
    def add_force_marker(self, state_info, env_id):
        kick_force = state_info['force_kick'][env_id].cpu().numpy()    # Force vector (XY direction)
        robot_pos = state_info['global_pos'][env_id].cpu().numpy()  # Position of the robot
        kick_magnitude = state_info['kick_force_magnitude'][env_id].cpu().numpy()  # Magnitude of the force
        
        kick_force = np.array([kick_force[0], kick_force[1], 0.0])
        norm = np.linalg.norm(kick_force)
        # Ensure the renderer viewer is available
        if (self.mujoco_renderer.viewer is not None) and norm > 1e-6:
            # Arrow base position
            arrow_start = np.array(robot_pos)

            if norm > 1e-6:  # Avoid division by zero
                kick_force /= norm  # Normalize the vector
                print(f"Direction XY: {kick_force}, norm: {norm}")
            else:
                self.mujoco_renderer.viewer._markers.clear()
                return 
                
            # Compute a rotation matrix (orientation) for the arrow
            z_axis = kick_force  # The arrow points along this direction (Z-axis)
            x_axis = np.array([-z_axis[1], z_axis[0], 0])  # Perpendicular in XY plane
            if np.linalg.norm(x_axis) < 1e-6:  # If x_axis is degenerate, set a default
                x_axis = np.array([1.0, 0.0, 0.0])
            x_axis /= np.linalg.norm(x_axis)  # Normalize 

            y_axis = np.cross(z_axis, x_axis)  # Ensure orthogonality
            mat = np.stack([x_axis, y_axis, z_axis], axis=1)  # Rotation matrix

            assert not np.any(np.isinf(arrow_start)), "Marker position contains inf!"
            assert not np.any(np.isnan(arrow_start)), "Marker position contains NaN!"
            assert not np.any(np.isnan(mat)), "Orientation matrix contains NaN!"
            assert not np.any(np.isinf(mat)), "Marker position contains inf!"   
            assert mat.shape == (3, 3), "Orientation matrix must be 3x3!"
                       
            # Add the arrow marker
            self.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_ARROW,  # Arrow marker type
                pos=arrow_start,                  # Base position of the arrow
                size=np.array([0.02, 0.02, 1.0]),  # Size: width, height, length
                mat=mat,                          # Orientation matrix
                rgba=np.array([0.0, 1.0, 0.0, 1.0])  # RGBA color (green)
            )                                                     

    def render(self):
        """
        Renders the environment and returns an RGB array for video recording.
        """
        with self.render_lock:
            if self.render_mode in ["rgb_array", "rgb_array_list"]:
                return self.mujoco_renderer.render(
                    render_mode="rgb_array",  # Explicitly request an RGB frame
                    camera_name="tracking",
                )
            
            elif self.render_mode == "human":
                # Render to the screen, does not return a frame
                return self.mujoco_renderer.render(
                    render_mode=self.render_mode,
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                )
                
            else:
                raise ValueError(f"Unsupported render mode: {self.render_mode}")
    
