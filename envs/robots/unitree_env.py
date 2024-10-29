import jax
import mujoco
from brax import math
from brax.base import Transform, Motion
from typing import Any, Dict, Tuple, Union
from jax import numpy as jp
from jax import config
from mujoco import mjx
from mujoco.mjx._src.forward import _integrate_pos
from mujoco.mjx._src.support import jac
from mujoco.mjx._src import smooth
from envs.common.mjx_env import MjxEnv, State
from envs.common.helper import unscale
from pathlib import Path
import os
import numpy as np


# config.update("jax_debug_nans", True)
# config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)


class UnitreeEnv(MjxEnv):
    def __init__(
            self,
            cfg,
            model_path,
            soft_limits=0.9,
    ):
        
        resource_directory = os.path.join(os.getcwd(), 'envs','resources') # it is anticipated to be executed from TALocoMotion
        mj_model = mujoco.MjModel.from_xml_path(os.path.join(resource_directory, model_path))
        if "terrain" in model_path:
            self.terminate_map = True
            print(f"Changing termination to MAP TERMINATION: {self.terminate_map}")
        else:
            print(f"DEFAULT TERMINATION")
            self.terminate_map = False

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.jacobian = 1        # 'sparse'

        super().__init__(mj_model=mj_model,
                         physics_steps_per_control_step=cfg.physics_steps_per_control_step)
        print(self.sys.opt)
        
        # Overall parameters
        self.control_mode = cfg.control_mode
        self.action_scale = cfg.action_scale
        self.action_stiff_scale = cfg.action_stiff_scale
        self.num_history_actor = cfg.num_history_actor
        self.num_history_critic = cfg.num_history_critic
        self._kick_vel = cfg.kick_vel
        self.is_training = cfg.is_training
        self.push_interval = cfg.push_interval
        self.episode_length = cfg.episode_length
        self.sample_command_interval=cfg.sample_command_interval

        # Randomisation parameters
        self.randomize = cfg.domain_rand.randomisation
        self.local_v_noise = cfg.domain_rand.local_v_noise
        self.local_w_noise = cfg.domain_rand.local_w_noise
        self.joint_noise = cfg.domain_rand.joint_noise
        self.joint_vel_noise = cfg.domain_rand.joint_vel_noise
        self.gravity_noise = cfg.domain_rand.gravity_noise


        # Randomization ranges:
        self.x_pos = cfg.reset_pos_x
        self.y_pos = cfg.reset_pos_y 
        self.theta = [0, jp.pi/8] # in rad
        self.a_x = [-1,1]
        self.a_y = [-1,1]
        self.kp_range = cfg.domain_rand.kp_range
        self.kd_range = cfg.domain_rand.kd_range
        self.motor_strength_range = cfg.domain_rand.motor_strength_range

        # Control parameters
        self.manual_control = cfg.manual_control.enable
        self.track_traj = (cfg.manual_control.task == "track trajectory")
        self.cmd_x = cfg.manual_control.cmd_x
        self.cmd_y = cfg.manual_control.cmd_y
        self.cmd_yaw = cfg.manual_control.cmd_ang

        self.soft_limits = soft_limits
        self.single_obs_size = cfg.single_obs_size # defined in _get_obs
        self.privileged_obs_size = cfg.single_obs_size_priv

        # Kick parameters
        self.enable_force_kick = cfg.enable_force_kick
        self.kick_force = cfg.kick_force
        self.force_kick_duration = cfg.force_kick_duration
        self.force_kick_interval = cfg.force_kick_interval
        self.force_kick_counter = self.force_kick_duration / self.dt
        self.force_kick_impulse = cfg.force_kick_impulse
        self.impulse_force_kick = cfg.impulse_force_kick
        
        # Variable impedance control parameters
        self.stiff_range = cfg.control.stiff_range  
        if cfg.control_mode == "VIC_1": # for hip,thigh and knee
            self.action_shape = self.action_size-2 + 3
            self.single_obs_size = self.single_obs_size + 3
            self.privileged_obs_size = self.privileged_obs_size +3
        elif cfg.control_mode == "VIC_2": # for every leg
            self.action_shape = self.action_size-2 + 4
            self.single_obs_size = self.single_obs_size + 4
            self.privileged_obs_size = self.privileged_obs_size + 4 
        elif cfg.control_mode == "VIC_3": # for every leg
            self.action_shape = self.action_size-2 + 12
            self.single_obs_size = self.single_obs_size + 12
            self.privileged_obs_size = self.privileged_obs_size + 12
        elif cfg.control_mode == "VIC_4":
            self.action_shape = self.action_size-2 + 7
            self.single_obs_size = self.single_obs_size + 7
            self.privileged_obs_size = self.privileged_obs_size + 7
        else:
            self.action_shape = self.action_size-2
        
        # Normalization ranges
        self.local_v_scale = cfg.normalization.local_v_scale
        self.local_w_scale = cfg.normalization.local_w_scale
        self.joint_vel_scale = cfg.normalization.joint_vel_scale
        self.command_scale = cfg.normalization.command_scale

        # Specify Gains for PD controller for each joint
        self.p_gain = cfg.control.p_gain
        self.d_gain = cfg.control.d_gain

        self.control_range = cfg.control_range
        self.reward_scales = cfg.reward_scales
        self.min_z = 0.15
        # set up robot properties
        self._setup()

    def _setup(self):

        self._foot_radius = 0.023
        self._torso_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk')
        
        # Define body and geometry names
        feet_names = ['FR', 'FL', 'RR', 'RL'] 
        hip_body = ['FR_hip', 'FL_hip', 'RR_hip', 'RL_hip']
        torso_bodies = ['base_mirror_0', 'base_mirror_1', 'base_mirror_2', 'base_mirror_3', 'base_mirror_4']
        # body_geometries = [
        #     "base_0", "base_1", "base_2", 
        #     "FR_hip", "FR_thigh", "FR_calf_0", "FR_calf_1",  
        #     "FL_hip", "FL_thigh", "FL_calf_0", "FL_calf_1", 
        #     "RR_hip", "RR_thigh", "RR_calf_0", "RR_calf_1", 
        #     "RL_hip", "RL_thigh", "RL_calf_0", "RL_calf_1"
        #     ]
        body_geometries = [
            "base_0", "base_1", "base_2", 
            "FR_hip", "FR_thigh",   
            "FL_hip", "FL_thigh", 
            "RR_hip", "RR_thigh", 
            "RL_hip", "RL_thigh"
            ]
        terminate_geometries = ["base_0", "base_1", "base_2","FR_hip","FL_hip","RR_hip","RL_hip"]
        foot_body = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        lower_leg_body=['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']
        
        # Convert to IDs
        feet_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_names
        ]
        feet__geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in feet_names
        ]
        foot_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in foot_body
        ]
        body_geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in body_geometries
        ]
        terminate_geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in terminate_geometries
        ]
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, 'floor')
        hip_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in hip_body
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]

        # Assert if Bodies and Geometries are not found
        assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        
        assert not any(id_ == -1 for id_ in body_geom_id), 'Body Geom not found.'
        self.body_geom_id = jp.array(body_geom_id)

        assert not any(id_ == -1 for id_ in body_geom_id), 'Terminate Geom not found.'
        self.terminate_geom_id = jp.array(terminate_geom_id)
        
        assert not any(id_ == -1 for id_ in feet_site_id), 'Feet Site not found.'
        self.feet_site_id = jp.array(feet_site_id)

        assert not any(id_ == -1 for id_ in feet__geom_id), 'Feet Geom not found.'
        self.feet_geom_id= jp.array(feet__geom_id)

              
        assert not any(id_ == -1 for id_ in hip_body), 'Hip Body not found.'
        self.hip_body_id = jp.array(hip_body_id)

        assert not any(id_ == -1 for id_ in foot_body), 'Foot Body not found.'
        self.foot_body_id = jp.array(foot_body_id)
        
        self.floor_id = floor_id  
        assert floor_id != -1, 'Floor not found.'
        self.floor_id = floor_id


    def get_foot_contacts(self, data)->jax.Array: # should be returned in the order of FR, FL, RR, RL
        """
        Returns the connection indices of foot contacts with the ground.

        Args:
            data: The data containing contact information.

        Returns:
            conn_indices: An array of connection indices representing foot contacts with the ground.
        """
        # Number of collisions and arrays are set statically due to brax troubles        
        num_collisions = 4
        expected_total_cont=23 + 3*23*self.terminate_map
        geom_temp = jp.zeros((expected_total_cont,2))
        conn_indices = jp.zeros((num_collisions,1), dtype=int)
        #jax.debug.print('Shape of contacts:  {x}', x=data.contact.geom.shape)
        geom_temp = geom_temp.at[0:expected_total_cont,0:2].set(data.contact.geom[0:expected_total_cont,0:2])
        feet_mask = jp.zeros((expected_total_cont),dtype=int)
        foot_cont = jp.zeros((expected_total_cont),dtype=int)
        connection_index = jp.zeros((1),dtype=int)
        ground_mask = jp.zeros((expected_total_cont),dtype=int)

        ground_mask = ground_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, 0)[:,0])
        
        for i in range(num_collisions):
            feet_mask=feet_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, self.feet_geom_id[i])[:,1])
            foot_cont = foot_cont.at[0:expected_total_cont].set(feet_mask*ground_mask)
            connection_index= connection_index.at[:].set(jp.where(foot_cont, size=1)[0])
            conn_indices = conn_indices.at[i,0].set(connection_index[0]) 
        #jax.debug.print('Foot contacts: {x}', x=conn_indices)       
        return conn_indices

    def get_body_contacts(self, data)->jax.Array:
        """
        Returns the connection indices of body contacts with the ground.

        Args:
            data: The data containing contact information.

        Returns:
            conn_indices: An array of connection indices representing body contacts with the ground.
        """
        # Number of collisions and arrays are set statically due to brax troubles
        num_collisions = len(self.body_geom_id)
        expected_total_cont=23 + 3*23*self.terminate_map
        geom_temp = jp.zeros((expected_total_cont,2))
        conn_indices = jp.zeros((num_collisions,1), dtype=int)
        geom_temp = geom_temp.at[0:expected_total_cont,0:2].set(data.contact.geom[0:expected_total_cont,0:2])
        body_mask = jp.zeros((expected_total_cont),dtype=int)
        body_cont = jp.zeros((expected_total_cont),dtype=int)
        connection_index = jp.zeros((1),dtype=int)
        ground_mask = jp.zeros((expected_total_cont),dtype=int)

        ground_mask = ground_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, 0)[:,0])
        
        for i in range(num_collisions):
            body_mask=body_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, self.body_geom_id[i])[:,1])
            body_cont = body_cont.at[0:expected_total_cont].set(body_mask*ground_mask)
            connection_index= connection_index.at[:].set(jp.where(body_cont, size=1)[0])
            #jax.debug.print('Connection index: {x}', x=connection_index)
            conn_indices = conn_indices.at[i,0].set(connection_index[0])        
        return conn_indices
    
    def get_terminate_contacts(self, data)->jax.Array: 
        # Number of collisions and arrays are set statically due to brax troubles
        num_collisions = 7
        expected_total_cont=23 + 3*23*self.terminate_map
        geom_temp = jp.zeros((expected_total_cont,2))
        conn_indices = jp.zeros((num_collisions,1), dtype=int)
        geom_temp = geom_temp.at[0:expected_total_cont,0:2].set(data.contact.geom[0:expected_total_cont,0:2])
        body_mask = jp.zeros((expected_total_cont),dtype=int)
        body_cont = jp.zeros((expected_total_cont),dtype=int)
        connection_index = jp.zeros((1),dtype=int)
        ground_mask = jp.zeros((expected_total_cont),dtype=int)

        ground_mask = ground_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, 0)[:,0])
        
        for i in range(num_collisions):
            body_mask=body_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, self.terminate_geom_id[i])[:,1])
            body_cont = body_cont.at[0:expected_total_cont].set(body_mask*ground_mask)
            connection_index= connection_index.at[:].set(jp.where(body_cont, size=1)[0])
            #jax.debug.print('Connection index: {x}', x=connection_index)
            conn_indices = conn_indices.at[i,0].set(connection_index[0])        
        return conn_indices
    
    def _resample_commands(self, rng: jax.Array) -> jax.Array:
        # Define constraints for the commands# From turtoial
        lin_vel_x = self.control_range['cmd_x'] # min max [m/s]
        lin_vel_y = self.control_range['cmd_y'] # min max [m/s]
        ang_vel_yaw = self.control_range['cmd_ang']  # min max [rad/s] 

        _, key1, key2, key3 = jax.random.split(rng, 4)

        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]]) 
        return new_cmd

    def reset(self, rng: jp.ndarray, initial_xy=jp.array([0.,0.]), manual_cmd=jp.array([0.,0.,0.]), traj = jp.zeros((500, 3))) -> State:
        """Resets the environment to an initial state.self
        
        Args:
            rng: random number generator
            initial_xy: offset position for the robot

        Returns:
            state: the initial state of the environment
        """

        rng, rng1, rng2, rng3, rng4, rng5 ,rng6, rng7, kp_rng, kd_rng, motor_strength_rng = jax.random.split(rng, 11)
        
        # Actuator randomisation
        kp_factor = jax.random.uniform(kp_rng, (1,), minval=self.kp_range[0], maxval=self.kp_range[1])
        kd_factor = jax.random.uniform(kd_rng, (1,), minval=self.kd_range[0], maxval=self.kd_range[1])
        motor_strength = jax.random.uniform(motor_strength_rng, (1,), minval=self.motor_strength_range[0], maxval=self.motor_strength_range[1])


        reset_pos = self.default_pos
        reset_x= initial_xy[0]+jax.random.uniform(rng1, (1,), minval=self.x_pos[0], maxval=self.x_pos[1])
        reset_y= initial_xy[1]+jax.random.uniform(rng2, (1,), minval=self.y_pos[0], maxval=self.y_pos[1])

        # Randomisation of the drop off position
        theta = self.theta # in rad
        a_x = self.a_x
        a_y = self.a_y
        a_x=jax.random.uniform(rng6, (1,), minval=a_x[0], maxval=a_x[1])
        a_y=jax.random.uniform(rng7, (1,), minval=a_y[0], maxval=a_y[1])
        a_x, a_y = a_x/jp.linalg.norm(jp.array([a_x, a_y])), a_y/jp.linalg.norm(jp.array([a_x, a_y]))
        
        theta = jax.random.uniform(rng5, (1,), minval=theta[0], maxval=theta[1])      
        q1 = jp.cos(theta/2)
        q2 = a_x*jp.sin(theta/2)
        q3 = a_y*jp.sin(theta/2)
        q4 = 0

        # Reset position and orientation
        #reset_pos = reset_pos.at[0:7].set(jp.array([reset_x[0], reset_y[0], 0.27, q1[0], q2[0], q3[0], q4]))
        # Reset position only
        reset_pos = reset_pos.at[0:7].set(jp.array([reset_x[0], reset_y[0], 0.27, 1, 0, 0, 0]))        

        # Get initial state
        data = self.pipeline_init(reset_pos, jp.zeros((self.sys.nv,)))
        reward, done, zero = jp.zeros(3)
        command_rand = self._resample_commands(rng3)
        command = jp.where(self.manual_control, manual_cmd, command_rand)     
        state_info = {
            'rng': rng,
            'action_minus_2t': jp.zeros(self.action_shape), # added for smoothness
            'last_act': jp.zeros(self.action_shape),
            'last_vel': jp.zeros(18),
            'foot_acc': jp.zeros(12),
            'command': command,
            'contact': jp.zeros(4, dtype=bool), 
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'feet_contact_time': jp.zeros(4),
            'rewards': {reward_: jp.array(0.) for reward_ in self.reward_scales.keys()},
            'kick': jp.array([0.0, 0.0]),
            'step': jp.array(0.),
            'time_out': jp.array(0.),
            'nan': jp.array(0.),
            'kp_factor': kp_factor,
            'kd_factor': kd_factor,
            'motor_strength': motor_strength,
            'gait_idx': jp.array(0.),
            'trajectory': traj,
            'force_kick': jp.array([0.0,0.0]),
            'kick_counter': jp.array(0.),
            'kick_counter_initial': jp.array(0.),
            'force_direction_counter': jp.array(0),
            'kick_theta': jp.array(0.0),
            'kick_force_magnitude': jp.array(0.0),
            'last_kick_force_magnitude': jp.array(0.0),
            'des_foot_height': jp.zeros((4,50)),
            'foot_pos': jp.zeros((4,3)),

        }
        # Define obs history
        obs_history = jp.zeros(self.num_history_actor * self.single_obs_size)  # store num_history steps of history
        privileged_obs_history = jp.zeros(self.num_history_critic*self.privileged_obs_size)
        # Get initial observation
        obs, privileged_obs = self._get_obs(data, state_info, obs_history, privileged_obs_history, obs_rng=rng4)

        metrics = {
            'total_dist': 0.0,
            'total_time': 0.0, 
            'power': 0.0,
            'p_gains':jp.zeros(12),
            'local_v':jp.array([0.0, 0.0, 0.0]), 
            'm_total':jp.sum(self.sys.body_mass), 
            'foot_pos_z':jp.zeros(4),
            'target_dof_pos': jp.zeros(12),
            'dof_pos': jp.zeros(12),
            'glob_pos': jp.zeros(2),
            'command': jp.zeros(3),
            }
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        return State(pipeline_state=data,
                     obs=obs,
                     privileged_obs=privileged_obs,
                     reward=reward,
                     done=done,
                     metrics=metrics,
                     info=state_info
                     )

    def compute_torque(self, data: mjx.Data, action: jp.ndarray, actuator_param: jp.ndarray) -> jp.ndarray:
        """
        :param action: in (-1, 1)
        :return:
        """
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]

        

        if self.control_mode == "P" or self.control_mode == "VIC_1" or self.control_mode == "VIC_2" or self.control_mode == "VIC_3" or self.control_mode == "VIC_4":
            target_dof_pos = jp.clip(self.action_scale * action[:12] + self.default_pos[7:],
                                    a_min=self.lower_limits, a_max=self.upper_limits)
            p_gains = self.compute_stiffness(self.action_stiff_scale * action)
            
            if self.control_mode == "VIC_1" or self.control_mode == "VIC_2" or self.control_mode == "VIC_3" or self.control_mode == "VIC_4":
                d_gains = 0.2*jp.sqrt(p_gains)
            else:
                d_gains = self.d_gains
            if self.randomize:
                p_gains = p_gains * actuator_param[0]
                d_gains = d_gains * actuator_param[1]
            torques = p_gains * (target_dof_pos - dof_pos) - d_gains * dof_vel
        elif self.control_mode == "T":
            torques = unscale(self.action_scale * action[:12], lower=-self.torque_limits, upper=self.torque_limits)
        
        if self.randomize:
            torques = jp.clip(torques*actuator_param[2], a_min=-self.torque_limits, a_max=self.torque_limits)
        else:
            torques = jp.clip(torques, a_min=-self.torque_limits, a_max=self.torque_limits)

        # Add the force kick to the torques
        torques = jp.concatenate([torques, action[-2:]])

        return torques

    def compute_stiffness(self, action: jp.ndarray) -> jp.ndarray:
        """
        Compute the stiffness for each joint based on the action

        Args:
            action: the action to be taken

        Returns:    
            p_gains: the proportional gains (12) for each joint 
        """

        if self.control_mode == "P" or self.control_mode == "T":
            return self.p_gains
        elif self.control_mode == "VIC_1":
            action_stiff = unscale(jp.tile(action[12:12+3],4), self.stiff_range[0], self.stiff_range[1])
        elif self.control_mode == "VIC_2":
            action_stiff = unscale(jp.repeat(action[12:12+4],3), self.stiff_range[0], self.stiff_range[1])
        elif self.control_mode == "VIC_3":
            action_stiff = unscale(action[12:12+12], self.stiff_range[0], self.stiff_range[1])
        elif self.control_mode == "VIC_4":
            stiff_leg = jp.tile(unscale(action[12:12+4], self.stiff_range[0], self.stiff_range[1]), 3).reshape(3,4)
            stiff_joint = unscale(action[12+4:12+4+3], self.stiff_range[0], self.stiff_range[1])
            action_stiff = jp.ravel((stiff_leg*stiff_joint[:,jp.newaxis]).T)
        else:
            raise RuntimeError("control model: P|T")
        p_gains = self.p_gains * action_stiff
        return p_gains


    def kick_robot(self, state: State, rng: jp.ndarray):
        """
        Adjusts the velocity of the robot base to simulate a kick, for every push_interval steps.

        Args:
            state: current state
            rng: random key
        """
        push_interval = self.push_interval
        kick_theta = jax.random.uniform(rng, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0  #& (state.info['step'] > 0)
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})
        # update st ate info
        state.info['kick'] = kick
        return state
       
    def force_kick_robot(self, state: State, rng: jp.ndarray):
        # This is intended to be used for evaluation only
        if self.enable_force_kick:
            rng_kick, rng_theta, rng_impulse = jax.random.split(rng, 3)
            # Push randomly
            kick_theta = jax.random.uniform(rng_theta, maxval= 2* jp.pi)
            kick_force = jax.random.uniform(rng_kick, minval=50.0, maxval=self.kick_force)
            kick_impulse = jax.random.uniform(rng_impulse, minval=self.force_kick_impulse[0], maxval=self.force_kick_impulse[1])
            kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)]) * kick_force 
            kick_condition = jp.logical_and(jp.mod(state.info['step'], self.force_kick_interval)==0, state.info['step']>1 )
            # Get the random values at kick interval
            state.info['force_kick'] = jp.where(kick_condition, kick, state.info['force_kick'])
            state.info['kick_theta']=jp.where(kick_condition, kick_theta, state.info['kick_theta'])
            state.info['kick_force_magnitude'] = jp.where(kick_condition, kick_force , 0.0) #state.info['kick_force_magnitude']
            
            # Hold the same value for a few steps
            state.info['kick_counter_initial']= jp.where(jp.logical_and(self.impulse_force_kick, kick_condition), ((kick_impulse/state.info['kick_force_magnitude']) / self.dt).astype(int), self.force_kick_counter)
            #jax.debug.print('Kick counter initial: {x}', x=state.info['kick_counter_initial'])
            state.info['kick_counter'] = jp.where(kick_condition, state.info['kick_counter_initial'], state.info['kick_counter'])
            state.info['kick_counter'] = jp.where( state.info['kick_counter']>-1 , state.info['kick_counter']-1, state.info['kick_counter'])
            #jax.debug.print('Kick counter: {x}', x=state.info['kick_counter'])
            state.info['force_kick'] = jp.where(state.info['kick_counter']>-1, state.info['force_kick'], 0.0)
            state.info['kick_theta'] = jp.where(state.info['kick_counter']>-1, state.info['kick_theta'], 0.0)
            state.info['kick_force_magnitude'] = jp.where(state.info['kick_counter']>-1, state.info['kick_force_magnitude'],0.0)
            #state.info['last_kick_force_magnitude'] = state.info['kick_force_magnitude']
            #jax.debug.print('Force kick: {x}', x=state.info['kick_force_magnitude'])
        else:
            state.info['force_kick'] = jp.zeros(2)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        """
        Runs one control step of the environment.
        
        Args:
            state: current state
            action: action to take

        Returns:
            state: the new state of the environment
        """
        # For randomization
        rng, obs_rng, kick_noise, cmd_rng = jax.random.split(state.info['rng'], 4)
        
        # kick robot
        state = self.kick_robot(state, kick_noise)
        #state = self.force_kick_robot(state, kick_noise)
        state = self.force_kick_robot(state, kick_noise)

        # Add the force kick to the action, if not needed it will be zero
        action_kick = jp.concatenate([action, jp.array(state.info['force_kick'])])
        # Get the current state of the physics
        data0 = state.pipeline_state

        # Performs physics timesteps per control step
        actuator_param = jp.concatenate([state.info['kp_factor'], state.info['kd_factor'], state.info['motor_strength']])

        p_gains = self.compute_stiffness(self.action_stiff_scale*action)
        data = self.pipeline_step(data0, action_kick, actuator_param) #passed data is action as angle -> convert to torque in mjx
        
        # ----------------- POST Physics step --------------- #
        # Here we 1.)extract the state information, 2.)check termination, 3.) calculate the reward and 4.) get observations 
        
        #1.) Extract the state information: positions/velocities, joint angles/velocities, foot contacts, etc.
        x, xd = self._pos_vel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]  

        # Foot contact data
        foot_pos = data.site_xpos[self.feet_site_id]  # pytype: disable=attribute-error
        #jax.debug.print('Foot pos: {x}', x=foot_pos)
        state.info['foot_pos'] = foot_pos
        target_foot_height = self.get_des_foot_height(state.info['gait_idx'])
        state.info['des_foot_height']=  self.get_foot_height_traj(state.info['step'] )
        
        # jax.debug.print('Target foot height: {x}', x=state.info['des_foot_height'])
        # jax.debug.print('Shape of foot height: {x}', x=state.info['des_foot_height'].shape)

        #state.info['des_foot_height'] = target_foot_height
        ## Foot contacts
        foot_contacts = jp.zeros((4),dtype=int)
        foot_contacts = foot_contacts.at[0:4].set(self.get_foot_contacts(data)[0:4,0].astype(int))
        foot_floor_dist = jp.zeros((4),dtype=float)
        foot_floor_dist = foot_floor_dist.at[:].set(data.contact.dist[foot_contacts])
        ## general contact management
        contact = foot_floor_dist< 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']        
        contact_filt_cm = (foot_floor_dist < 1e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        # for observations
        state.info['contact'] = contact_filt_mm
        state.info['feet_air_time'] += self.dt
        state.info['feet_contact_time'] += self.dt

        # 2.) Check termination
        done = self._check_terminate(data, x, state.info['step'])

        # 3.) Caluclate reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x), 
            #'torques': self._reward_torques(data.qfrc_actuator), 

            #'smooth_rate': self._reward_smooth_rate(joint_vel, state.info['last_vel']),
            'stand_still': self._reward_stand_still( 
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            #'feet_contact_time': self._reward_feet_contact_time(
            #     state.info['feet_contact_time'],
            #     state.info['command'],
            # ),
            'foot_slip': self._reward_foot_slip(data, xd, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step'], data=data),
            'action_rate': self.action_rate(action, state.info['last_act']),
            #'action_smoothnes': self.action_rate2(action, state.info['last_act'], state.info['action_minus_2t']),
            
            'rew_pos_limits': self._reward_pos_limits(joint_angles),
            'rew_stiff_limits': self._reward_stiff_limits(p_gains),
            'rew_acceleration': self._reward_acceleration(joint_vel, state.info['last_vel'][6:]),
            #'rew_velocity': self._reward_velocity(joint_vel),
            'rew_collision': self._reward_collision(data),
            'rew_power': self._reward_power(data.ctrl[:12], joint_vel),
            'rew_power_distro': self._reward_power_distro(data.ctrl[:12], joint_vel),
            # Feet posture
            'hip': self.rew_hip(joint_angles),
            # Variable impedance control
            'rew_joint_track': self._reward_joint_track(joint_angles, action[:12]),
            'rew_base_height': self._reward_base_height(data.qpos[2]),
            'rew_foot_clearance': self._reward_foot_clearance( xd , foot_pos[:, 2], 0.09),
            #'rew_foot_tracking': self._reward_foot_clearance(state.info['gait_idx'], foot_z=foot_pos[:, 2])
        }
        rewards = {
            k: v * self.reward_scales[k] for k, v in rewards.items()
        }
        #Reward clipping like in unitree rl
        reward = jp.clip(sum(rewards.values())*self.dt , 0.0, 10000.0)

        # state management
        state.info['feet_air_time'] *= ~contact_filt_mm # bitwise negation
        state.info['feet_contact_time'] *= contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step']+= 1
        state.info['gait_idx'] = jp.remainder(state.info['step']*self.dt, 1.0)
        state.info['nan']= jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        state.info['time_out'] = state.info['step'] > self.episode_length
        state.info['rng'] = rng
        state.info['action_minus_2t'] = state.info['last_act'] 
        state.info['last_act'] = action
        state.info['last_vel'] = data.qvel
        state.info['last_qpos'] = data.qpos
        state.info['foot_pos_z'] = foot_pos[:, 2]

        # # log total displacement as a proxy metric
        # jax.debug.print('Total distance: {x}', x=jp.linalg.norm(x.pos[self._torso_idx-1][:2]))
        # jax.debug.print('Total distance(old): {x}', x=math.normalize(x.pos[self._torso_idx - 1])[1])
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics['total_time'] = state.info['step'] * self.dt
        state.metrics['power'] = jp.sum(jp.abs(data.ctrl[:12])) * jp.abs(jp.sum(joint_vel))
        
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        state.metrics['local_v'] = local_vel
        state.metrics['foot_pos_z'] = foot_pos[:, 2]

        state.metrics['target_dof_pos'] = jp.clip(self.action_scale * action[:12] + self.default_pos[7:], a_min=self.lower_limits, a_max=self.upper_limits)
        state.metrics['dof_pos'] = data.qpos[7:]
        state.metrics['glob_pos'] = data.qpos[:2]
        state.metrics['command'] = state.info['command']
        state.metrics['p_gains'] = p_gains   

        # sample new command
        state.info['command'] = jp.where(
            jp.logical_and( state.info['step'] % self.sample_command_interval == 0, jp.logical_not(self.manual_control)), #normally a ~ would be sufficient but it somehow converts the boolean into -2 and thus this and does not work anymore
            self._resample_commands(cmd_rng),
            state.info['command'],
            )
        
        # track the trajectory (for evaluation)
        state.info['command'] = jp.where(
            jp.logical_and(self.manual_control, self.track_traj),
            state.info['trajectory'][state.info['step'].astype(int), :],
            state.info['command'],
        )
        #jax.debug.print('Command: {x}', x=state.info['command'])
        # reset the step counter when done
        state.info['step'] = jp.where(
        (state.info['step'] > self.episode_length), 0, state.info['step']
        )
        state.metrics.update(state.info['rewards'])

        # observation
        obs, privileged_obs = self._get_obs(data, state.info, state.obs, state.privileged_obs, obs_rng=obs_rng)
        state.info['nan'] |= jp.isnan(obs).any() | jp.isnan(privileged_obs).any()
        done = jp.float32(done)
        
        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, privileged_obs=privileged_obs
        )
        return state

    def _get_jacobian(self, data, qpos):
        d = smooth.kinematics(self.sys, data.replace(qpos=qpos))
        d = smooth.com_pos(self.sys, d)
        foot_pos = d.xpos[self.foot_body_id]
        jacp_FR, jacr_FR = jac(self.sys, d, foot_pos[0], self.foot_body_id[0])
        jacp_FL, jacr_FL = jac(self.sys, d, foot_pos[1], self.foot_body_id[1])
        jacp_RR, jacr_RR = jac(self.sys, d, foot_pos[2], self.foot_body_id[2])
        jacp_RL, jacr_RL = jac(self.sys, d, foot_pos[3], self.foot_body_id[3])

        J_FR = jacp_FR
        J_FL = jacp_FL
        J_RR = jacp_RR
        J_RL = jacp_RL

        J = jp.concatenate([J_FR.T, J_FL.T, J_RR.T, J_RL.T], axis=0)
        return J

    def _get_obs(self,
                  data: mjx.Data,
                  state_info: Dict[str, Any],
                  obs_history: jax.Array,
                  privileged_obs_history: jax.Array,
                  obs_rng: jp.ndarray ,
                 ) -> jp.ndarray:
        """
        Get observation from the environment. The observation is a numpy array containing the following
        items: [torso_z, yaw rate, proj_gravity, qpos, qvel, last_act, command] 

        Args:
            data: mjx.Data
            state_info: a dictionary containing the following keys [rng, last_act, last_vel, command, last_contact, feet_air_time, feet_contact_time, rewards, kick, step] 
            obs_history: jax.Array
        """
        x, xd = self._pos_vel(data)
        inv_torso_rot = math.quat_inv(x.rot[0]) #calculates the inverse of a quaternion
        torso_z = data.qpos[2:3]

        # Calculating the local measurable velocities
        local_v = math.rotate(xd.vel[0], inv_torso_rot)
        local_w = math.rotate(xd.ang[0], inv_torso_rot) # yaw rate at index 2
        
        proj_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot)      # projected gravity
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        foot_pos = data.site_xpos[self.feet_site_id]

        J = self._get_jacobian(data, data.qpos)
        # Local foot position
        foot_indices = self.foot_body_id - 1
        foot_transform = x.take(foot_indices)
        foot_transform_local = foot_transform.vmap().to_local(x.take(jp.array([0, 0, 0, 0])))
        foot_pos_local = foot_transform_local.pos.reshape(-1)
        foot_vel_local = J @ data.qvel

        # Joint error
        target_dof_pos = jp.clip(self.action_scale * state_info['last_act'][:12]  + self.default_pos[7:],a_min=self.lower_limits, a_max=self.upper_limits)
        err = target_dof_pos - data.qpos[7:]

        # Orientation quaternion
        quaternion = data.qpos[3:7] 
        rpy = math.quat_to_euler(quaternion)

        # Observation, if anything is changed here, the noise vector must be adjusted accordingly
        obs = jp.concatenate([
            self.local_v_scale*jp.array(local_v),
            self.local_w_scale*jp.array(local_w),
            proj_gravity,
            data.qpos[7:]-self.default_pos[7:],  
            self.joint_vel_scale *data.qvel[6:],
            state_info['last_act'], 
            self.command_scale * state_info['command'],
            state_info['contact'],
        ])

        obs = jp.clip(obs, -100.0, 100.0)
        # Privileged observation: passed to the critic
        privileged_obs = jp.concatenate([
            state_info['kp_factor'],
            state_info['kd_factor'],
            state_info['motor_strength'],
            jp.array([self.sys.geom_friction[0, 0]]),
            jp.array([self.sys.body_mass[1]]),
            state_info['kick'],
            
            obs
        ])

        assert obs.shape[0] == self.single_obs_size, f"obs.shape: {obs.shape}"
        assert privileged_obs.shape[0] == self.privileged_obs_size, f"privileged_obs.shape {privileged_obs.shape}"

        # Add noise to the observation(not privileged), this has to be altered if the observation space changes
        noise_vec = jax.random.uniform(obs_rng, (self.single_obs_size,), minval=-1., maxval=1.)
        noise_vec = noise_vec.at[:3].multiply(self.local_v_noise*self.local_v_scale)
        noise_vec = noise_vec.at[3:6].multiply(self.local_w_noise*self.local_w_scale)
        noise_vec = noise_vec.at[6:9].multiply(self.gravity_noise)
        noise_vec = noise_vec.at[9:21].multiply(self.joint_noise)
        noise_vec = noise_vec.at[21:33].multiply(self.joint_vel_noise*self.joint_vel_scale)
        noise_vec = noise_vec.at[33:].multiply(0.0)
        obs = jp.where(self.randomize, obs+noise_vec, obs)

        # Stack observations through time all in 1x(timesteps x obs_size) array
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
        privileged_obs = jp.roll(privileged_obs_history, privileged_obs.size).at[:privileged_obs.size].set(privileged_obs)

        return obs, privileged_obs

    def _check_terminate(self, data: mjx.Data, x, step) -> bool:
        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0]) 
        # check if robot is falling, dot product of rotated upward direction and actual up. Less than 0 means falling.
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        # Check if compliant with limits
        done |= jp.any(data.qpos[7:] < self.lower_limits) 
        done |= jp.any(data.qpos[7:] > self.upper_limits)        
        # Termination if body touches the ground
        terminate_contacts = jp.zeros((7),dtype=int)
        terminate_contacts = terminate_contacts.at[:].set(self.get_terminate_contacts(data)[:,0])
        done |= jp.any(data.contact.dist[terminate_contacts] < 0.0)
        # Terminate if body height not high enough
        #done |= data.xpos[self._torso_idx, 2] < self.min_z
        # Termination in case of finite terrain
        done_map = (jp.abs(data.qpos[0]) > 10.0) | (jp.abs(data.qpos[1]) > 10.0) | (jp.abs(data.qpos[2]) < -3.0)
        done |= done_map*self.terminate_map
        # Catch em all Nans
        done |= jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return done

    ### ------------ Reward functions---------------- ###
    def _reward_tracking_lin_vel(
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-4 * lin_vel_error) # change to sigma
        return lin_vel_reward

    def _reward_tracking_ang_vel(
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-4 * ang_vel_error) #change to sigma

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array: 
        # Penalize z axis base linear velocity
        return jp.sum(jp.square(xd.vel[0, 2]))
    
    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array: 
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))
    
    def _reward_orientation(self, x: Transform) -> jax.Array: 
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
    
    def _reward_acceleration(self, last_dof_vel: jax.Array, dof_vel:jax.Array) -> jax.Array:
        return jp.sum(jp.square((dof_vel - last_dof_vel)/self.dt))
    
    def _reward_velocity(self, dof_vel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(dof_vel))

    def _reward_torque_limit(self, torque: jax.Array) -> jax.Array:
        return jp.exp(-jp.sum(jp.abs(torque)-0.1*self.torque_limits))
    
    def _reward_pos_limits(self, pos: jax.Array) -> jax.Array:
        out_of_bounds = -(pos - self.lower_limits).clip(max=0.)
        out_of_bounds += (pos - self.upper_limits).clip(min=0.)
        return jp.sum(out_of_bounds)
    
    def _reward_stiff_limits(self, stiff: jax.Array) -> jax.Array:
        out_of_bounds = -(stiff - self.stiff_range[0]*self.p_gain).clip(max=0.)
        out_of_bounds += (stiff - self.stiff_range[1]*self.p_gain).clip(min=0.)
        return jp.sum(out_of_bounds)
    
    def _reward_collision(self, data) -> jax.Array:
        body_contacts = jp.zeros(len(self.body_geom_id),dtype=int)
        body_contacts = body_contacts.at[:].set(self.get_body_contacts(data)[:,0])
        #jax.debug.print('Distances: {x}', x=data.contact.dist[body_contacts])
        return 1.0*jp.sum(data.contact.dist[body_contacts] < 0.05)
        
    ## Related to smoothness of the actions:
    def _reward_smooth_rate( # to be continued...(why the velocities?)
            self, joint_vel: jax.Array, last_vel: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jp.exp(-0.4*jp.linalg.norm(joint_vel - last_vel))

    def action_rate(self, action: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(action - last_act))
    
    def action_rate2(self, action: jax.Array, last_act: jax.Array, action_minus_2t:jax.Array) -> jax.Array:
        return jp.exp(-0.05*jp.sum(jp.power(action-2*last_act+action_minus_2t,2)))
    
    def rew_hip(
            self, joint_angles: jax.Array
    ):
        return jp.exp(-0.4*jp.sum(jp.square(joint_angles[::3])))
    

    def _reward_feet_air_time(
            self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            #math.normalize(commands[:])[1] > 0.05
            jp.any(jp.abs(commands) > 0.05)
        ) #  * jp.exp(-0.2*cmd_norm)  # no reward for zero command and weighting reward by velocity
        return rew_air_time
    
    def _reward_foot_clearance(
            self, xd: Motion, foot_z: jax.Array, des_z = 0.09
    ) -> jax.Array:
        foot_indices = self.foot_body_id - 1
        # jax.debug.print('Foot indices: {x}', x=foot_indices)
        foot_vel = xd.take(foot_indices).vel
        #jax.debug.print('Foot vel: {x}', x=foot_vel)
        # jax.debug.print('Foot vel mag: {x}', x=jp.linalg.norm(foot_vel[:, :2], axis=1))
        rew_clearance = jp.sum(jp.square(foot_z - des_z) * jp.linalg.norm(foot_vel[:, :2], axis=1 ))
        return rew_clearance

    def _reward_feet_contact_time(
            self, contact_time: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Punish contact time.
        rew_contact_time = jp.sum(contact_time)
        rew_contact_time *= (
                math.normalize(commands[:2])[1] > 0.1
        )  # no reward for zero command
        return rew_contact_time

    def _reward_stand_still( 
            self,
            commands: jax.Array,
            joint_angles: jax.Array,
    ) -> jax.Array:
        
        # return jp.sum(jp.abs(joint_angles - self.default_pos[7:])) * (
        # math.normalize(commands[:2])[1] < 0.1
        # )
    
        return jp.sum(jp.abs(joint_angles - self.default_pos[7:])) * (
            jp.all(jp.abs(commands) < 0.05)
        )

    def _reward_foot_slip(self, pipeline_state: State, xd, contact_filt: jax.Array) -> jax.Array:
        foot_indices = self.foot_body_id - 1
        foot_vel = xd.take(foot_indices).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step, data: mjx.Data) -> jax.Array:
        #receive reward if the robot is terminated, the step is within its range and the robot is within the limits
        default_termination = done & (step < self.episode_length)
        terrain_termination = done & (step < self.episode_length) & self.terminate_map*((jp.abs(data.qpos[0]) < 10.0) & (jp.abs(data.qpos[1]) < 10.0) & (jp.abs(data.qpos[2]) > -3.0))
        return jp.where(self.terminate_map, terrain_termination, default_termination)

    def _reward_power(self, torques: jax.Array, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques)*jp.abs(qvel))

    def _reward_power_distro(self, torques: jax.Array, qvel: jax.Array) -> jax.Array:
        power = jp.abs(torques * qvel)

        leg_power = jp.array([jp.sum(power[:3]), jp.sum(power[3:6]) , \
                              jp.sum(power[6:9]), jp.sum(power[9:12])])
        var = jp.var(leg_power)
        return var
    
    def _reward_joint_track(self, joint_angles: jax.Array, action: jax.Array) -> jax.Array:
        target_dof_pos = jp.clip(self.action_scale * action + self.default_pos[7:],a_min=self.lower_limits, a_max=self.upper_limits)
        return jp.sum(jp.square(joint_angles-target_dof_pos))
    


    ## Rewards for Gait behaviours ---------------

    def get_des_foot_height(self, gait_cycle_idx: jax.Array) -> jax.Array:
        foot_cycles = jp.array([
            gait_cycle_idx+0.5,
            gait_cycle_idx ,
            gait_cycle_idx,
            gait_cycle_idx +0.5,
        ])
        foot_cycles = jp.remainder(foot_cycles, 1.0)
        phases = 1 - jp.abs(1.0 - jp.clip((foot_cycles*2.0)-1.0, 0.0,1.0) *2.0 )
        target_height = 0.2*phases
        return target_height
    
    def get_foot_height_traj(self, current_step: jax.Array) -> jax.Array:
        steps = jp.arange(50)
        steps = jp.remainder((steps+current_step)*self.dt, 1.0)
        heights = self.get_des_foot_height(steps)
        return heights
    
    

    
    def _reward_base_height(self, base_z: jax.Array) -> jax.Array:
        target_height = 0.27
        #return jp.min(target_height-base_z, 0)
        return jp.square(base_z-target_height)

    