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

config.update("jax_debug_nans", True)


class UnitreeEnv(MjxEnv):
    def __init__(
            self,
            cfg,
            model_path,
            soft_limits=0.9,
    ):
        
        resource_directory = os.path.join(os.getcwd(), 'envs','resources') # it is anticipated to be executed from TALocoMotion
        mj_model = mujoco.MjModel.from_xml_path(os.path.join(resource_directory, model_path))
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.jacobian = 1        # 'sparse'

        super().__init__(mj_model=mj_model,
                         physics_steps_per_control_step=cfg.physics_steps_per_control_step)
        print(self.sys.opt)
        
        self.control_mode = cfg.control_mode
        self.action_scale = cfg.action_scale
        self.num_history = cfg.num_history
        self._kick_vel = cfg.kick_vel
        self.is_training = cfg.is_training
        
        self.episode_length = cfg.episode_length

        self.randomize = cfg.domain_rand.randomisation
        self.local_v_noise = cfg.domain_rand.local_v_noise
        self.local_w_noise = cfg.domain_rand.local_w_noise
        self.joint_noise = cfg.domain_rand.joint_noise
        self.joint_vel_noise = cfg.domain_rand.joint_vel_noise
        self.gravity_noise = cfg.domain_rand.gravity_noise

        self.manual_control = cfg.manual_control.enable
        self.cmd_x = cfg.manual_control.cmd_x
        self.cmd_y = cfg.manual_control.cmd_y
        self.cmd_yaw = cfg.manual_control.cmd_ang

        self.soft_limits = soft_limits
        self.single_obs_size = 48 # defined in _get_obs
        self.priviledged_obs_size = self.single_obs_size

        # Randomization ranges:
        self.x_pos = [-0.1, 0.1]#[-3, 3]
        self.y_pos = [-0.1, 0.1]#[-3, -2] 
        self.theta = [0, jp.pi/8] # in rad
        self.a_x = [-1,1]
        self.a_y = [-1,1]
        
        # set up robot properties
        self._setup()

    def _setup(self):
        self._foot_radius = 0.023

        self._torso_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk')
        
        feet_names = ['FR', 'FL', 'RR', 'RL'] 
        hip_body = ['FR_hip', 'FL_hip', 'RR_hip', 'RL_hip']
        torso_bodies = ['base_mirror_0', 'base_mirror_1', 'base_mirror_2', 'base_mirror_3', 'base_mirror_4']

        body_geometries = [
            "base_0", "base_1", "base_2", 
            "FR_hip", "FR_thigh", "FR_calf_0", "FR_calf_1",  
            "FL_hip", "FL_thigh", "FL_calf_0", "FL_calf_1", 
            "RR_hip", "RR_thigh", "RR_calf_0", "RR_calf_1", 
            "RL_hip", "RL_thigh", "RL_calf_0", "RL_calf_1"
            ]
        
        terminate_geometries = ["base_0", "base_1", "base_2", "FR_hip","FL_hip","RR_hip","RL_hip"]

        feet_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_names
        ]
        feet__geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in feet_names
        ]
        foot_body = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        lower_leg_body=['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']
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

        assert floor_id != -1, 'Floor not found.'
        self.floor_id = floor_id        

        # Scaling of the rewards
        # These rewards are from the tutorial: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        self.reward_scales = {
            # From turtoial
            'tracking_lin_vel': 1.5, 
            'tracking_ang_vel': 0.8, 
            "lin_vel_z": -2.0, 
            "ang_vel_xy": -0.05, 
            "orientation": -5.0, 
            "torques": -0.0002, 
            "smooth_rate": 0.0, #action_rate from tutorial
            'feet_air_time': 0.2,
            'feet_contact_time': 0.0,
            'termination': -10.0,
            'stand_still': -0.5, #-0.5, # adapted
            "foot_slip": -0.1,
            # Additional self created
            "action_rate": -0.01,
            "action_rate2": 0.0,
            "abduction": 0.0,
            "rew_pos_limits": -0.0,
            "rew_acceleartion": -0.000000,
            "rew_collision": -10.0,
        }

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
        expected_total_cont=23#221
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
            #jax.debug.print('Connection index: {x}', x=connection_index)
            conn_indices = conn_indices.at[i,0].set(connection_index[0])        
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
        num_collisions = 19
        expected_total_cont=23
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
        expected_total_cont=23
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
        lin_vel_x = [-0.6, 1.0]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s] 

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
        rand_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]]) 
        #Construct manual command
        manual_command = jp.array([self.cmd_x, self.cmd_y,self.cmd_yaw])
        # Decide whether to use manual control or random command
        #new_cmd = jp.where(self.manual_control, manual_command, rand_cmd)
        new_cmd = rand_cmd

        return new_cmd

    def reset(self, rng: jp.ndarray, initial_xy=jp.array([0.,0.]), manual_control = False) -> State:
        """Resets the environment to an initial state.
        
        Args:
            rng: random number generator
            initial_xy: offset position for the robot
        
        Returns:
            state: the initial state of the environment
        """
        self.manual_control = manual_control

        rng, rng1, rng2, rng3, rng4, rng5 ,rng6, rng7 = jax.random.split(rng, 8)
        
        #
        reset_pos = self.default_pos


        reset_x= initial_xy[0]+jax.random.uniform(rng1, (1,), minval=self.x_pos[0], maxval=self.x_pos[1])
        reset_y= initial_xy[1]+jax.random.uniform(rng2, (1,), minval=self.y_pos[0], maxval=self.y_pos[1])

        # Get random theta and rotation vector
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

        #reset_pos = reset_pos.at[0:7].set(jp.array([reset_x[0], reset_y[0], 0.27, q1[0], q2[0], q3[0], q4]))
        reset_pos = reset_pos.at[0:7].set(jp.array([reset_x[0], reset_y[0], 0.27, 1, 0, 0, 0]))        
        #jax.debug.print('Resulting position: {x}', x=reset_pos)        

        #reset_pos = reset_pos.at[0:7].set(jp.array([0, 0, 0.27, q1[0], q2[0], q3[0], q4]))

        data = self.pipeline_init(reset_pos, jp.zeros((self.sys.nv,)))
        reward, done, zero = jp.zeros(3)
        command = self._resample_commands(rng3)     
        #jax.debug.print('Mjdata: {x}', x=data)
        state_info = {
            'rng': rng,
            'action_minus_2t': jp.zeros(12), # added for smoothness
            'last_act': jp.zeros(12),
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
            'priviledged_obs': jp.zeros(self.single_obs_size, dtype=jp.float32),
            'time_out': jp.array(0.),
        }
        obs_history = jp.zeros(self.num_history * self.single_obs_size)  # store num_history steps of history
        obs, priviledged_obs = self._get_obs(data, state_info, obs_history, obs_rng=rng4)
        obs = obs.at[self.single_obs_size:].set(jp.tile(obs[:self.single_obs_size], self.num_history-1))
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        return State(pipeline_state=data,
                     obs=obs,
                     priviledged_obs=priviledged_obs,
                     reward=reward,
                     done=done,
                     metrics=metrics,
                     info=state_info
                     )

    def compute_torque(self, data: mjx.Data, action: jp.ndarray):
        """
        :param action: in (-1, 1)
        :return:
        """
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]
        
        #action = jp.clip(action, a_min=-100.0, a_max=100.0)

        if self.control_mode == "P":
            #target_dof_pos = self.action_scale * action + self.default_pos[7:]
            target_dof_pos = jp.clip(self.action_scale * action + self.default_pos[7:],
                                    a_min=self.lower_limits, a_max=self.upper_limits)
            err = target_dof_pos - dof_pos
            torques = self.p_gains * err - self.d_gains * dof_vel
        elif self.control_mode == "T":
            torques = unscale(self.action_scale * action, lower=-self.torque_limits, upper=self.torque_limits)
        else:
            raise RuntimeError("control model: P|T")

        return jp.clip(torques, a_min=-self.torque_limits, a_max=self.torque_limits)

    def kick_robot(self, state: State, rng: jp.ndarray):
        """
        Adjusts the velocity of the robot base to simulate a kick, for every push_interval steps.

        Args:
            state: current state
            rng: random key
        """
        push_interval = 10
        kick_theta = jax.random.uniform(rng, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})
        # update state info
        state.info['kick'] = kick
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

        # Get the current state of the physics
        data0 = state.pipeline_state
        
        #action_noise = jax.random.uniform(action_rng, (12,), minval=-self._act_noise, maxval=self._act_noise)
        #action = action.at[:].add(action_noise)

        # Performs physics timesteps per control step
        data = self.pipeline_step(data0, action) #passed data is action as angle -> convert to torque in mjx
        # Alternative approach: use same torque for every timestep -> speedup
        #ctrl = self.compute_torque(action) 
        #data = self.pipeline_step2(data0, ctrl)

        # ----------------- POST Physics step --------------- #
        # Here we 1.)extract the state information, 2.)check termination, 3.) calculate the reward and 4.) get observations 
        
        #1.) Extract the state information: positions/velocities, joint angles/velocities, foot contacts, etc.
        x, xd = self._pos_vel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel       

        # Foot contact data
        foot_pos = data.site_xpos[self.feet_site_id]  # pytype: disable=attribute-error
        ## Foot contacts
        foot_contacts = jp.zeros((4),dtype=int)
        foot_contacts = foot_contacts.at[0:4].set(self.get_foot_contacts(data)[0:4,0].astype(int))
        foot_floor_dist = jp.zeros((4),dtype=float)
        foot_floor_dist = foot_floor_dist.at[:].set(data.contact.dist[foot_contacts])
        #jax.debug.print('Foot distances: {x}', x=foot_floor_dist)
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
            'torques': self._reward_torques(data.qfrc_actuator),  # pytype: disable=attribute-error

            'smooth_rate': self._reward_smooth_rate(joint_vel, state.info['last_vel']),
            'stand_still': self._reward_stand_still( 
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'feet_contact_time': self._reward_feet_contact_time(
                state.info['feet_contact_time'],
                state.info['command'],
            ),
            'foot_slip': self._reward_foot_slip(data, xd, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),

            'action_rate': self.action_rate(action, state.info['last_act']),

            'action_rate2': self.action_rate2(action, state.info['last_act'], state.info['action_minus_2t']),
            'abduction': self.abduction(joint_angles),
            'rew_pos_limits': self._reward_pos_limits(joint_angles),
            'rew_acceleartion': self._reward_acceleration(joint_vel, state.info['last_vel']),
            'rew_collision': self._reward_collision(data),
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
        state.info['time_out'] = state.info['step'] > self.episode_length
        state.info['rng'] = rng
        state.info['action_minus_2t'] = state.info['last_act'] 
        state.info['last_act'] = action
        state.info['last_vel'] = data.qvel
        state.info['last_qpos'] = data.qpos
        state.info['foot_pos_z'] = foot_pos[:, 2]
        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        
        # sample new command
        state.info['command'] = jp.where(
            state.info['step'] > self.episode_length,
            self._resample_commands(cmd_rng),
            state.info['command'],
            )
        
        # reset the step counter when done
        state.info['step'] = jp.where(
        (state.info['step'] > self.episode_length), 0, state.info['step']
        )
        
        state.metrics.update(state.info['rewards'])

        # observation
        obs, priviledged_obs = self._get_obs(data, state.info, state.obs, obs_rng=obs_rng)
        done = jp.float32(done)

        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, priviledged_obs=priviledged_obs
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
        reward = jp.array([sum(state_info['rewards'].values())])
        foot_pos = data.site_xpos[self.feet_site_id]

        J = self._get_jacobian(data, data.qpos)
        # Local foot position
        foot_indices = self.foot_body_id - 1
        foot_transform = x.take(foot_indices)
        foot_transform_local = foot_transform.vmap().to_local(x.take(jp.array([0, 0, 0, 0])))
        foot_pos_local = foot_transform_local.pos.reshape(-1)
        foot_vel_local = J @ data.qvel

        #jax.debug.print('Foot pos local: {x}', x=foot_pos_local)

        # Joint error
        target_dof_pos = jp.clip(self.action_scale * state_info['last_act']  + self.default_pos[7:],a_min=self.lower_limits, a_max=self.upper_limits)
        err = target_dof_pos - data.qpos[7:]

        # Orientation quaternion
        quaternion = data.qpos[3:7] 
        rpy = math.quat_to_euler(quaternion)

        # Observation space dimension: 1+6+3+12+12+12+4+3 53 #old calculation
        obs = jp.concatenate([
            #torso_z,
            jp.array([2.0, 2.0, 2.0, 0.25, 0.25, 0.25]) * jp.concatenate([local_v, local_w]),  # yaw rate at index 6
            proj_gravity,
            data.qpos[7:]-self.default_pos[7:],  # joint angles
            0.1 *data.qvel[6:],
            state_info['last_act'], 
            #state_info['contact'], #added
            2.0* state_info['command'],
            # foot_pos_local,
            # foot_vel_local,
            # err,
            # rpy[:-1], # quaternion, leaving yaw out
        ])

        obs = jp.clip(obs, -100.0, 100.0)

        priviledged_obs = jp.concatenate([
            # Privileged
            obs
        ])

        assert obs.shape[0] == self.single_obs_size, f"obs.shape: {obs.shape}"
        assert priviledged_obs.shape[0] == self.priviledged_obs_size, f"priviledged_obs.shape {priviledged_obs.shape}"

        # Add noise to the observation, this has to be altered if the observation space changes
        noise_vec = jax.random.uniform(obs_rng, (self.single_obs_size,), minval=-1., maxval=1.)
        noise_vec = noise_vec.at[:3].multiply(self.local_v_noise*2.0)
        noise_vec = noise_vec.at[3:6].multiply(self.local_w_noise*0.25)
        noise_vec = noise_vec.at[6:9].multiply(self.gravity_noise)
        noise_vec = noise_vec.at[9:21].multiply(self.joint_noise)
        noise_vec = noise_vec.at[21:33].multiply(self.joint_vel_noise*0.1)
        noise_vec = noise_vec.at[33:].multiply(0.0)
        #jax.debug.print('observsation before noise: {x}', x=obs)
        obs = jp.where(self.randomize, obs+noise_vec, obs)
        #jax.debug.print('observsation after noise: {x}', x=obs)

        # Stack observations through time all in 1x(timesteps x obs_size) array
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs, priviledged_obs

    def _check_terminate(self, data: mjx.Data, x, step) -> bool:
        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        # check if robot is falling, dot product of rotated upward direction and actual up. Less than 0 means falling.
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        # Check if compliant with limits
        # done |= jp.any(data.qpos[7:] < self.lower_limits) 
        # done |= jp.any(data.qpos[7:] > self.upper_limits)
        # Old termination: based on z-height
        # done |= data.xpos[self._torso_idx, 2] < self.min_z
        
        # New termination: If body touches the ground
        terminate_contacts = jp.zeros((7),dtype=int)
        terminate_contacts = terminate_contacts.at[:].set(self.get_terminate_contacts(data)[:,0])
        done |= jp.any(data.contact.dist[terminate_contacts] < 0.0)
        #jax.debug.print('Timeout: {x}', x=step > self.episode_length)
        #done |= step > self.episode_length

        return done

    ### ------------ Reward functions---------------- ###
    def _reward_tracking_lin_vel(#DONE
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-4 * lin_vel_error) # change to sigma
        return lin_vel_reward

    def _reward_tracking_ang_vel(#DONE
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-4 * ang_vel_error) #change to sigma

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array: #DONE
        # Penalize z axis base linear velocity
        return jp.sum(jp.square(xd.vel[0, 2]))
    
    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array: #DONE
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))
    
    def _reward_orientation(self, x: Transform) -> jax.Array: #DONE
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
    
    def _reward_acceleration(self, last_dof_vel: jax.Array, dof_vel:jax.Array) -> jax.Array:
        return jp.sum(jp.square((dof_vel - last_dof_vel)/self.dt))
    
    def _reward_torque_limit(self, torque: jax.Array) -> jax.Array:
        return jp.exp(-jp.sum(jp.abs(torque)-0.1*self.torque_limits))
    
    def _reward_pos_limits(self, pos: jax.Array) -> jax.Array:
        out_of_bounds = -(pos - self.lower_limits).clip(max=0.)
        out_of_bounds += (pos - self.upper_limits).clip(min=0.)
        return jp.sum(out_of_bounds)
    
    def _reward_collision(self, data) -> jax.Array:
        body_contacts = jp.zeros((19),dtype=int)
        body_contacts = body_contacts.at[:].set(self.get_body_contacts(data)[:,0])
        return 1.0*jp.sum(data.contact.dist[body_contacts] < 0.0)
        
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
    
    def abduction(
            self, joint_angles: jax.Array
    ):
        return jp.exp(-4*jp.sum(jp.square(joint_angles[::3])))

    def _reward_feet_air_time(
            self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum(air_time * first_contact)
        rew_air_time *= (
                math.normalize(commands[:2])[1] > 0.1
        )  # no reward for zero command
        return rew_air_time

    def _reward_feet_contact_time(
            self, contact_time: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Punish contact time.
        rew_contact_time = jp.sum(contact_time)
        rew_contact_time *= (
                math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_contact_time

    def _reward_stand_still( 
            self,
            commands: jax.Array,
            joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.exp(-2 * jp.linalg.norm(joint_angles - self.default_pos[7:])) * (
                math.normalize(commands[:2])[1] < 0.05
        )

    def _reward_foot_slip(self, pipeline_state: State, xd, contact_filt: jax.Array) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self.feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))
   

    def _reward_termination(self, done: jax.Array, step) -> jax.Array:
        return done &(step < self.episode_length)
