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
            soft_limits=0.99,
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
        self._obs_noise = cfg.obs_noise
        self._kick_vel = cfg.kick_vel
        self.is_training = cfg.is_training
        if not self.is_training:
            self.cmd_x = cfg.cmd_x
            self.cmd_y = cfg.cmd_y
            self.cmd_yaw = cfg.cmd_ang
        self.soft_limits = soft_limits
        self.single_obs_size = 53 # defined in _get_obs
        # Randomization ranges:
        self.x_pos = [-3, 3]
        self.y_pos = [-3, -2] 
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

        geometries = [
            "base_0", "base_1", "base_2", 
            "FR_hip", "FR_thigh", "FR_calf_0", "FR_calf_1",  
            "FL_hip", "FL_thigh", "FL_calf_0", "FL_calf_1", 
            "RR_hip", "RR_thigh", "RR_calf_0", "RR_calf_1", 
            "RL_hip", "RL_thigh", "RL_calf_0", "RL_calf_1"
            ]

        feet_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_names
        ]
        feet__geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in feet_names
        ]
        foot_body = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        foot_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in foot_body
        ]

        body_geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, f) for f in geometries
        ]

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, 'floor')
        
        hip_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in hip_body
        ]
        
        assert not any(id_ == -1 for id_ in body_geom_id), 'Body Geom not found.'
        self.body_geom_id = jp.array(body_geom_id)
        
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

        #print(f"Feet Geom  IDs: {self.feet_geom_id}")
        #print(f"Body Geom IDs: {self.body_geom_id}")
        #print(f"Body size: {self.body_geom_id.size}")
        #print(f"Geom Torso body IDs: {self.torso_body_id}")
        #print(f"Torso ID: {self._torso_idx}")
       
        #print(f"Floor ID: {floor_id}") # Floor ID: 0 
        
        # # # interesting the foot and the torso seem to have the same value
        #for i in range(60):
            #name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM.value, i)
            #print(f"Name of Geom {i}: {name}")
            # name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i)
            # print(f"Name of Body {i}: {name}")
        

        # Scaling of the rewards
        # These rewards are from the tutorial: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
        # And partially adapted
        self.reward_scales = {
            # From turtoial
            'tracking_lin_vel': 1.0, 
            'tracking_ang_vel': 1.0, 
            "lin_vel_z": 0.5, #-2.0, adapted
            "ang_vel_xy": 0.5, #-0.05, adapted
            "orientation": 1.0, #-5.0, adapted
            "torques": 0.00001, #-0.0002, #adapted
            "smooth_rate": 0.02, #action_rate from tutorial
            'feet_air_time': 0.5,
            'feet_contact_time': -0.2,
            'termination': -10.0,
            'stand_still': 0.5, #-0.5, # adapted
            "foot_slip": -0.1,
            # Additional self created
            "action_rate": 0.02,
            "action_rate2": 0.02,
            "abduction": 0.1,
            #"foot_clearance": 0.5
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
        expected_total_cont=221
        geom_temp = jp.zeros((expected_total_cont,2))
        conn_indices = jp.zeros((num_collisions,1), dtype=int)
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
        num_collisions = 21
        expected_total_cont=221
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

    def _resample_commands(self, rng: jax.Array) -> jax.Array:
        # Define constraints for the commands# From turtoial
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s] 

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
        
        # Just for test purposes!
        if not self.is_training:
            new_cmd = jp.array([self.cmd_x, self.cmd_y,self.cmd_yaw])


        return new_cmd

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3, rng4, rng5 ,rng6, rng7 = jax.random.split(rng, 8)
        
        reset_pos = self.default_pos
        # Get random x,y coordinates for the robot
        x_pos = self.x_pos
        y_pos = self.y_pos
        reset_x=jax.random.uniform(rng1, (1,), minval=x_pos[0], maxval=x_pos[1])
        reset_y=jax.random.uniform(rng2, (1,), minval=y_pos[0], maxval=y_pos[1])

        # Get random theta and 
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
        reset_pos = reset_pos.at[0:7].set(jp.array([reset_x[0], reset_y[0], 0.27, q1[0], q2[0], q3[0], q4]))        

        data = self.pipeline_init(reset_pos, jp.zeros((self.sys.nv,)))
        reward, done, zero = jp.zeros(3)
        command = self._resample_commands(rng3)     

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
        }
        obs_history = jp.zeros(self.num_history * self.single_obs_size)  # store num_history steps of history
        obs = self._get_obs(data, state_info, obs_history)
        obs = obs.at[self.single_obs_size:].set(jp.tile(obs[:self.single_obs_size], self.num_history-1))
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        return State(pipeline_state=data,
                     obs=obs,
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

        if self.control_mode == "P":
            target_dof_pos = jp.clip(self.action_scale * action + self.default_pos[7:],
                                     a_min=self.lower_limits, a_max=self.upper_limits)
            err = target_dof_pos - dof_pos
            torques = self.p_gains * err - self.d_gains * dof_vel
            torques = jp.clip(torques, a_min=-self.torque_limits, a_max=self.torque_limits)
        elif self.control_mode == "T":
            torques = unscale(self.action_scale * action, lower=-self.torque_limits, upper=self.torque_limits)
            torques = jp.clip(torques, a_min=-self.torque_limits, a_max=self.torque_limits)
        else:
            raise RuntimeError("control model: P|T")
        return torques

    def kick_robot(self, state: State, rng: jp.ndarray):
        """
        Adjusts the velocity of the robot base to simulate a kick, for every push_interval steps.

        Args:
            state: current state
            rng: random key
        """
        push_interval = 100
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
        """
        rng, cmd_rng, kick_noise = jax.random.split(state.info['rng'], 3)
        # kick robot
        state = self.kick_robot(state, kick_noise)

        # ACTUAL STEP (in physics)
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # ----------------- Compute rewards --------------- #
        x, xd = self._pos_vel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel

        # foot contact data based on z-position
        foot_pos = data.site_xpos[self.feet_site_id]  # pytype: disable=attribute-error

        # Original foot contact management
        # foot_contact_z = foot_pos[:, 2] - self._foot_radius
        # contact = foot_contact_z < 1e-3  # a mm or less off the floor
        # contact_filt_mm = contact | state.info['last_contact']
        # contact_filt_cm = (foot_contact_z < 1e-2) | state.info['last_contact']
        # first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        # state.info['contact'] = contact_filt_mm
        # state.info['feet_air_time'] += self.dt
        # state.info['feet_contact_time'] += self.dt

        # Contact based foot management
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
        state.info['contact'] = contact_filt_mm
        state.info['feet_air_time'] += self.dt
        state.info['feet_contact_time'] += self.dt


        #jax.debug.print('Foot contacts: {x}', x=contact)

        # Contact based foot management
        foot_contacts = jp.zeros((4),dtype=int)
        foot_contacts = foot_contacts.at[0:4].set(self.get_foot_contacts(data)[0:4,0].astype(int))
        foot_floor_dist = data.contact.dist[foot_contacts]
        contact = foot_floor_dist< 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_floor_dist < 1e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['contact'] = contact_filt_mm
        state.info['feet_air_time'] += self.dt
        state.info['feet_contact_time'] += self.dt
        

        # check termination
        done = self._check_terminate(data, x)


        # get reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd), # implement potential based reward?
            'ang_vel_xy': self._reward_ang_vel_xy(x, xd),
            'orientation': self._reward_orientation(x), # implement potential based reward?
            'torques': self._reward_torques(data.ctrl),  # pytype: disable=attribute-error
            'smooth_rate': self._reward_smooth_rate(joint_vel, state.info['last_vel']),
            'stand_still': self._reward_stand_still( # implement potential based reward?
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
            'foot_slip': self._reward_foot_slip(xd, contact_filt_cm),
            'termination': self._reward_termination(done),
            'action_rate': self.action_rate(action, state.info['last_act']),
            'action_rate2': self.action_rate2(action, state.info['last_act'], state.info['action_minus_2t']),
            'abduction': self.abduction(joint_angles),
            #'foot_clearance': self._reward_foot_clearance(xd, contact_filt_cm, foot_pos[:, 2], state.info['feet_air_time'], state.info['command'])
        }
        rewards = {
            k: v * self.reward_scales[k] for k, v in rewards.items()
        }

        reward = sum(rewards.values())

        # state management
        state.info['feet_air_time'] *= ~contact_filt_mm # bitwise negation
        state.info['feet_contact_time'] *= contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step']+= 1
        state.info['rng'] = rng
        state.info['action_minus_2t'] = state.info['last_act'] 
        state.info['last_act'] = action
        state.info['last_vel'] = data.qvel
        state.info['last_qpos'] = data.qpos
        state.info['foot_pos_z'] = foot_pos[:, 2]
        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        # observation
        obs = self._get_obs(data, state.info, state.obs)
        
        done = jp.float32(done)
        # jax.debug.print('Observations: {x}', x=obs)
        # #jax.debug.print('Foot contacts: {x}', x=foot_contacts)
        # #jax.debug.print('Is done?: {x}', x=done)
        # jax.debug.print('Reward: {x}', x=reward)

        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

        return state

    def _get_obs(self,
                  data: mjx.Data,
                  state_info: dict[str, Any],
                  obs_history: jax.Array
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
        # Inverse torso quaternion
        inv_torso_rot = math.quat_inv(x.rot[0])
        torso_z = data.qpos[2:3]

        local_v = math.rotate(xd.vel[0], inv_torso_rot)
        local_w = math.rotate(xd.ang[0], inv_torso_rot) # yaw rate at index 2
        
        proj_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot)      # projected gravity

        # Observation space dimension: 1+6+3+12+12+12+4+3 53 #old calculation
        obs = jp.concatenate([
            torso_z,
            0.1 * jp.concatenate([local_v, local_w]),  # yaw rate at index 5
            proj_gravity,
            data.qpos[7:],
            0.1 * data.qvel[6:],
            state_info['last_act'], 
            state_info['contact'], #added
            #jp.array([state_info['step']]), #added  
            state_info['command'] # 3 commands(can add more->observation space update needed)
        ])

        assert obs.shape[0] == self.single_obs_size, f"obs.shape: {obs.shape}"
        # Stack observations through time
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

    def _check_terminate(self, data: mjx.Data, x) -> bool:
        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        # check if robot is falling, dot product of rotated upward direction and actual up. Less than 0 means falling.
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        # Check if compliant with limits
        done |= jp.any(data.qpos[7:] < self.lower_limits) 
        done |= jp.any(data.qpos[7:] > self.upper_limits)
        # Old termination: based on z-height
        #done |= data.xpos[self._torso_idx, 2] < self.min_z
        # New termination: If body touches the ground
        body_contacts = jp.zeros((21),dtype=int)
        body_contacts = body_contacts.at[:].set(self.get_body_contacts(data)[:,0])
        done |= jp.any(data.contact.dist[body_contacts] < 0.0)

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
        ang_vel_error = jp.abs(commands[2] - base_ang_vel[2])
        return jp.exp(-4 * ang_vel_error) #change to sigma

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.exp(-6*jp.abs(xd.vel[0, 2]))

    def _reward_ang_vel_xy(self, x: Transform, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))

        return jp.exp(-4*jp.linalg.norm(base_ang_vel[:2])) 

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.exp(-6*jp.linalg.norm(rot_up[:2])) # change this

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.exp(-0.01*jp.linalg.norm(torques))
    
    ## Related to smoothness of the actions:
    def _reward_smooth_rate( # to be continued...(why the velocities?)
            self, joint_vel: jax.Array, last_vel: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jp.exp(-0.4*jp.linalg.norm(joint_vel - last_vel))

    def action_rate(self, action: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.exp(-0.05*jp.sum(jp.power(action - last_act, 2)))
    
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

    def _reward_foot_slip(
            self, xd: Motion, contact_filt: jax.Array
    ) -> jax.Array:
        foot_indices = self.foot_body_id - 1
        foot_vel = xd.take(foot_indices).vel
        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))
        

    def _reward_termination(self, done: jax.Array) -> jax.Array:
        return done
