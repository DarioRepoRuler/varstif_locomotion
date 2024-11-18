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

# For debugging set this:
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
        
        self.cfg = cfg
        self.force_kick_counter = self.cfg.force_kick_duration / self.dt
        # Control parameters
        self.track_traj = (cfg.manual_control.task == "track trajectory")
        self.soft_limits = soft_limits

        # Set obs sizes(size is adapted in PPOTaskBase) 
        self.single_obs_size = self.cfg.single_obs_size
        self.privileged_obs_size = self.cfg.single_obs_size_priv

        # Variable impedance control parameters
        if self.cfg.control_mode == "VIC_1": # for hip,thigh and knee
            self.action_shape = self.action_size-2 + 3
            
        elif self.cfg.control_mode == "VIC_2": # for every leg
            self.action_shape = self.action_size-2 + 4
            
        elif self.cfg.control_mode == "VIC_3": # for every leg
            self.action_shape = self.action_size-2 + 12
            
        elif self.cfg.control_mode == "VIC_4":
            self.action_shape = self.action_size-2 + 7
        else:
            self.action_shape = self.action_size-2
        
        # Specify Gains for PD controller for each joint
        self.p_gain = cfg.control.p_gain
        self.d_gain = cfg.control.d_gain

        self.min_z = 0.15
        # set up robot properties
        self._setup()

    def _setup(self):

        self._foot_radius = 0.023
        self._torso_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk')
        
        # Define body and geometry names
        feet_names = ['FR', 'FL', 'RR', 'RL'] #
        foot_body = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        lower_leg_body=['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']
        hip_body = ['FR_hip', 'FL_hip', 'RR_hip', 'RL_hip']
        torso_bodies = ['base_mirror_0', 'base_mirror_1', 'base_mirror_2', 'base_mirror_3', 'base_mirror_4']
        
        # Define body and termination geometries
        body_geometries = self.cfg.body_penalty_geom
        terminate_geometries = self.cfg.terminate_geoms 
        
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
    
    def get_contacts(self, data, num_collisions, geom_indices)->jax.Array:
        
        expected_total_cont=self.cfg.contacts.total + 3*self.cfg.contacts.total*self.terminate_map
        geom_temp = jp.zeros((expected_total_cont,2))
        conn_indices = jp.zeros((num_collisions,1), dtype=int)
        geom_temp = geom_temp.at[0:expected_total_cont,0:2].set(data.contact.geom[0:expected_total_cont,0:2])
        body_mask = jp.zeros((expected_total_cont),dtype=int)
        body_cont = jp.zeros((expected_total_cont),dtype=int)
        connection_index = jp.zeros((1),dtype=int)
        ground_mask = jp.zeros((expected_total_cont),dtype=int)

        ground_mask = ground_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, 0)[:,0])
        
        for i in range(num_collisions):
            body_mask=body_mask.at[0:expected_total_cont].set(jp.isin(geom_temp, geom_indices[i])[:,1])
            body_cont = body_cont.at[0:expected_total_cont].set(body_mask*ground_mask)
            connection_index= connection_index.at[:].set(jp.where(body_cont, size=1)[0])
            conn_indices = conn_indices.at[i,0].set(connection_index[0])        
        return conn_indices
    
    def _resample_commands(self, rng: jax.Array) -> jax.Array:
        # Define constraints for the commands# From turtoial
        lin_vel_x = self.cfg.control_range['cmd_x'] # min max [m/s]
        lin_vel_y = self.cfg.control_range['cmd_y'] # min max [m/s]
        ang_vel_yaw = self.cfg.control_range['cmd_ang']  # min max [rad/s] 

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
        kp_factor = jax.random.uniform(kp_rng, (1,), minval=self.cfg.domain_rand.kp_range[0], maxval=self.cfg.domain_rand.kp_range[1])
        kd_factor = jax.random.uniform(kd_rng, (1,), minval=self.cfg.domain_rand.kd_range[0], maxval=self.cfg.domain_rand.kd_range[1])
        motor_strength = jax.random.uniform(motor_strength_rng, (1,), minval=self.cfg.domain_rand.motor_strength_range[0], maxval=self.cfg.domain_rand.motor_strength_range[1])

        reset_pos = self.default_pos
        reset_x= initial_xy[0]+jax.random.uniform(rng1, (1,), minval=self.cfg.reset_pos_x[0], maxval=self.cfg.reset_pos_x[1])
        reset_y= initial_xy[1]+jax.random.uniform(rng2, (1,), minval=self.cfg.reset_pos_y[0], maxval=self.cfg.reset_pos_y[1])

        # Randomisation of the drop off position
        theta = self.cfg.reset_theta # in rad
        a_x = self.cfg.reset_a_x
        a_y = self.cfg.reset_a_y
        a_x=jax.random.uniform(rng6, (1,), minval=a_x[0], maxval=a_x[1])
        a_y=jax.random.uniform(rng7, (1,), minval=a_y[0], maxval=a_y[1])
        a_x, a_y = a_x/jp.linalg.norm(jp.array([a_x, a_y])), a_y/jp.linalg.norm(jp.array([a_x, a_y]))
        
        # Randomise the initial orientation
        theta = jax.random.uniform(rng5, (1,), minval=theta[0], maxval=theta[1])      
        q1 = jp.cos(theta/2)
        q2 = a_x*jp.sin(theta/2)
        q3 = a_y*jp.sin(theta/2)
        q4 = 0

        # Reset position only
        reset_pos = reset_pos.at[0:2].set(jp.array([reset_x[0], reset_y[0]]))        
        #reset_pos = reset_pos.at[4:7].set(jp.array([q1[0], q2[0], q3[0], q4]))
        
        # Get initial state
        data = self.pipeline_init(reset_pos, jp.zeros((self.sys.nv,)))
        reward, done, zero = jp.zeros(3)
        command_rand = self._resample_commands(rng3)
        command = jp.where(self.cfg.manual_control.enable, manual_cmd, command_rand)     
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
            'rewards': {reward_: jp.array(0.) for reward_ in self.cfg.reward_scales.keys()},
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
        obs_history = jp.zeros(self.cfg.num_history_actor * self.single_obs_size)  # store num_history steps of history
        privileged_obs_history = jp.zeros(self.cfg.num_history_critic*self.privileged_obs_size)
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

        if self.cfg.control_mode == "P" or self.cfg.control_mode == "VIC_1" or self.cfg.control_mode == "VIC_2" or self.cfg.control_mode == "VIC_3" or self.cfg.control_mode == "VIC_4":
            target_dof_pos = jp.clip(self.compute_target_dof(action),
                                a_min=self.lower_limits, a_max=self.upper_limits)
            
            p_gains = self.compute_stiffness(action)
            
            if self.cfg.control_mode == "VIC_1" or self.cfg.control_mode == "VIC_2" or self.cfg.control_mode == "VIC_3" or self.cfg.control_mode == "VIC_4":
                d_gains = 0.2*jp.sqrt(p_gains)
            else:
                d_gains = self.d_gains
            if self.cfg.domain_rand.randomisation: # and self.cfg.control_mode =="P":
                p_gains = p_gains * actuator_param[0]
                d_gains = d_gains * actuator_param[1]
            torques = p_gains * (target_dof_pos - dof_pos) - d_gains * dof_vel
        elif self.cfg.control_mode == "T":
            torques = unscale(self.cfg.action_scale * action[:12], lower=-self.torque_limits, upper=self.torque_limits)
        
        if self.cfg.domain_rand.randomisation:
            torques = jp.clip(torques*actuator_param[2], a_min=-self.torque_limits, a_max=self.torque_limits)
        else:
            torques = jp.clip(torques, a_min=-self.torque_limits, a_max=self.torque_limits)

        # Add the force kick to the torques
        torques = jp.concatenate([torques, action[-2:]])

        return torques
    
    def compute_target_dof(self, action: jp.ndarray) -> jp.ndarray:
        scaled_action = self.cfg.action_scale * action[:12]
        # additional scaling for hip scale reduction
        scaled_action = scaled_action.at[0].multiply(self.cfg.hip_scale)
        scaled_action = scaled_action.at[3].multiply(self.cfg.hip_scale)
        scaled_action = scaled_action.at[6].multiply(self.cfg.hip_scale)
        scaled_action = scaled_action.at[9].multiply(self.cfg.hip_scale)

        target_dof_pos = scaled_action + self.default_pos[7:]
        return target_dof_pos

    def compute_stiffness(self, action: jp.ndarray) -> jp.ndarray:
        """
        Compute the stiffness for each joint based on the action

        Args:
            action: the action to be taken

        Returns:    
            p_gains: the proportional gains (12) for each joint 
        """
        action = self.cfg.action_stiff_scale * action
        if self.cfg.control_mode == "P" or self.cfg.control_mode == "T":
            return self.p_gains
        
        elif self.cfg.control_mode == "VIC_1":
            action_stiff = unscale(jp.tile(action[12:12+3],4), self.cfg.control.stiff_range[0], self.cfg.control.stiff_range[1])
        elif self.cfg.control_mode == "VIC_2":
            action_stiff = unscale(jp.repeat(action[12:12+4],3), self.cfg.control.stiff_range[0], self.cfg.control.stiff_range[1])
        elif self.cfg.control_mode == "VIC_3":
            action_stiff = unscale(action[12:12+12], self.cfg.control.stiff_range[0], self.cfg.control.stiff_range[1])
        elif self.cfg.control_mode == "VIC_4":
            stiff_leg = jp.tile(unscale(action[12:12+4], self.cfg.control.stiff_range[0], self.cfg.control.stiff_range[1]), 3).reshape(3,4)
            stiff_joint = unscale(action[12+4:12+4+3], self.cfg.control.stiff_range[0], self.cfg.control.stiff_range[1])
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
        push_interval = self.cfg.push_interval
        kick_theta = jax.random.uniform(rng, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0  #& (state.info['step'] > 0)
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self.cfg.kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})
        # update st ate info
        state.info['kick'] = kick
        return state
       
    def force_kick_robot(self, state: State, rng: jp.ndarray):
        # This is intended to be used for evaluation only
        if self.cfg.enable_force_kick:
            rng_kick, rng_theta, rng_impulse = jax.random.split(rng, 3)
            # Push randomly
            kick_theta = jax.random.uniform(rng_theta, maxval= 2* jp.pi)
            kick_force = jax.random.uniform(rng_kick, minval=self.cfg.kick_force[0], maxval=self.cfg.kick_force[1])
            kick_impulse = jax.random.uniform(rng_impulse, minval=self.cfg.force_kick_impulse[0], maxval=self.cfg.force_kick_impulse[1])
            kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)]) * kick_force 
            kick_condition = jp.logical_and(jp.mod(state.info['step'], self.cfg.force_kick_interval)==0, state.info['step']>1 )
            # Get the random values at kick interval
            state.info['force_kick'] = jp.where(kick_condition, kick, state.info['force_kick'])
            state.info['kick_theta']=jp.where(kick_condition, kick_theta, state.info['kick_theta'])
            state.info['kick_force_magnitude'] = jp.where(kick_condition, kick_force , 0.0) #state.info['kick_force_magnitude']
            
            # Hold the same value for a few steps
            state.info['kick_counter_initial']= jp.where(jp.logical_and(self.cfg.impulse_force_kick, kick_condition), ((kick_impulse/state.info['kick_force_magnitude']) / self.dt).astype(int), self.force_kick_counter)
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
        #action = jp.clip(action, a_min=-1.0, a_max=1.0)
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

        p_gains = self.compute_stiffness(self.cfg.action_stiff_scale*action)
        target_dof = self.compute_target_dof(action)
        data = self.pipeline_step(data0, action_kick, actuator_param) #passed data is action as angle -> convert to torque in mjx
              
        # ----------------- POST Physics step --------------- #
        # Here we 1.)extract the state information, 2.)check termination, 3.) calculate the reward and 4.) get observations 
        
        #1.) Extract the state information: positions/velocities, joint angles/velocities, foot contacts, etc.
        x, xd = self._pos_vel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]  

        # Foot contact data
        foot_pos = data.site_xpos[self.feet_site_id]  # pytype: disable=attribute-error
        #state.info['des_foot_height'] = target_foot_height
        
        ## Foot contacts
        foot_contacts = jp.zeros((len(self.feet_geom_id)),dtype=int)
        foot_contacts = foot_contacts.at[0:4].set(self.get_contacts(data, len(self.feet_geom_id), geom_indices= self.feet_geom_id)[0:4,0].astype(int))
        foot_floor_dist = jp.zeros((4),dtype=float)
        foot_floor_dist = foot_floor_dist.at[:].set(data.contact.dist[foot_contacts])
        #jax.debug.print('Foot floor distance: {x}', x=foot_floor_dist)
        
        ## general contact management
        contact = foot_floor_dist < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']        
        contact_filt_cm = (foot_floor_dist < 1e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        # for observations
        state.info['contact'] = contact_filt_mm
        state.info['feet_air_time'] += self.dt
        state.info['feet_contact_time'] += self.dt

        # 2.) Check termination
        done = self._check_terminate(data, x, state.info['step'])
        #com = data.subtree_com[self._torso_idx]
        
        # 3.) Calculate reward
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
            'stand_still': self._reward_stand_still( 
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            
            'foot_slip': self._reward_foot_slip(data, xd, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step'], data=data),
            'action_rate': self.action_rate(action, state.info['last_act']),
            #'action_smoothnes': self.action_rate2(action, state.info['last_act'], state.info['action_minus_2t']),
            
            #'rew_pos_limits': self._reward_pos_limits(joint_angles),
            #'rew_stiff_limits': self._reward_stiff_limits(p_gains),
            'rew_limits': self._reward_limits(target_dof, p_gains), 
            'rew_acceleration': self._reward_acceleration(joint_vel, state.info['last_vel'][6:]),
            #'rew_velocity': self._reward_velocity(joint_vel),
            'rew_collision': self._reward_collision(data),
            'rew_power': self._reward_power(data.ctrl[:12], joint_vel),
            #'rew_power_distro': self._reward_power_distro(data.ctrl[:12], joint_vel),
            # Feet posture
            'hip': self.rew_hip(joint_angles, state.info['command']),
            # Variable impedance control
            'rew_joint_track': self._reward_joint_track(joint_angles, action[:12]),
            #'rew_base_height': self._reward_base_height(data.qpos[2]),
            'rew_foot_clearance': self._reward_foot_clearance( xd , foot_pos[:, 2], 0.09),
            #'rew_com': self._reward_com(com, foot_pos=foot_pos),
        }
        rewards = {
            k: rewards[k] * v for k, v in self.cfg.reward_scales.items()
        }
        
        # jax.debug.print('Step: {x}', x=state.info['step'])
        # jax.debug.print('Done: {x}', x=done)
        # jax.debug.print('Termination reward: {x}', x=rewards['termination'])
        
        #Reward clipping like in unitree rl
        reward = jp.clip(sum(rewards.values())*self.dt , 0.0, 10000.0)


        # J = self._get_jacobian(data, data.qpos)
        # jax.debug.print('Jacobian: {x}', x=J[:,6:])
        # jax.debug.print('jacobian shape: {x}', x=J[:,6:].shape)
        # K = jp.array([1.0, 1.0, 1.0]*4)
        # jax.debug.print('stiffness: {x}', x=K.shape)
        # jax.debug.print('joint stiffness: {x}', x=J[:,6:].T @ K @ J[:,6:])
        # #jax.debug.print('K: {x}', x=K)

        # state management
        state.info['feet_air_time'] *= ~contact_filt_mm # bitwise negation
        state.info['feet_contact_time'] *= contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step']+= 1
        state.info['gait_idx'] = jp.remainder(state.info['step']*self.dt, 1.0)
        state.info['nan']= jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        state.info['time_out'] = state.info['step'] > self.cfg.episode_length
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

        state.metrics['target_dof_pos'] = jp.clip(target_dof, a_min=self.lower_limits, a_max=self.upper_limits)
        state.metrics['dof_pos'] = data.qpos[7:]
        state.metrics['glob_pos'] = data.qpos[:2]
        state.metrics['command'] = state.info['command']
        state.metrics['p_gains'] = p_gains   

        # sample new command
        state.info['command'] = jp.where(
            jp.logical_and( state.info['step'] % self.cfg.sample_command_interval == 0, jp.logical_not(self.cfg.manual_control.enable)), #normally a ~ would be sufficient but it somehow converts the boolean into -2 and thus this and does not work anymore
            self._resample_commands(cmd_rng),
            state.info['command'],
            )
        
        # track the trajectory (for evaluation)
        state.info['command'] = jp.where(
            jp.logical_and(self.cfg.manual_control.enable, self.track_traj),
            state.info['trajectory'][state.info['step'].astype(int), :],
            state.info['command'],
        )
        # reset the step counter when done
        # state.info['step'] = jp.where(
        # (state.info['step'] > self.cfg.episode_length), 0, state.info['step']
        # )
        
        
        state.metrics.update(state.info['rewards'])

        # observation
        obs, privileged_obs = self._get_obs(data, state.info, state.obs, state.privileged_obs, obs_rng=obs_rng)
        state.info['nan'] |= jp.isnan(obs).any() | jp.isnan(privileged_obs).any() | jp.isnan(reward).any() | jp.isnan(done).any()
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
        
        # Calculating the local measurable velocities
        local_v = math.rotate(xd.vel[0], inv_torso_rot)
        local_w = math.rotate(xd.ang[0], inv_torso_rot) # yaw rate at index 2
        proj_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot)      # projected gravity
        
        
        # foot_pos = data.site_xpos[self.feet_site_id]

        # Local foot position
        # J = self._get_jacobian(data, data.qpos)
        # foot_indices = self.foot_body_id - 1
        # foot_transform = x.take(foot_indices)
        # foot_transform_local = foot_transform.vmap().to_local(x.take(jp.array([0, 0, 0, 0])))
        # foot_pos_local = foot_transform_local.pos.reshape(-1)
        # foot_vel_local = J @ data.qvel

        # Orientation quaternion
        # quaternion = data.qpos[3:7] 
        # rpy = math.quat_to_euler(quaternion)
        # torso_z = data.qpos[2:3]

        # Observation, if anything is changed here, the noise vector must be adjusted accordingly
        obs = jp.concatenate([
            self.cfg.normalization.local_v_scale*jp.array(local_v),
            self.cfg.normalization.local_w_scale*jp.array(local_w),
            proj_gravity,
            data.qpos[7:]-self.default_pos[7:],  
            self.cfg.normalization.joint_vel_scale *data.qvel[6:],
            state_info['last_act'], 
            self.cfg.normalization.command_scale * state_info['command'],
        ])

        obs = jp.clip(obs, -100.0, 100.0)
        # Privileged observation: passed to the critic
        privileged_obs = jp.concatenate([
            state_info['kp_factor'],
            state_info['kd_factor'],
            state_info['motor_strength'],
            jp.array([self.sys.geom_friction[0, 0]]),
            jp.array([self.sys.body_mass[1]]),
            state_info['contact'],
            state_info['force_kick'],
            obs
        ])

        assert obs.shape[0] == self.single_obs_size, f"obs.shape: {obs.shape}"
        assert privileged_obs.shape[0] == self.privileged_obs_size, f"privileged_obs.shape {privileged_obs.shape}"

        # Add noise to the observation(not privileged), this has to be altered if the observation space changes
        noise_vec = jax.random.uniform(obs_rng, (self.single_obs_size,), minval=-1., maxval=1.)
        noise_vec = noise_vec.at[:3].multiply(self.cfg.domain_rand.local_v_noise*self.cfg.normalization.local_v_scale)
        noise_vec = noise_vec.at[3:6].multiply(self.cfg.domain_rand.local_w_noise*self.cfg.normalization.local_w_scale)
        noise_vec = noise_vec.at[6:9].multiply(self.cfg.domain_rand.gravity_noise)
        noise_vec = noise_vec.at[9:21].multiply(self.cfg.domain_rand.joint_noise)
        noise_vec = noise_vec.at[21:33].multiply(self.cfg.domain_rand.joint_vel_noise*self.cfg.normalization.joint_vel_scale)
        noise_vec = noise_vec.at[33:].multiply(0.0)
        obs = jp.where(self.cfg.domain_rand.randomisation, obs+noise_vec, obs)

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
        terminate_contacts = jp.zeros((len(self.terminate_geom_id)),dtype=int)
        terminate_contacts = terminate_contacts.at[:].set(self.get_contacts(data, len(self.terminate_geom_id), geom_indices= self.terminate_geom_id)[:,0])
        done |= jp.any(data.contact.dist[terminate_contacts] < 0.0)
        # Terminate if body height not high enough
        #done |= data.xpos[self._torso_idx, 2] < self.min_z
        # Termination in case of finite terrain
        done_map = (jp.abs(data.qpos[0]) > 10.0) | (jp.abs(data.qpos[1]) > 10.0) | (jp.abs(data.qpos[2]) < -3.0)
        done |= done_map*self.terminate_map
        done |= step > self.cfg.episode_length
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
        lin_vel_error_new = jp.abs(1-local_vel[:2]/(commands[:2] +1.e-6))
        #jax.debug.print('portion: {x}', x=local_vel[:2]/(commands[:2] +1.e-6))
        #jax.debug.print('Lin vel error: {x}', x=lin_vel_error_new)
        #lin_vel_reward = jp.exp(-4 * lin_vel_error) # change to sigma
        #lin_vel_reward_low = jp.exp(-16 * lin_vel_error) # change to sigma
        #lin_vel_final = jp.where(jp.linalg.norm(commands[:2])<0.4, lin_vel_reward_low, lin_vel_reward)
        lin_vel_reward_new = jp.sum(jp.exp(-4 * lin_vel_error_new)) # change to sigma
        return lin_vel_reward_new

    def _reward_tracking_ang_vel(
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        ang_vel_error_new = jp.abs(1-base_ang_vel[2]/(commands[2] +1.e-6))
        
        ang_vel_rew = jp.exp(-4 * ang_vel_error) #change to sigma
        ang_vel_rew_new = jp.sum(jp.exp(-4 * ang_vel_error_new))
        return ang_vel_rew_new #change to sigma

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
        pos_out_of_bounds = -(pos - self.lower_limits).clip(max=0.)
        pos_out_of_bounds += (pos - self.upper_limits).clip(min=0.)
        return jp.sum(pos_out_of_bounds)
    
    def _reward_stiff_limits(self, stiff: jax.Array) -> jax.Array:
        stiff_out_of_bounds = -(stiff - self.cfg.control.stiff_range[0]*self.p_gain).clip(max=0.)
        stiff_out_of_bounds += (stiff - self.cfg.control.stiff_range[1]*self.p_gain).clip(min=0.)
        return jp.sum(stiff_out_of_bounds)
    
    def _reward_limits(self, pos: jax.Array, stiff: jax.Array) -> jax.Array:
        # position and stiffness is passed unclipped
        return self._reward_pos_limits(pos) + self._reward_stiff_limits(stiff)
    
    def _reward_collision(self, data) -> jax.Array:
        body_contacts = jp.zeros(len(self.body_geom_id),dtype=int)
        body_contacts = body_contacts.at[:].set(self.get_contacts(data, len(self.body_geom_id), geom_indices= self.body_geom_id)[:,0])
        #jax.debug.print('Body distance: {x}', x=data.contact.dist[body_contacts])
        return 1.0*jp.sum(data.contact.dist[body_contacts] < 0.05)
        
    ## Related to smoothness of the actions:
    def action_rate(self, action: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(action - last_act))
    
    def action_rate2(self, action: jax.Array, last_act: jax.Array, action_minus_2t:jax.Array) -> jax.Array:
        return jp.exp(-0.05*jp.sum(jp.power(action-2*last_act+action_minus_2t,2)))
    
    def rew_hip(
            self, joint_angles: jax.Array, commands: jax.Array
    ):
        reward_hip = jp.exp(-4*jp.sum(jp.square(joint_angles[::3])))
        reward_hip *=(jp.any(jp.abs(commands) > 0.05))
        return reward_hip
    

    def _reward_feet_air_time(
            self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            #math.normalize(commands[:])[1] > 0.05
            jp.any(jp.abs(commands) > 0.05)
        )
        return rew_air_time
    
    def _reward_foot_clearance(
            self, xd: Motion, foot_z: jax.Array, des_z = 0.09
    ) -> jax.Array:
        foot_indices = self.foot_body_id - 1
        foot_vel = xd.take(foot_indices).vel
        #rew_clearance = jp.sum(jp.square(foot_z - des_z) * jp.sqrt(jp.linalg.norm(foot_vel[:, :2], axis=1 )))
        rew_clearance = jp.sum(jp.min(jp.clip(foot_z - des_z, min=0.0, max=100.), 0) * jp.linalg.norm(foot_vel[:, :2], axis=1 ))
        return rew_clearance

    def _reward_stand_still( 
            self,
            commands: jax.Array,
            joint_angles: jax.Array,
    ) -> jax.Array:
    
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
        default_termination = done & (step < self.cfg.episode_length)
        terrain_termination = done & (step < self.cfg.episode_length) & self.terminate_map*((jp.abs(data.qpos[0]) < 10.0) & (jp.abs(data.qpos[1]) < 10.0) & (jp.abs(data.qpos[2]) > -3.0))
        return jp.where(self.terminate_map, terrain_termination, default_termination)

    def _reward_power(self, torques: jax.Array, qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques)*jp.abs(qvel))

    def _reward_power_distro(self, torques: jax.Array, qvel: jax.Array) -> jax.Array:
        power = jp.abs(torques) * jp.abs(qvel)

        leg_power = jp.array([jp.sum(power[:3]), jp.sum(power[3:6]) , \
                              jp.sum(power[6:9]), jp.sum(power[9:12])])
        var = jp.var(leg_power)
        return var
    
    def _reward_joint_track(self, joint_angles: jax.Array, action: jax.Array) -> jax.Array:
        
        target_dof_pos = jp.clip(self.compute_target_dof(action),
                                a_min=self.lower_limits, a_max=self.upper_limits)
        
        return jp.sum(jp.square(joint_angles-target_dof_pos))   
    
    def _reward_base_height(self, base_z: jax.Array) -> jax.Array:
        target_height = 0.27
        return jp.square(base_z-target_height)

    def _reward_com(self, com: jax.Array, foot_pos: jax.Array) -> jax.Array:
        des_com = foot_pos[:,:2].mean(axis=0)
        return jp.sum(jp.square(com[:2]-des_com))