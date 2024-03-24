import jax
import mujoco
from brax import math
from brax.base import Transform, Motion
from typing import Any, Dict, Tuple, Union
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src.forward import _integrate_pos
from mujoco.mjx._src.support import jac
from mujoco.mjx._src import smooth
from envs.common.mjx_env import MjxEnv, State
from envs.common.helper import unscale
from pathlib import Path
import os


class UnitreeEnv(MjxEnv):
    def __init__(
            self,
            cfg,
            model_path,
            soft_limits=0.99,
    ):
        mj_model = mujoco.MjModel.from_xml_path(os.path.join(Path.home(), 'projects/RL4Quadrupeds/envs/resources/', model_path))
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
        self.soft_limits = soft_limits
        self.single_obs_size = 49
        # set up robot properties
        self._setup()

    def _setup(self):
        self._foot_radius = 0.023

        self._torso_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk')

        feet_site = ['FR', 'FL', 'RR', 'RL']
        feet_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self.feet_site_id = jp.array(feet_site_id)

        hip_body = ['FR_hip', 'FL_hip', 'RR_hip', 'RL_hip']
        hip_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in hip_body
        ]
        assert not any(id_ == -1 for id_ in hip_body), 'Hip not found.'
        self.hip_body_id = jp.array(hip_body_id)

        foot_body = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        foot_body_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, i) for i in foot_body
        ]
        assert not any(id_ == -1 for id_ in foot_body), 'Foot not found.'
        self.foot_body_id = jp.array(foot_body_id)

        self.reward_scales = {
            'tracking_lin_vel': 1.0,
            'tracking_ang_vel': 1.0,
            "lin_vel_z": 0.5,
            "ang_vel_xy": 0.5,
            "orientation": 1.0,
            "torques": 0.00001,
            "smooth_rate": 0.02,
            'feet_air_time': 0.5,
            'feet_contact_time': -0.2,
            'termination': -10.0,
            'stand_still': 0.5,
            "foot_slip": -0.1,
        }

    def _resample_commands(self, rng: jax.Array) -> jax.Array:
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
        return new_cmd

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        data = self.pipeline_init(self.default_pos, jp.zeros((self.sys.nv,)))
        reward, done, zero = jp.zeros(3)
        command = self._resample_commands(rng3)

        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(18),
            'foot_acc': jp.zeros(12),
            'command': command,
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
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng, kick_noise = jax.random.split(state.info['rng'], 3)
        # kick robot
        state = self.kick_robot(state, kick_noise)

        # physics step
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # ----------------- compute rewards --------------- #
        x, xd = self._pos_vel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel

        # foot contact data based on z-position
        foot_pos = data.site_xpos[self.feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 1e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
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
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(x, xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(data.ctrl),  # pytype: disable=attribute-error
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
            'foot_slip': self._reward_foot_slip(xd, contact_filt_cm),
            'termination': self._reward_termination(done),
        }
        rewards = {
            k: v * self.reward_scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values())

        # state management
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['feet_contact_time'] *= contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng
        state.info['last_act'] = action
        state.info['last_vel'] = data.qvel

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        # observation
        obs = self._get_obs(data, state.info, state.obs)

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

        return state

    def _get_obs(self,
                  data: mjx.Data,
                  state_info: dict[str, Any],
                  obs_history: jax.Array
                 ) -> jp.ndarray:

        x, xd = self._pos_vel(data)
        inv_torso_rot = math.quat_inv(x.rot[0])
        torso_z = data.qpos[2:3]
        local_v = math.rotate(xd.vel[0], inv_torso_rot)
        local_w = math.rotate(xd.ang[0], inv_torso_rot)
        proj_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot)      # projected gravity

        # 1+6+3+12+12+12+3 = 49
        obs = jp.concatenate([
            torso_z,
            0.1 * jp.concatenate([local_v, local_w]),  # yaw rate
            proj_gravity,
            data.qpos[7:],
            0.1 * data.qvel[6:],
            state_info['last_act'],
            state_info['command']
        ])

        assert obs.shape[0] == self.single_obs_size, f"obs.shape: {obs.shape}"
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

    def _check_terminate(self, data: mjx.Data, x):
        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(data.qpos[7:] < self.lower_limits)
        done |= jp.any(data.qpos[7:] > self.upper_limits)
        done |= data.xpos[self._torso_idx, 2] < self.min_z
        # done |= data.xpos[self._torso_idx, 2] > self.max_z

        return done

    # ------------ reward functions---------------- #
    def _reward_tracking_lin_vel(
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-2 * lin_vel_error)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
            self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.abs(commands[2] - base_ang_vel[2])
        return jp.exp(-10 * ang_vel_error)

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.exp(-2*jp.abs(xd.vel[0, 2]))

    def _reward_ang_vel_xy(self, x: Transform, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        return jp.exp(-2*jp.linalg.norm(base_ang_vel[:2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.exp(-2*jp.linalg.norm(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.exp(-0.01*jp.linalg.norm(torques))

    def _reward_smooth_rate(
            self, joint_vel: jax.Array, last_vel: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jp.exp(-0.1*jp.linalg.norm(joint_vel - last_vel))

    def _reward_feet_air_time(
            self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum(air_time * first_contact)
        rew_air_time *= (
                math.normalize(commands[:2])[1] > 0.05
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
