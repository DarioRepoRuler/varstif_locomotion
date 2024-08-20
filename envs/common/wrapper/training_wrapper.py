# Code from: https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py
from typing import Callable, Optional, Tuple
from envs.common.mjx_env import MjxEnv, State
from brax.envs.base import Wrapper
import jax
from jax import numpy as jp
from mujoco import mjx
import mujoco

def wrap(
        env: MjxEnv,
        num_envs: 1000,
        #action_repeat: int = 1,
        randomization_args=None,
        randomization_fn: Optional[
            Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
        ] = None,
) -> MjxEnv:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped

      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if randomization_fn is None:
        env = VmapWrapper(env, batch_size=num_envs)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn=randomization_fn, batch_size=num_envs, randomization_args=randomization_args)
    env = AutoResetWrapper(env)
    return env



class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.model = env.model
        self.data = env.data
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, initial_xy: jax.Array, manual_control:bool= False) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
            # Match dimensions for vmaping at axis=0
            initial_xy = jp.expand_dims(initial_xy, 0)
            initial_xy = jp.repeat(initial_xy, self.batch_size, axis=0)
            
            manual_control = jp.array(manual_control, dtype=jp.bool)
            manual_control = jp.expand_dims(manual_control, 0)
            manual_control = jp.repeat(manual_control, self.batch_size,0)
            
        return jax.vmap(self.env.reset)(rng, initial_xy, manual_control)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class DomainRandomizationVmapWrapper(Wrapper):
    """
    Wrapper for domain randomization. It randomizes the environment parameters at every reset call.
    """

    def __init__(
            self,
            env: MjxEnv,
            randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
            batch_size: Optional[int] = None,
            randomization_args=None           
    ):
        super().__init__(env)
        self.batch_size = batch_size
        self._sys_v, self._in_axes = randomization_fn(self.env.sys, batch_size=batch_size, randomization_args=randomization_args)
        #print(self._sys_v)

    def _env_fn(self, sys: mjx.Model) -> MjxEnv:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array, initial_xy: jax.Array, manual_control:bool= False) -> State:
        def reset(sys, rng, initial_xy=initial_xy, manual_control=manual_control):
            env = self._env_fn(sys=sys)
            return env.reset(rng, initial_xy=initial_xy, manual_control=manual_control)
        
        initial_xy = jp.expand_dims(initial_xy, 0)
        initial_xy = jp.repeat(initial_xy, self.batch_size, axis=0)
        rng = jax.random.split(rng, self.batch_size)
        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng, initial_xy=initial_xy )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._sys_v, state, action
        )
        return res

# Code from: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=1K45Kp2ASV9s
def domain_randomize(sys, batch_size: Optional[int] = None, randomization_args=None):
    """
    Randomizes the mjx.Model in terms of friction, gravitational vector and masses

    Args:   
        sys: mjx.Model to be randomized
        batch_size: number of randomizations to perform
    
    Returns:
        sys: randomized mjx.Model
        in_axes: in_axes for jax.vmap

    """
    friction_range = randomization_args.friction_range
    gravity_offset = randomization_args.gravity_offset
    payload_range = randomization_args.payload_range
    hip_mass_range = randomization_args.hip_mass_range

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)
        # friction = jax.random.uniform(key,(3,), minval=0.4, maxval=1.25)
        # friction = sys.geom_friction.at[0, :].set(friction)
        
        # gravity - in z direction
        gravity = jax.random.uniform(key, minval=gravity_offset[0], maxval=gravity_offset[1])
        gravity = sys.opt.gravity.at[2].add(gravity) 
        
        # masses: randomize hips(for COM dial) and trunk
        masses = sys.body_mass 
        hip_masses = jax.random.uniform(key, (4,), minval=hip_mass_range[0], maxval=hip_mass_range[1])
        indices = jp.array([4, 6, 10, 14])
        masses = sys.body_mass.at[indices].add(hip_masses)
        payload = jax.random.uniform(key, minval=payload_range[0], maxval=payload_range[1])
        masses = masses.at[1].add(payload)

        # actuator_ NOT USED!
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            key, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias, gravity, masses
    friction, gain, bias, gravity, masses = jax.vmap(rand)(rng)


    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'opt.gravity': 0,
        'body_mass':0,
        #'actuator_gainprm': 0,
        # 'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'opt.gravity': gravity,
        'body_mass': masses,
        #'actuator_gainprm': gain,
        # 'actuator_biasprm': bias,
    })

    return sys, in_axes
  


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""
    def __init__(self, env):
        super().__init__(env)
        self.model = env.model
        self.data = env.data

    def reset(self, rng: jax.Array, initial_xy: jax.Array, manual_control:bool=False) -> State:
        state = self.env.reset(rng, initial_xy, manual_control=manual_control)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        state.info['first_priviledged_obs'] = state.priviledged_obs
        return state

    def step(self, state: State, action: jax.Array) -> State:       
        
        ## Custom step function
        # if 'step' in state.info:
        #     steps = state.info['step']
        #     steps = jp.where(state.done, jp.zeros_like(steps), steps)
        #     state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # reset done envs
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)
        
        def where_nan(x,y):
            nan = state.info['nan']
            if nan.shape:
                nan = jp.reshape(nan, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(nan, x,y)
        
        pipeline_state = jax.tree_map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        
        for key in state.info['rewards']:
            # Update each value based on the condition
            state.info['rewards'][key] = jp.where(state.info['nan'], jp.array(0.0), state.info['rewards'][key])
        obs = where_nan(state.info['first_obs'], state.obs)
        reward = where_nan(jp.zeros_like(state.reward), state.reward)
        priviledged_obs = where_done(state.info['first_priviledged_obs'], state.priviledged_obs)
        # reset information
        state.info['last_act'] = where_done(jp.zeros_like(state.info['last_act']), state.info['last_act'])
        state.info['last_vel'] = where_done(jp.zeros_like(state.info['last_vel']), state.info['last_vel'])
        state.info['step'] = where_done(jp.zeros_like(state.info['step']), state.info['step'])
        # Additional custom reset information
        state.info['feet_air_time'] =  where_done(jp.zeros_like(state.info['feet_air_time']), state.info['feet_air_time'])
        state.info['feet_contact_time'] = where_done(jp.zeros_like(state.info['feet_contact_time']), state.info['feet_contact_time'])
        state.info['last_contact'] = where_done(jp.zeros_like(state.info['last_contact']), state.info['last_contact'])
        state.info['time_out']= where_done(jp.zeros_like(state.info['time_out']), state.info['time_out'])
        state.info['rng'] =where_done(jp.zeros_like(state.info['rng']), state.info['rng'])
        state.info['action_minus_2t'] = where_done(jp.zeros_like(state.info['action_minus_2t']), state.info['action_minus_2t'])
        

        return state.replace(pipeline_state=pipeline_state, obs = obs, reward= reward,  priviledged_obs=priviledged_obs)