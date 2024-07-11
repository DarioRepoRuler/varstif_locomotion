# Code from: https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py
from typing import Callable, Optional, Tuple
from envs.common.mjx_env import MjxEnv, State
from brax.envs.base import Wrapper
import jax
from jax import numpy as jp
from mujoco import mjx


def wrap(
        env: MjxEnv,
        num_envs: 1000,
        #action_repeat: int = 1,
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
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn=randomization_fn, batch_size=num_envs)
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
            initial_xy = jp.repeat(initial_xy, self.batch_size, axis=0)
        return jax.vmap(self.env.reset)(rng, initial_xy,  manual_control)

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
    ):
        super().__init__(env)
        self.batch_size = batch_size
        self._sys_v, self._in_axes = randomization_fn(self.env.sys, batch_size=batch_size)
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
        rng= jp.expand_dims(rng, 0)
        rng = jp.repeat(rng, self.batch_size, axis=0)
        #print(f"Randomised friction from sys_v: {self._sys_v.geom_friction}")
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
def domain_randomize(sys, batch_size: Optional[int] = None):
    """
    Randomizes the mjx.Model in terms of friction, gravitational vector and masses

    Args:
        sys: mjx.Model to be randomized
        batch_size: number of randomizations to perform
    
    Returns:
        sys: randomized mjx.Model
        in_axes: in_axes for jax.vmap

    """

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=-0.5, maxval=0.5)
        friction = sys.geom_friction.at[:, 0].add(friction)
        # gravity
        gravity = jax.random.uniform(key, (3,), minval=-0.2, maxval=0.2)
        #gravity = sys.opt.gravity.at[:].add(jp.array([0.,0.,-10]))
        gravity = sys.opt.gravity.at[:].add(gravity)
        # masses
        masses = jax.random.uniform(key, (sys.body_mass.shape[0],), minval=-0.01, maxval=0.01)
        masses = sys.body_mass.at[:].add(masses)

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
    #print(f"Randomized friction: {friction.shape}")
    #print(f"Randomized gain: {gain.shape}")

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
        return state

    def step(self, state: State, action: jax.Array) -> State:
        ## Original step function
        # if 'steps' in state.info:
        #     steps = state.info['steps']
        #     steps = jp.where(state.done, jp.zeros_like(steps), steps)
        #     state.info.update(steps=steps)
        # state = state.replace(done=jp.zeros_like(state.done))
        # state = self.env.step(state, action)

        # def where_done(x, y):
        #     done = state.done
        #     if done.shape:
        #         done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
        #     return jp.where(done, x, y)

        # pipeline_state = jax.tree.map(
        #     where_done, state.info['first_pipeline_state'], state.pipeline_state
        #     )
        # obs = where_done(state.info['first_obs'], state.obs)
        
        
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

        pipeline_state = jax.tree_map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = where_done(state.info['first_obs'], state.obs)
        # reset information
        state.info['last_act'] = where_done(jp.zeros_like(state.info['last_act']), state.info['last_act'])
        state.info['last_vel'] = where_done(jp.zeros_like(state.info['last_vel']), state.info['last_vel'])
        state.info['step'] = where_done(jp.zeros_like(state.info['step']), state.info['step'])
        # Additional custom reset information
        state.info['feet_air_time'] =  where_done(jp.zeros_like(state.info['feet_air_time']), state.info['feet_air_time'])
        state.info['feet_contact_time'] = where_done(jp.zeros_like(state.info['feet_contact_time']), state.info['feet_contact_time'])
        state.info['last_contact'] = where_done(jp.zeros_like(state.info['last_contact']), state.info['last_contact'])
        #state.info['rewards'] = where_done(jp.zeros_like(state.info['rewards']), state.info['rewards'])
        state.info['step'] = where_done(jp.zeros_like(state.info['step']), state.info['step'])
        state.info['rng'] =where_done(jp.zeros_like(state.info['rng']), state.info['rng'])
        state.info['action_minus_2t'] = where_done(jp.zeros_like(state.info['action_minus_2t']), state.info['action_minus_2t'])
        

        return state.replace(pipeline_state=pipeline_state, obs=obs)