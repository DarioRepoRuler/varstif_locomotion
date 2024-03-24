from typing import Callable, Optional, Tuple
from envs.common.mjx_env import MjxEnv, State
from brax.envs.base import Wrapper
import jax
from jax import numpy as jp
from mujoco import mjx


def wrap(
        env: MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        randomization_fn: Optional[
            Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
        ] = None,
) -> MjxEnv:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
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
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = AutoResetWrapper(env)
    return env


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.model = env.model
        self.data = env.data
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""
    def __init__(self, env):
        super().__init__(env)
        self.model = env.model
        self.data = env.data

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
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

        return state.replace(pipeline_state=pipeline_state, obs=obs)


class DomainRandomizationVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
            self,
            env: MjxEnv,
            randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(self.env.sys)

    def _env_fn(self, sys: mjx.Model) -> MjxEnv:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._sys_v, state, action
        )
        return res
