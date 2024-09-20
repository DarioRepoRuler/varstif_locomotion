from typing import Optional
from brax.io import torch
import jax


class TorchWrapper:
    """
    Wrapper that converts Jax tensors to PyTorch tensors.
    """

    def __init__(self,
                 env,
                 device: Optional[torch.Device] = None,
                 backend: Optional[str] = None):
        """Creates a gym Env to one that outputs PyTorch tensors."""
        self.device = device
        self._env = env
        self.model = env.model
        self.data = env.data
        self.backend = backend
        self.seed()
        self.state = None
        self.observation_size = self._env.observation_size
        self.single_observation_size = self._env.single_obs_size
        self.privileged_observation_size = self._env.privileged_obs_size
        
        self.action_size = self._env.action_size

        def reset_mjx(key, initial_xy=None, manual_cmd=None):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2, initial_xy, manual_cmd)
            return state, state.obs, state.privileged_obs, key1

        self._reset_jit = jax.jit(reset_mjx, backend=self.backend)

        def step_mjx(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.privileged_obs, state.reward, state.done, info

        self._step_jit = jax.jit(step_mjx, backend=self.backend)

    def reset(self, initial_xy: jax.Array, manual_cmd: jax.Array):
        self.state, obs, privileged_obs, self._key = self._reset_jit(self._key, initial_xy, manual_cmd)
        obs = torch.jax_to_torch(obs, device=self.device)
        privileged_obs = torch.jax_to_torch(privileged_obs, device=self.device)
        return obs, privileged_obs

    def step(self, action):
        action = torch.torch_to_jax(action)
        self.state, obs, privileged_obs, reward, done, info = self._step_jit(self.state, action)
        obs = torch.jax_to_torch(obs, device=self.device)
        privileged_obs = torch.jax_to_torch(privileged_obs, device=self.device)
        reward = torch.jax_to_torch(reward, device=self.device)
        done = torch.jax_to_torch(done, device=self.device)
        info = torch.jax_to_torch(info, device=self.device)
        metrics = torch.jax_to_torch(self.state.metrics, device=self.device)
        return obs, privileged_obs, reward, done, info, metrics

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)
