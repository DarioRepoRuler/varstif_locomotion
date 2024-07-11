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
        self.action_size = self._env.action_size

        def reset_mjx(key, initial_xy=None):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2, initial_xy)
            return state, state.obs, key1

        self._reset_jit = jax.jit(reset_mjx, backend=self.backend)

        def step_mjx(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step_jit = jax.jit(step_mjx, backend=self.backend)

    def reset(self, initial_xy: jax.Array):
        self.state, obs, self._key = self._reset_jit(self._key, initial_xy)
        return torch.jax_to_torch(obs, device=self.device)

    def step(self, action):
        action = torch.torch_to_jax(action)
        self.state, obs, reward, done, info = self._step_jit(self.state, action)
        obs = torch.jax_to_torch(obs, device=self.device)
        reward = torch.jax_to_torch(reward, device=self.device)
        done = torch.jax_to_torch(done, device=self.device)
        info = torch.jax_to_torch(info, device=self.device)
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)
