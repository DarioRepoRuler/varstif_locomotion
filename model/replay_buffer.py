import torch


class ReplayBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.progress = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 num_obs,
                 num_actions,
                 num_robots=1,
                 device='cpu'):
        self.device = device
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_robots = num_robots

        self.step = 0

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, num_obs, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.progress = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) # represents the time step in the episode in percentage

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)

        # Maybe something for logging?

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.progress[self.step].copy_(transition.progress.view(-1, 1))

        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lamb):
        advantage = 0
        for step in reversed(range(self.step)):
            if step == self.step - 1:
                next_values = last_values
            else:
                next_values = self.values[step+1]
            next_is_not_terminate = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminate * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminate * gamma * lamb * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages[:self.step] = self.returns[:self.step] - self.values[:self.step]
        self.advantages[:self.step] = (self.advantages[:self.step] - self.advantages[:self.step].mean()) / (self.advantages[:self.step].std() + 1e-8)

    def statistics(self):
        """
        This function returns the statistics of the rollout buffer.
        This includes the average traverse, average reward, number of dones, and the done rate.
        """
        row_idx, col_idx = torch.where(self.dones[..., 0] == 1)
        if len(row_idx) == 0:
            avg_traverse = self.step
        else:
            avg_traverse = torch.mean(self.progress[row_idx, col_idx])

        row_idx, col_idx = torch.where(self.progress[..., 0] > 0.9999)

        # dones = torch.unique(col_idx, return_counts=False)
        num_dones = col_idx.shape[0]

        avg_reward = torch.mean(self.rewards[:self.step])
        stat = {
            "observations": self.observations[:self.step],
            "avg_traverse": avg_traverse,
            "avg_reward": avg_reward,
            "dones": num_dones,
            "done_rate": num_dones / self.num_envs * self.num_robots,
        }

        return stat

    def mini_batch_generator(self, num_batches, num_epochs=8):
        batch_size = self.num_envs * self.step // num_batches # integer floor division

        obs_g = self.observations[:self.step].flatten(0, 1)
        actions = self.actions[:self.step].flatten(0, 1)
        values = self.values[:self.step].flatten(0, 1)
        returns = self.returns[:self.step].flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob[:self.step].flatten(0, 1)
        advantages = self.advantages[:self.step].flatten(0, 1)
        old_mu = self.mu[:self.step].flatten(0, 1)
        old_sigma = self.sigma[:self.step].flatten(0, 1)

        for epoch in range(num_epochs):
            indices = torch.randperm(num_batches * batch_size, requires_grad=False, device=self.device)
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_idx = indices[start:end]

                obs_batch = obs_g[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (obs_batch, actions_batch, target_values_batch, advantages_batch,
                       returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch)
