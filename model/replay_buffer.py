import torch
from utils.utils import split_and_pad_trajectories


class ReplayBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.priv_obs = None # Privileged observations for the critic 
            #self.priv_estimations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            
            self.progress = None
            self.values = None
            

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 num_obs,
                 num_priv_obs,
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

        ## Advanced
        self.priv_obs = torch.zeros(num_transitions_per_env, num_envs, num_priv_obs, device=self.device) # Privileged observations for the critic
        #self.priv_estimations = torch.zeros(num_transitions_per_env, num_envs, num_priv_obs, device=self.device) # Estimations of the privileged observations
        
        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)


    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        self.priv_obs[self.step].copy_(transition.priv_obs) ## Privileged observations for the critic
        #self.priv_estimations[self.step].copy_(transition.priv_estimations) ## estimations of the privileged observations
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
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step+1]
            next_is_not_terminate = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminate * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminate * gamma * lamb * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def statistics(self):
        """
        This function returns the statistics of the rollout buffer.
        This includes the average traverse, average reward, number of dones, and the done rate.
        """
        
        # Shape of dones: timesteps_per_rollout, num_envs, 1
        #print(f"Self Dones: {self.dones[ ..., 0]}")
        # done = self.dones
        # done[-1]=1
        #print(f"Done: {done[... ,0]}")
        #print(f"Check where dones:{self.dones[..., 0] == 1} ")
        #print(f"Check where done shape: {(self.dones[..., 0] == 1).shape}")
        #print(f"Dones in statistics: {self.dones[... ,0]}")
        row_idx, col_idx = torch.where(self.dones[..., 0] == 1)
        
        print(f"Row Index: {row_idx} and col index: {col_idx}")
        if len(row_idx) == 0:
            avg_traverse = self.step
        else:
            avg_traverse = torch.mean(self.progress[row_idx, col_idx])

        #print(f"Progress: {self.progress[..., 0].shape}")
        row_idx, col_idx = torch.where(self.progress[..., 0] > 0.9999)

        dones = torch.unique(col_idx, return_counts=False)
        num_dones = dones.shape[0]
        #print(f"Number of dones: {num_dones}")
        #print(f"Row Index: {row_idx}")
        #print(f"Col index: {col_idx}")
        #num_dones = col_idx.shape[0]
        
        print(f"Number of dones: {num_dones}")

        avg_reward = torch.mean(self.rewards[:self.step])
        stat = {
            "observations": self.observations[:self.step],
            "priv_obs": self.priv_obs[:self.step],
            "avg_traverse": avg_traverse,
            "avg_reward": avg_reward,
            "dones": num_dones,
            "done_rate": num_dones / self.num_envs * self.num_robots,
        }

        return stat

    def mini_batch_generator(self, num_batches, num_epochs=8):
        batch_size = self.num_envs * self.step // num_batches # integer floor division
        #print(f"Mini batch size: {batch_size}")
        obs_g = self.observations[:self.step].flatten(0, 1)       
        actions = self.actions[:self.step].flatten(0, 1)
        values = self.values[:self.step].flatten(0, 1)
        returns = self.returns[:self.step].flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob[:self.step].flatten(0, 1)
        advantages = self.advantages[:self.step].flatten(0, 1)
        old_mu = self.mu[:self.step].flatten(0, 1)
        old_sigma = self.sigma[:self.step].flatten(0, 1)

        priv_obs_g = self.priv_obs[:self.step].flatten(0, 1)
        #priv_obs_estimations_g = self.priv_estimations[:self.step].flatten(0, 1)

        for epoch in range(num_epochs):
            indices = torch.randperm(num_batches * batch_size, requires_grad=False, device=self.device)
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_idx = indices[start:end]

                obs_batch = obs_g[batch_idx]
                priv_obs_batch = priv_obs_g[batch_idx]
                #priv_obs_estimations_batch = priv_obs_estimations_g[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (obs_batch, priv_obs_batch, actions_batch, target_values_batch, advantages_batch,
                       returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch)

    # copied from legged gym
    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.priv_obs is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.priv_obs, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj