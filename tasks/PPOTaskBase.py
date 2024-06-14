import os
import torch
from torch import nn
from rl_algo.ppo import PPO
from envs.common.mjx_env import MjxEnv
import mujoco

class PPOTaskBase(nn.Module):
    def __init__(self,
                 cfg,
                 env,
                 eval_interval=50,
                 save_interval=50,
                 test_interval=100,
                 wandb_logger=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.test_interval = test_interval

        self.env = env
        self.wandb_logger = wandb_logger

        self.algo = PPO(cfg=self.cfg.policy,
                        num_envs=self.cfg.num_envs,
                        num_actions=self.env.action_size,
                        episode_length=self.cfg.episode_length,
                        num_env_obs=self.env.observation_size,
                        device=self.device)

        self.current_learning_iteration = 0

    def step(self, obs_g, is_training=True):
        """
        Performs action in the environment and returns the next observation.
        """
        if is_training:
            actions = self.algo.act(obs_g)
        else:
            actions = self.algo.act_eval(obs_g)
        next_obs_g, rewards, dones, infos = self.env.step(actions)

        self.algo.process_env_step(obs_g, rewards, dones, infos)

        return next_obs_g, dones, infos['rewards']

    def rollout(self, is_training=True):
        """
        Steps through the environment for one episode and returns the final observation and statistics.
        Rewards are average over the episode.
        """
        episode_infos = {}
        with torch.inference_mode(): # No gradadient computation in torch domain
            obs_g = self.env.reset()
            for i in range(self.cfg.episode_length):
                next_obs_g, dones, rew_info = self.step(obs_g, is_training)
                for key in rew_info.keys():
                    if key not in episode_infos.keys():
                        episode_infos[key] = torch.mean(rew_info[key])
                    else:
                        episode_infos[key] += torch.mean(rew_info[key])
                # update observation
                obs_g = next_obs_g

        for key in episode_infos.keys():
            episode_infos[key] = episode_infos[key] / self.cfg.episode_length
        return obs_g, episode_infos

    def simulate(self, is_training): # Simulates through one episode
        """
        Simulate through one episode and store the statistics.
        """
        # Simulate through one episode
        next_obs_g, episode_infos = self.rollout()
        # get goal conditioned state
        self.algo.compute_returns(next_obs_g)
        # Store the statistics
        stat = self.algo.storage.statistics()

        return stat, episode_infos

    def agent_train_step(self, it):
        self.algo.actor_critic.train() # Switch to training mode
        # Simulate through one episode
        stat, episode_infos = self.simulate(is_training=True)
        mean_value_loss, mean_actor_loss = self.algo.update()

        if self.wandb_logger:
            self.wandb_logger.log({'loss/critic': mean_value_loss,
                                   'loss/actor': mean_actor_loss,
                                   'action_std': self.algo.actor_critic.std.mean()
                                   }, step=it)
            self.wandb_logger.log({'train/avg_traverse': stat["avg_traverse"],
                                   'train/avg_reward': stat["avg_reward"],
                                   'train/dones': stat["dones"],
                                   'train/done_rate': stat["done_rate"],
                                   }, step=it)
            # ---------------- logging reward ------------------------------#
            for key in episode_infos.keys():
                self.wandb_logger.log({f'rewards/train/{key}': episode_infos[key]}, step=it)

    def agent_eval_step(self, it):
        self.algo.actor_critic.eval()
        stat, episode_infos = self.simulate(is_training=False)

        if self.wandb_logger:
            self.wandb_logger.log({'val/avg_traverse': stat["avg_traverse"],
                                   'val/avg_reward': stat["avg_reward"],
                                   'val/dones': stat["dones"],
                                   'val/done_rate': stat["done_rate"],
                                   }, step=it)
            # ---------------- logging reward ------------------------------#
            for key in episode_infos.keys():
                self.wandb_logger.log({f'rewards/val/{key}': episode_infos[key]}, step=it)

        self.algo.storage.clear()

    def train_loop(self, num_learning_iterations, save_dir, ckpt_path=None):
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=True)
        os.makedirs(save_dir, exist_ok=False)

        num_total_iteration = num_learning_iterations + self.current_learning_iteration
        for it in range(self.current_learning_iteration, num_total_iteration):
            self.agent_train_step(it)
            self.current_learning_iteration += 1

            # Attempt on changing the position of the robot
            # if it % 2 ==0:
            #     self.env.x_pos = [2, 3] # Thats not possible
            #     self.env.y_pos = [2, 3]
            #     print(f"Changed position to {self.env.x_pos}, {self.env.y_pos}")

            if it % self.eval_interval == 0:
                self.agent_eval_step(it)
                self.save(os.path.join(save_dir, f'model_{it}.pt'))

        self.current_learning_iteration = num_total_iteration
        self.save(os.path.join(save_dir, f'last.pt'))

    def validate_agent(self, ckpt_path=None):
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)

        self.rollout(is_training=False)
        failure_ids = self.algo.storage.find_failures()
        self.algo.storage.clear()
        return failure_ids

    def test_agent(self, num_iterations, ckpt_path=None):
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        self.algo.actor_critic.eval()
        for _ in range(num_iterations):
            self.simulate(is_training=False)
            self.algo.storage.clear()

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.algo.actor_critic.state_dict(),
            'optimizer_state_dict': self.algo.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.algo.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.algo.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.algo.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.algo.actor_critic.to(device)
        return self.algo.actor_critic.act_inference