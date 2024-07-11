import os
import torch
from torch import nn
from rl_algo.ppo import PPO
from envs.common.mjx_env import MjxEnv
import mujoco
from utils.graphs_gen import eval_graph, create_multiple_box_plots
import jax.numpy as jp
import jax

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
        self.initial_xy = jp.array([[0, 0]]) #starting in the first quadrant
        self.env = env
        self.wandb_logger = wandb_logger
        self.curriculum = cfg.curriculum
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
        return next_obs_g, dones, infos

    def rollout(self,it, is_training=True):
        """
        Steps through the environment for one episode and returns the final observation and statistics.
        Episode infos represent the average reward
        """
        episode_infos = {}
        # Gather evaluation information only in test mode
        eval_infos = None
        #initial_xy=jp.array([[-2, -2]])
        if not is_training:
            eval_infos={'foot_pos_z': torch.zeros((self.cfg.episode_length, 4), device=self.device, dtype=torch.float32), 
                        'q_vel': torch.zeros((self.cfg.episode_length,18), device=self.device, dtype=torch.float32), 
                        'cmd': torch.zeros((self.cfg.episode_length, 3), device=self.device, dtype=torch.float32)}
        # Based on the learning iteration the initial position of the agent is changed 
        if self.curriculum:
            if it==0:
                self.initial_xy=jp.array([[-2, -2.5]])
            elif it==1000:
                self.initial_xy=jp.array([[2, -2.5]])
            elif it==2000:
                self.initial_xy=jp.array([[2, 2.5]])
            elif it==3000:
                self.initial_xy=jp.array([[-2, 2.5]])

        
        with torch.inference_mode(): # No gradadient computation in torch domain
            #print(f"Initial xy(in rollout function): {self.initial_xy}")
            obs_g = self.env.reset(initial_xy=self.initial_xy)
            for i in range(self.cfg.episode_length):
                next_obs_g, dones, info = self.step(obs_g, is_training)
                rew_info= info['rewards']
                for key in rew_info.keys():
                    if key not in episode_infos.keys():
                        episode_infos[key] = torch.mean(rew_info[key])
                    else:
                        episode_infos[key] += torch.mean(rew_info[key])
                if not is_training:
                    eval_infos['foot_pos_z'][i] = info['foot_pos_z']
                    eval_infos['q_vel'][i] = info['last_vel']
                    eval_infos['cmd'][i] = info['command']

                # update observation
                obs_g = next_obs_g
        for key in episode_infos.keys():
            episode_infos[key] = episode_infos[key] / self.cfg.episode_length
        return obs_g, episode_infos, eval_infos

    def simulate(self,it, is_training=True): # Simulates through one episode
        """
        Simulate through one episode and store the statistics.
        """
        # Simulate through one episode
        next_obs_g, episode_infos, eval_infos = self.rollout(it,is_training=is_training)
        # get goal conditioned state
        self.algo.compute_returns(next_obs_g)
        # Store the statistics
        stat = self.algo.storage.statistics()

        return stat, episode_infos, eval_infos

    def agent_train_step(self, it):
        self.algo.actor_critic.train() # Switch to training mode
        # Simulate through one episode
        stat, episode_infos, _ = self.simulate(it,is_training=True)
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

    def agent_eval_step(self, it, is_training=True): # this function can be called via test or train
        self.algo.actor_critic.eval()
        stat, episode_infos, _ = self.simulate(it,is_training=is_training)
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

            if it % self.eval_interval == 0:
                self.agent_eval_step(it, is_training=True)
                self.save(os.path.join(save_dir, f'model_{it}.pt'))

        self.current_learning_iteration = num_total_iteration
        self.save(os.path.join(save_dir, f'last.pt'))

    def validate_agent(self, ckpt_path=None): # not used
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        it =0
        self.rollout(it,is_training=False)
        failure_ids = self.algo.storage.find_failures()
        self.algo.storage.clear()
        return failure_ids

    def test_agent(self, num_iterations, ckpt_path=None):
        """
        Test loop for the agent.
        """
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        self.algo.actor_critic.eval()
        #eval_results = []
        single_obs_size = self.env.observation_size // self.cfg.env.num_history
        for it in range(num_iterations):
            stat, episode_info, eval_infos = self.simulate(it,is_training=False)

            # Evaluate test run: Create box plots for the tracking error + draw foot z position graph            
            # Optionally it is also possible to store the values in csv files with the functions save_tensors_to_csv and load_tensor_from_csv
            total_tracking_error = torch.zeros((self.cfg.episode_length, 3), device=self.device, dtype=torch.float32)
            ang_tracking_error = torch.abs(stat['observations'][:,0,6] - stat['observations'][:,0,single_obs_size-1])
            #print(f"Observed ang. vel: {stat['observations'][:,0,6]}")
            #print(f"Observed lin. vel: {eval_infos['q_vel'][:,:2]}")
            #print(f"Commanded vel: {eval_infos['cmd']}")
            #print(f"Command ang: {stat['observations'][:,0,single_obs_size-1]}")
            #print(f"Command lin: {stat['observations'][:,0,single_obs_size-3:single_obs_size-1]}")

            lin_tracking_error = torch.abs(eval_infos['q_vel'][:,:2] - stat['observations'][:,0,single_obs_size-3:single_obs_size-1])
            #print(f"Lin tracking error: {lin_tracking_error}")
            total_tracking_error[:,0] = ang_tracking_error
            total_tracking_error[:,1:3] = lin_tracking_error
            lin_tracking_error = torch.linalg.norm(lin_tracking_error, dim=1)
            #print(f"Total tracking error: {total_tracking_error} with shape {total_tracking_error.shape}")
            total_tracking_error = torch.linalg.norm(total_tracking_error, dim=1)
            #print(f"Total tracking error norm: {total_tracking_error} with shape {total_tracking_error.shape}")
            create_multiple_box_plots([total_tracking_error, lin_tracking_error, ang_tracking_error], ['total error', 'linear vel error', 'angular vel error'], 'tracking_test_error')

            eval_graph([eval_infos['foot_pos_z'][:,0], eval_infos['foot_pos_z'][:,1], 
                        eval_infos['foot_pos_z'][:,2], eval_infos['foot_pos_z'][:,3]], 
                        ['FR_foot','FL_foot','RR_foot','RL_foot'], f'Foot z position test run {it}', 0.02)
            
            self.algo.storage.clear()

        #print(f"Evaluation results: {eval_results}")


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