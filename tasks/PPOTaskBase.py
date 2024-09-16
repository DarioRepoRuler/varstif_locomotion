import os
import torch
from torch import nn
from rl_algo.ppo import PPO
from envs.common.mjx_env import MjxEnv
import mujoco
from utils.graphs_gen import eval_graph, create_multiple_box_plots, create_power_energy_bar_chart
import jax.numpy as jp
import jax

# import threading
# from pynput import keyboard as pynput_keyboard
# import time

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
        self.control_mode = cfg.env.control_mode
        self.device = cfg.device
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.test_interval = test_interval
        self.initial_xy = jp.array([0, 0]) #starting in the first quadrant
        self.env = env
        self.wandb_logger = wandb_logger
        self.curriculum = cfg.curriculum
        self.view_env_id = 0
        if self.control_mode == 'P' or self.control_mode == 'T':
            num_actions = 12
        elif self.control_mode == 'VIC_1':
            num_actions= 15
        elif self.control_mode == 'VIC_2':
            num_actions= 16
        elif self.control_mode == 'VIC_3':
            num_actions= 24

        self.algo = PPO(cfg=self.cfg.policy,
                        num_envs=self.cfg.num_envs,
                        num_actions=num_actions,
                        episode_length=self.cfg.timesteps_per_rollout,
                        num_single_obs = self.env.observation_size // self.cfg.env.num_history_actor,
                        num_env_obs=self.env.observation_size,
                        num_priv_obs=self.env.privileged_observation_size*self.cfg.env.num_history_critic,
                        control_mode=self.control_mode,
                        device=self.device)

        self.current_learning_iteration = 0
        self.level = 0
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_control = self.cfg.env.manual_control.enable)


        # # Start the keyboard listener thread
        if self.cfg.viz:
            self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener)
            self.keyboard_listener_thread.daemon = True
            self.keyboard_listener_thread.start()

    def on_press(self, key):
        try:
            if key == pynput_keyboard.Key.up:
                print(f"Up arrow pressed Viewed env: {self.view_env_id}")
                self.view_env_id = min(self.cfg.num_envs, self.view_env_id+1)
            elif key == pynput_keyboard.Key.down:
                self.view_env_id = max(0, self.view_env_id-1)
                print(f"Down arrow pressed,  Viewed env: {self.view_env_id}")
        except AttributeError:
            pass
    
    def keyboard_listener(self):
        with pynput_keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def step(self, obs_g, privileged_obs_g, is_training=True):
        """
        Performs action in the environment and returns the next observation. Everything outside this function will not directly
        affect the environment or the learning process.
        """
        
        #print(f"Observations: {obs_g.shape}")
        if torch.isnan(obs_g).any():
            print(f"observation shape: {obs_g.shape}")
            print("Nan in obs_g!! Coming from simulation then...")
            print(f"In observation: {torch.where(torch.isnan(obs_g))}")
            print(f"Last action: {self.algo.storage.actions[-1][torch.where(torch.isnan(obs_g))[0]]}")
            print(f"Action before: {self.algo.storage.actions[-2][torch.where(torch.isnan(obs_g))[0]]}")
            print(f"Observation: {obs_g[torch.where(torch.isnan(obs_g))]}")

        if is_training:
            actions = self.algo.act(obs_g, privileged_obs_g)
        else:
            actions = self.algo.act_eval(obs_g, privileged_obs_g)
        if torch.isnan(actions).any():
            print(f"Action: {actions}")
        
        if self.cfg.viz:
            next_obs_g, next_priv_obs_g,rewards, dones, infos, metrics = self.env.step(actions, env_id=self.view_env_id)
        else:
            next_obs_g, next_priv_obs_g,rewards, dones, infos, metrics = self.env.step(actions)

        self.algo.process_env_step(obs_g, privileged_obs_g,rewards, dones, infos)

        return next_obs_g, next_priv_obs_g, dones, infos, metrics

    def rollout(self,it, is_training=True):
        """
        Steps through the environment for one episode and returns the final observation and statistics.
        Episode infos represent the average reward
        """
        episode_infos = {}
        # Gather evaluation information only in test mode
        eval_infos = {}
        foot_pos_z = []
        q_vel = []
        power = []
        cmd = []
        with torch.inference_mode():
            pos_x = torch.zeros(self.cfg.timesteps_per_rollout, device=self.device, dtype=torch.float32)
            time_out = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.bool)

            for i in range(self.cfg.timesteps_per_rollout):
                next_obs_g, next_priv_obs_g, dones, info, metrics = self.step(self.obs, self.obs_priv, is_training)
                #print(f"Last x: {info['last_qpos'][0,0]}")
                rew_info= info['rewards']
                #print(f"Rewards: {rew_info}")
                pos_x[i] = info['last_qpos'][0,0]
                time_out |= info['time_out'] > 0
                #print(f"time out : {info['time_out']}")
                for key in rew_info.keys():
                    if key not in episode_infos.keys():
                        episode_infos[key] = torch.mean(rew_info[key]) #mean over all envs
                    else:
                        episode_infos[key] += torch.mean(rew_info[key])

                #print(f"Foot pos z: {info['foot_pos_z']}")                
                if not is_training:
                    foot_pos_z.append(info['foot_pos_z'])
                    q_vel.append(info['last_vel'])
                    cmd.append(info['command'])
                    power.append(metrics['power'])


                # update observation
                self.obs = next_obs_g
                self.priv_obs = next_priv_obs_g

        if not is_training:
            foot_pos_z= torch.stack(foot_pos_z)
            q_vel = torch.stack(q_vel)
            cmd = torch.stack(cmd)
            power = torch.stack(power)
            eval_infos['foot_pos_z'] = foot_pos_z
            eval_infos['q_vel'] = q_vel
            eval_infos['cmd'] = cmd
            eval_infos['power'] = power

            #print(f" Torch foot z:{foot_pos_z}")
            # print(f"Foot pos z shape: {foot_pos_z.shape}")
            # print(f"Power shape: {power.shape}")
        
        #print(f"Foot pos z: {foot_pos_z}")

        self.pos_x = pos_x

        for key in episode_infos.keys():
            episode_infos[key] = episode_infos[key] / self.cfg.timesteps_per_rollout
        #print(f"Termination reward: {episode_infos['termination']}")
        #print(f"Time outs:{torch.sum(time_out)}")
        print(f"Rewards infos: {episode_infos}")
        if episode_infos['termination']>0.0:
            print(f"Episode infos: {episode_infos}")
            print(f"Termination reward: {episode_infos['termination']}")
        episode_infos['time_outs'] = time_out
        return self.obs, self.obs_priv, dones, episode_infos, eval_infos

    def simulate(self,it, is_training=True): # Simulates through one episode
        """
        Simulate through one episode and store the statistics.
        """
        # Simulate through one episode
        next_obs_g, next_priv_obs_g, dones, episode_infos, eval_infos = self.rollout(it,is_training=is_training)
        # get goal conditioned state
        self.algo.compute_returns(next_priv_obs_g, dones)
        # Show statistics
        stat = self.algo.storage.statistics()

        return stat, episode_infos, eval_infos

    def agent_train_step(self, it):
        self.algo.actor_critic.train() # Switch to training mode
        # Simulate through one episode
        stat, episode_infos, _ = self.simulate(it,is_training=True)
        mean_value_loss, mean_actor_loss, mean_state_estimation_loss = self.algo.update()

        loss_info = {'loss/critic': mean_value_loss,
                     'loss/actor': mean_actor_loss,
                     'loss/state_estimation': mean_state_estimation_loss,
                     'action_std': self.algo.actor_critic.std_action.mean()
                     }
        train_info = {'train/avg_traverse': stat["avg_traverse"],
                      'train/avg_reward': stat["avg_reward"],
                      'train/dones': stat["dones"],
                      'train/done_rate': stat["done_rate"],
                      }
        print(loss_info, train_info)
        if self.wandb_logger:
            self.wandb_logger.log(loss_info, step=it)
            self.wandb_logger.log(train_info, step=it)
            # ---------------- logging reward ------------------------------#
            for key in episode_infos.keys():
                if 'time_out' not in key:
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

    def update_level(self):
        """
        Check the level of the agent.
        """
        max_distance =max(self.pos_x[:] - self.pos_x[0])
        print(f"Max distance: {max_distance}")
        if max_distance>0.7:
            self.level += 1
        print(f"Level: {self.level}")
        if self.level==0:
            self.initial_xy = jp.array([-2, -2.5])
        elif self.level==1:
            self.initial_xy = jp.array([2, -2.5])
        elif self.level==2:
            self.initial_xy = jp.array([2, 2.5])
        elif self.level==3:
            self.initial_xy = jp.array([-2, 2.5])
        print(f"Set next position to: {self.initial_xy}")

    def train_loop(self, num_learning_iterations, save_dir, ckpt_path=None):
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=True)
        os.makedirs(save_dir, exist_ok=False)

        if self.cfg.curriculum:
            self.initial_xy = jp.array([-2., -2.5])

        num_total_iteration = num_learning_iterations + self.current_learning_iteration
        for it in range(self.current_learning_iteration, num_total_iteration):
            #print(f"Current position: {self.initial_xy}")
            print(f"Epoch: {it}")
            self.agent_train_step(it)
            self.current_learning_iteration += 1
            
            if it % self.eval_interval == 0:
                print(f"Evaluation at epoch: {it}")
                self.agent_eval_step(it, is_training=True)
                self.save(os.path.join(save_dir, f'model_{it}.pt'))

            if self.curriculum and it % (self.eval_interval+1) ==0:
                # Update level according to paper(Learn to walk in minutes)
                print(f"System level update at epoch: {it}")
                temp = self.cfg.env.manual_control
                self.cfg.env.manual_control = True # walk straight
                self.agent_eval_step(it, is_training=True)
                self.save(os.path.join(save_dir, f'model_{it}.pt'))
                # Move to different terrain if successfull
                self.update_level()
                self.cfg.env.manual_control = temp # TODO: improve layout and yaml

        self.current_learning_iteration = num_total_iteration
        self.save(os.path.join(save_dir, f'last.pt'))

    def validate_agent(self, ckpt_path=None): # not used
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        it = 0
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
        single_obs_size = self.env.observation_size // self.cfg.env.num_history_actor
        power_overall = []
        energy_overall = []
        for it in range(num_iterations):
            #print(f"iteration: {it} ")
            #print(f"configuration of env: {self.cfg.env}")
            stat, episode_info, eval_infos = self.simulate(it,is_training=False)
            
            print(f"Eval infos power shape: {eval_infos['power'].shape}")
            print(f"Eval infos foot pos z shape: {eval_infos['foot_pos_z'].shape}")
            print(f"Eval infos q vel shape: {eval_infos['q_vel'].shape}")

            power_overall.append(torch.mean(eval_infos['power']))
            energy_overall.append(torch.sum(eval_infos['power']*0.02)/3600)
            # # Evaluate test run: Create box plots for the tracking error + draw foot z position graph            
            # # Optionally it is also possible to store the values in csv files with the functions save_tensors_to_csv and load_tensor_from_csv
            # total_tracking_error = torch.zeros((self.cfg.timesteps_per_rollout, 3), device=self.device, dtype=torch.float32)
            # ang_tracking_error = torch.abs(stat['observations'][:,0,6] - stat['observations'][:,0,single_obs_size-1])
            # #print(f"Observed ang. vel: {stat['observations'][:,0,6]}")
            # #print(f"Observed lin. vel: {eval_infos['q_vel'][:,:2]}")
            # #print(f"Commanded vel: {eval_infos['cmd']}")
            # #print(f"Command ang: {stat['observations'][:,0,single_obs_size-1]}")
            # #print(f"Command lin: {stat['observations'][:,0,single_obs_size-3:single_obs_size-1]}")

            # lin_tracking_error = torch.abs(eval_infos['q_vel'][:,:2] - stat['observations'][:,0,single_obs_size-3:single_obs_size-1])
            # #print(f"Lin tracking error: {lin_tracking_error}")
            # total_tracking_error[:,0] = ang_tracking_error
            # total_tracking_error[:,1:3] = lin_tracking_error
            # lin_tracking_error = torch.linalg.norm(lin_tracking_error, dim=1)
            # #print(f"Total tracking error: {total_tracking_error} with shape {total_tracking_error.shape}")
            # total_tracking_error = torch.linalg.norm(total_tracking_error, dim=1)
            # #print(f"Total tracking error norm: {total_tracking_error} with shape {total_tracking_error.shape}")
            # create_multiple_box_plots([total_tracking_error, lin_tracking_error, ang_tracking_error], ['total error', 'linear vel error', 'angular vel error'], 'tracking_test_error')

            eval_graph([eval_infos['foot_pos_z'][:,0,0], eval_infos['foot_pos_z'][:,0,1], 
                        eval_infos['foot_pos_z'][:,0,2], eval_infos['foot_pos_z'][:,0,3]], 
                        ['FR_foot','FL_foot','RR_foot','RL_foot'], f'Foot z position test run {it}', 0.02)
            
            self.algo.storage.clear()
        

        power_overall = torch.mean(torch.stack(power_overall))
        energy_overall = torch.mean(torch.stack(energy_overall))

        y_lim=(500, 10)
        print(f"Mean Power[W]: {power_overall}, Energy overall[Wh]: {energy_overall}")
        print(f"Y lim: {y_lim}")
        create_power_energy_bar_chart(title="Power/Energy Consumption", names=["position baseline"], power=power_overall, energy=energy_overall , filename="power_energy_bar_chart", y_lim=y_lim)    


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