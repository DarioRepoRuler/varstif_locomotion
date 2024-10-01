import os
import torch
from torch import nn
from rl_algo.ppo import PPO
from envs.common.mjx_env import MjxEnv
import mujoco
from utils.graphs_gen import eval_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position

import jax.numpy as jp
import jax
from envs.robots.go2_env import GO2Env
from envs.common.wrapper import _create_env


import threading
from pynput import keyboard as pynput_keyboard
from time import time


from utils.helper_traj import create_combined_command

class PPOTaskBase(nn.Module):
    def __init__(self,
                 cfg,
                 #env,
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
        self.manual_cmd = jp.array([cfg.env.manual_control.cmd_x, cfg.env.manual_control.cmd_y, cfg.env.manual_control.cmd_ang])
        self.high_score_avg_reward =0.0
        self.env = self.init_env(self.cfg.scene_xml)
        self.wandb_logger = wandb_logger
        self.curriculum = cfg.curriculum
        
        if not cfg.env.is_training:
            self.result_file_name = cfg.result_name
        self.view_env_id = 0
        if self.control_mode == 'P' or self.control_mode == 'T':
            num_actions = 12
        elif self.control_mode == 'VIC_1':
            num_actions= 15
        elif self.control_mode == 'VIC_2':
            num_actions= 16
        elif self.control_mode == 'VIC_3':
            num_actions= 24
        elif self.control_mode == 'VIC_4':
            num_actions= 12+7

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

        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd = self.manual_cmd)


        # # Start the keyboard listener thread
        if self.cfg.viz:
            self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener)
            self.keyboard_listener_thread.daemon = True
            self.keyboard_listener_thread.start()

    def init_env(self, scene_xml=None):
        env = _create_env(GO2Env(self.cfg.env, scene_xml=scene_xml), num_envs=self.cfg.num_envs, device=self.cfg.device, viz=self.cfg.viz, domain_cfg=self.cfg.env.domain_rand)
        return env

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
        
        #time_start = time()

        if self.cfg.viz:
            next_obs_g, next_priv_obs_g,rewards, dones, infos, metrics = self.env.step(actions, env_id=self.view_env_id)
        else:
            next_obs_g, next_priv_obs_g,rewards, dones, infos, metrics = self.env.step(actions)
        #time_end = time()
        #time_diff = time_end - time_start
        #print(f"Time for step: {time_diff}")
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
        cmd = []
        eval_metrics = []
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
                    cmd.append(info['command'])
                    eval_metrics.append(metrics)
                    

                # update observation
                self.obs = next_obs_g
                self.priv_obs = next_priv_obs_g
        
        combined_metrics = {}
        if not is_training:
            cmd = torch.stack(cmd)
            eval_infos['cmd'] = cmd    
            for key in eval_metrics[0].keys():
                combined_metrics[key] = torch.stack([metric[key] for metric in eval_metrics])

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
        return self.obs, self.obs_priv, dones, episode_infos, eval_infos, combined_metrics

    def simulate(self,it, is_training=True): # Simulates through one episode
        """
        Simulate through one episode and store the statistics.
        """
        # Simulate through one episode
        next_obs_g, next_priv_obs_g, dones, episode_infos, eval_infos, eval_metrics = self.rollout(it,is_training=is_training)
        # get goal conditioned state
        self.algo.compute_returns(next_priv_obs_g, dones)
        # Show statistics
        stat = self.algo.storage.statistics()

        return stat, episode_infos, eval_infos, eval_metrics

    def agent_train_step(self, it):
        self.algo.actor_critic.train() # Switch to training mode
        # Simulate through one episode
        stat, episode_infos, _, _ = self.simulate(it,is_training=True)
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
                if 'tracking_lin_vel' in key:
                    self._rew_track_lin_vel = episode_infos[key]

    def agent_eval_step(self, it, save_dir, is_training=True): # this function can be called via test or train
        self.algo.actor_critic.eval()
        self.save(os.path.join(save_dir, f'model_{it}.pt'))
        
        stat, episode_infos, _,_ = self.simulate(it,is_training=is_training)
        if stat["avg_reward"] > self.high_score_avg_reward:
            print(f"New high score: {stat['avg_reward']}")
            self.save(os.path.join(save_dir, f'best.pt'))

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

    def create_sinusoidal_command(self, amplitude, frequency, episode_length, sampling_rate):
        """
        Creates a sinusoidal command with the given amplitude, frequency, episode length, and sampling rate.

        Args:
            amplitude: The maximum value of the sinusoid.
            frequency: The frequency of the sinusoid in Hz.
            episode_length: The duration of the episode in seconds.
            sampling_rate: The sampling rate in Hz.

        Returns:
            A JAX array containing the sinusoidal command.
        """

        num_timesteps = int(episode_length * sampling_rate)
        time_values = jp.linspace(0, episode_length, num_timesteps)
        sinusoidal_values = amplitude * jp.sin(2 * jp.pi * frequency * time_values)
        return jp.stack([time_values, sinusoidal_values], axis=1)


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
            print(f"Tracking lin vel: {self._rew_track_lin_vel}")
            
            self.current_learning_iteration += 1
            
            if it % self.eval_interval == 0:
                print(f"Evaluation at epoch: {it}")
                self.agent_eval_step(it, save_dir,is_training=True)
                


            # if (it > 500) and self._rew_track_lin_vel > 1.2:
            #     # Adapting the control range
            #     for key in self.cfg.env.control_range.keys():
            #         cr = self.cfg.env.control_range[key]
            #         cr_new = [num * 1.5 for num in cr]
            #         self.cfg.env.control_range[key] = cr_new

            #     self._rew_track_lin_vel = 0.0
            #     print(f"Config env: {self.cfg.env}")
            #     self.env = self.init_env(self.cfg.scene_xml)
            #     self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)

            # if (it == 6000):
            #     self.env = self.init_env('unitree_go2/terrain_gaussian.xml')
            #     self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)

            if self.curriculum and it % (self.eval_interval+1) ==0:
                # Update level according to paper(Learn to walk in minutes)
                print(f"System level update at epoch: {it}")
                temp = self.cfg.env.manual_control
                self.cfg.env.manual_control = True # walk straight
                self.agent_eval_step(it, save_dir, is_training=True)
                # Move to different terrain if successfull
                self.update_level()
                self.cfg.env.manual_control = temp # TODO: improve layout and yaml

        self.current_learning_iteration = num_total_iteration
        self.save(os.path.join(save_dir, f'last.pt'))


    def test_agent(self, num_iterations, ckpt_path=None):
        """
        Test loop for the agent.
        """
        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        self.algo.actor_critic.eval()
        #eval_results = []
        #single_obs_size = self.env.observation_size // self.cfg.env.num_history_actor
        if self.cfg.env.manual_control.enable and self.cfg.env.manual_control.task == 'experiments':
            self.test_experiments(num_iterations)
        elif self.cfg.env.manual_control and self.cfg.env.manual_control.task == 'track trajectory':
            self.test_tracking_traj(num_iterations)

    def test_tracking_traj(self, num_iterations):
        # eval data stores for every timestep, results are then the used metrics for overall comparison
        trajectory = create_combined_command(0.4, 0.25, 10.0, 50, 0.8, 0.0)[:,1:]
        print(f"Trajectory: {trajectory.shape}")
        print(f"Trajectory: {trajectory}")

        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0., 0., 0.]),traj=trajectory)

        for it in range(num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            self.algo.storage.clear()
        
    def test_experiments(self, num_iterations):
        # eval data stores for every timestep, results are then the used metrics for overall comparison
        eval_data = {'power': [], 'energy': [], 'COT': [], 'trajectory': [], 'local_v': [], 'total_dist':[], 'best_time':100.0, 'successful_envs': None, 'p_gains': []}
        results = {'power': [], 'energy': [], 'COT': [], 'local_v': [], 'top_times': [], 'success_rate':[]}
        time_per_experiment = 0.02*self.cfg.rollouts_per_experiment*self.cfg.timesteps_per_rollout
        print(f"Time per experiment: {time_per_experiment}")
        success_dist = self.cfg.env.manual_control.cmd_x*time_per_experiment*self.cfg.success_threshold
        print(f"Success after distance: {success_dist}")
        for it in range(num_iterations):
            print(f"iteration: {it} ")

            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            
            vel_xy = jp.sqrt(2)/2*self.cfg.env.manual_control.cmd_x
            directions = [jp.array([vel_xy, vel_xy, 0.0]),
                          jp.array([0.0, self.cfg.env.manual_control.cmd_x, 0.0]),
                          jp.array([-vel_xy, vel_xy, 0.0]),
                          jp.array([-self.cfg.env.manual_control.cmd_x, 0.0, 0.0]),
                          jp.array([-vel_xy, -vel_xy, 0.0]),
                          jp.array([0.0, -self.cfg.env.manual_control.cmd_x, 0.0]),
                          jp.array([vel_xy, -vel_xy, 0.0])]

            if it % self.cfg.rollouts_per_experiment == 0 and it >0:
                # Evaluate the results
                results['power'].append(torch.mean(torch.stack(eval_data['power'])))
                results['energy'].append(torch.mean(torch.stack(eval_data['power'])))
                results['COT'].append(torch.mean(torch.stack(eval_data['COT'])))
                results['local_v'].append(torch.mean(torch.stack(eval_data['local_v'])))
                results['top_times'].append(torch.tensor([eval_data['best_time']]))
                if eval_data['successful_envs'] is not None:
                    results['success_rate'].append(torch.tensor([len(eval_data['successful_envs'])/self.cfg.num_envs]))
                else:
                    results['success_rate'].append(torch.tensor([0.0]))

                # Delete the eval_data
                eval_data['successful_envs'] = None
                eval_data['best_time'] = 100.0
                eval_data['power'] = []
                eval_data['energy'] = []
                eval_data['COT'] = []
                eval_data['local_v'] = []
                

                print(f"Success rate: {results['success_rate'][-1]}")
                print(f"Finished experiment {it//self.cfg.rollouts_per_experiment} with mean power: {results['power'][-1]}, mean energy: {results['energy'][-1]}, mean COT: {results['COT'][-1]}, mean local_v: {results['local_v'][-1]}")

                # Change the direction of the manual command
                if it//self.cfg.rollouts_per_experiment < self.cfg.rollouts_per_experiment:
                    self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=directions[(it-1) // self.cfg.rollouts_per_experiment])
            

            COT = torch.mean( torch.abs(eval_metrics['power']/( eval_metrics['m_total']*9.81*torch.linalg.norm(eval_metrics['local_v'], dim=2) ) ))
            
            # Gather the eval_data data with a dictionary
            eval_data['power'].append(torch.mean(eval_metrics['power']))
            eval_data['energy'].append(torch.sum(torch.mean(eval_metrics['power']*0.02, dim=1)))
            eval_data['COT'].append(COT)
            eval_data['local_v'].append(torch.linalg.norm(eval_metrics['local_v'],dim=2))
            eval_data['total_dist'].append(eval_metrics['total_dist'])
            eval_data['p_gains'].append(eval_metrics['p_gains'])
            #print(f"Total distance: {eval_data['total_dist'][-1]}")
            if torch.any(eval_data['total_dist'][-1] > success_dist):

                indices = torch.where(eval_data['total_dist'][-1]>success_dist)
                #print(f"Finished envs: {indices[1]}")
                finished_env = torch.unique(indices[1])
                #print(f"Finished envs: {finished_env}")
                if eval_data['successful_envs'] is None:
                    eval_data['successful_envs'] = finished_env
                else:
                    missing_envs = finished_env[~torch.isin(finished_env, eval_data['successful_envs'])]
                    eval_data['successful_envs'] = torch.cat([eval_data['successful_envs'], missing_envs])
                
                finish_step = torch.min(eval_metrics['total_time'][indices])
                #print(f"Finished step: {finish_step}")
                eval_data['best_time'] = min(eval_data['best_time'], finish_step)

            eval_data['trajectory'].append(eval_metrics['glob_pos'])

            # Plot foot tracking trajectories + command tracking 
            eval_graph([eval_metrics['dof_pos'][:,0,2], eval_metrics['target_dof_pos'][:,0,2]], ['DOF pos', 'Target DOF pos'], f"dof_pos_track{it}", timestep=0.02)
            error = eval_metrics['dof_pos'][:,0,2] - eval_metrics['target_dof_pos'][:,0,2]
            eval_graph([error, eval_metrics['p_gains'][:,0,2]/50.0], ['Error', 'P gains'], f"error_track{it}", timestep=0.02)


            # Recording of foot trajectories
            save_tensors_to_csv([eval_metrics['foot_pos_z'].cpu(), COT.cpu()], [f'foot trajectories {it}', f'Cost of Transport {it}'], f'data_run_{it}.csv')
            eval_graph([eval_metrics['foot_pos_z'][:,0,0], eval_metrics['foot_pos_z'][:,0,1], 
                        eval_metrics['foot_pos_z'][:,0,2], eval_metrics['foot_pos_z'][:,0,3]], 
                        ['FR_foot','FL_foot','RR_foot','RL_foot'], f'Foot z position test run {it}', 0.02)
            self.algo.storage.clear()
        
        # Concatinating all recorded trajectories
        global_trajectory = torch.cat(eval_data['trajectory'], 0)
        print(f"Global trajectory: {global_trajectory.shape}")
        #plot_xy_position(global_trajectory[:,0,:], "Global Position Trajectory")

        COT = torch.mean(torch.stack(eval_data['COT']))
        for key in results.keys():      
            results[key] = torch.stack(results[key]).cpu()
        print(f"||  Results ||: Mean Power[W]: {results['power']}, Energy overall[Ws]: {results['energy']}, COT mean: {results['COT']}")
        save_tensors_to_csv([results['power'], results['energy'], results['local_v'], results['success_rate'], results['COT']], 
                            [f'power', f'energy', 'local_v', 'success_rate', 'COT'], self.result_file_name)
        
        # create_power_energy_bar_chart(title="Power/Energy Consumption", names=["position baseline"], power=power_overall, energy=energy_overall , filename="power_energy_bar_chart", y_lim=y_lim)    


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