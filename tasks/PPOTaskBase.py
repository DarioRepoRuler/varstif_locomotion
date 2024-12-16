import os
import torch
from torch import nn
from rl_algo.ppo import PPO
from envs.common.mjx_env import MjxEnv
import mujoco

import jax.numpy as jp
import jax
from envs.robots.go2_env import GO2Env
from envs.common.wrapper import _create_env
import matplotlib.pyplot as plt

from utils.helper_traj import create_combined_command

class PPOTaskBase(nn.Module):
    def __init__(self,
                 cfg,
                 eval_interval=50,
                 save_interval=50,
                 test_interval=100,
                 wandb_logger=None):
        super().__init__()
        self.cfg = cfg
        self.control_mode = cfg.env.control_mode
        self.device = cfg.device
        self.eval_interval = eval_interval
        self.save_interval = save_interval # not used
        self.test_interval = test_interval # not used
        
        self.render_mode = "human"
        
        self.initial_xy = jp.array([0, 0])
        self.manual_cmd = jp.array([cfg.env.manual_control.cmd_x, cfg.env.manual_control.cmd_y, cfg.env.manual_control.cmd_ang])
        self.high_score_avg_reward =0.0
        self.wandb_logger = wandb_logger
        self.curriculum = cfg.curriculum
        
        # Initialize control paradigm
        self.view_env_id = 0
        if self.control_mode == 'P' or self.control_mode == 'T':
            num_actions = 12
            offset =0
        elif self.control_mode == 'VIC_1':
            num_actions= 15
            offset =  3
        elif self.control_mode == 'VIC_2':
            num_actions= 16
            offset = 4
        elif self.control_mode == 'VIC_3':
            num_actions= 24
            offset = 12
        elif self.control_mode == 'VIC_4':
            num_actions= 12+7
            offset = 7
        elif self.control_mode == 'VIC_5':
            num_actions= 12+3
            offset = 3
        
        self.cfg.env.single_obs_size = self.cfg.env.single_obs_size + offset
        self.cfg.env.single_obs_size_priv = self.cfg.env.single_obs_size_priv + offset

        # Initialize the algorithm
        self.algo = PPO(cfg=self.cfg.policy,
                        num_envs=self.cfg.num_envs,
                        num_actions=num_actions,
                        episode_length=self.cfg.timesteps_per_rollout,
                        num_single_obs = self.cfg.env.single_obs_size,
                        num_env_obs=self.cfg.env.single_obs_size*self.cfg.env.num_history_actor,
                        num_priv_obs=self.cfg.env.single_obs_size_priv*self.cfg.env.num_history_critic,
                        control_mode=self.control_mode,
                        device=self.device)

        self.current_learning_iteration = 0
        self.level = 0

        # # Start the keyboard listener thread
        if self.cfg.viz:
            import threading
            from pynput import keyboard as pynput_keyboard
            self.pynput_keyboard = pynput_keyboard
            self.threading = threading
            self.keyboard_listener_thread = self.threading.Thread(target=self.keyboard_listener)
            self.keyboard_listener_thread.daemon = True
            self.keyboard_listener_thread.start()

    def init_env(self, scene_xml=None, render_mode = "human"):
        env = _create_env(GO2Env(self.cfg.env, scene_xml=scene_xml), num_envs=self.cfg.num_envs, device=self.cfg.device, viz=self.cfg.viz, domain_cfg=self.cfg.env.domain_rand, render_mode=self.render_mode)
        return env

    def on_press(self, key):
        '''
        Function to handle the key press events -> Changing the viewed environment
        '''
        try:
            if key == self.pynput_keyboard.Key.up:
                print(f"Up arrow pressed Viewed env: {self.view_env_id}")
                self.view_env_id = min(self.cfg.num_envs, self.view_env_id+1)
            elif key == self.pynput_keyboard.Key.down:
                self.view_env_id = max(0, self.view_env_id-1)
                print(f"Down arrow pressed,  Viewed env: {self.view_env_id}")
        except AttributeError:
            pass
    
    def keyboard_listener(self):
        '''
        Function to start the keyboard listener -> switching between the viewed environments
        '''
        with self.pynput_keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def step(self, obs_g, privileged_obs_g, is_training=True):
        """
        Performs action in the environment and returns the next observation. Everything outside this function will not directly
        affect the environment or the learning process.
        """        
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

    def rollout(self, is_training=True):
        """
        Steps through the environment for one episode and returns the final observation and statistics.
        Episode infos represent the average reward
        """
        episode_infos = {}
        # Gather evaluation information only in test mode
        eval_infos = {}
        cmd = []
        eval_metrics = []
        kick_metrics = {'kick_theta':[], 'kick_force_magnitude':[]}

        with torch.inference_mode():
            pos_x = torch.zeros(self.cfg.timesteps_per_rollout, device=self.device, dtype=torch.float32)
            time_out = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.bool)
            steps=torch.zeros(self.cfg.timesteps_per_rollout, self.cfg.num_envs, device=self.device, dtype=torch.int32)


            for i in range(self.cfg.timesteps_per_rollout):
                next_obs_g, next_priv_obs_g, dones, info, metrics = self.step(self.obs, self.obs_priv, is_training)
                rew_info= info['rewards']
                pos_x[i] = info['last_qpos'][0,0]
                time_out |= info['time_out'] > 0
                for key in rew_info.keys():
                    if key not in episode_infos.keys():
                        episode_infos[key] = torch.mean(rew_info[key]) #mean over all envs
                    else:
                        episode_infos[key] += torch.mean(rew_info[key])
                steps[i,:] = info['step']
                
                if not is_training:
                    cmd.append(info['command'])
                    eval_metrics.append(metrics)
                    kick_metrics['kick_theta'].append(info['kick_theta'])
                    kick_metrics['kick_force_magnitude'].append(info['kick_force_magnitude'])
                
                # update observation
                self.obs = next_obs_g
                self.priv_obs = next_priv_obs_g

        combined_metrics = {}
        if not is_training: 
            kick_metrics['kick_theta'] = torch.stack(kick_metrics['kick_theta'])
            kick_metrics['kick_force_magnitude'] = torch.stack(kick_metrics['kick_force_magnitude'])
            cmd = torch.stack(cmd)
            eval_infos['kick_theta'] = kick_metrics['kick_theta']
            eval_infos['kick_force_magnitude'] = kick_metrics['kick_force_magnitude']
            eval_infos['cmd'] = cmd 
            eval_infos['steps'] = steps 
            for key in eval_metrics[0].keys():
                combined_metrics[key] = torch.stack([metric[key] for metric in eval_metrics])

        for key in episode_infos.keys():
            episode_infos[key] = episode_infos[key] / self.cfg.timesteps_per_rollout

        if is_training:
            print(f"Rewards infos: {episode_infos}")
        if episode_infos['termination']>0.0:
            print(f"Episode infos: {episode_infos}")
            print(f"Termination reward: {episode_infos['termination']}")
        
        return self.obs, self.obs_priv, dones, episode_infos, eval_infos, combined_metrics

    def simulate(self,it, is_training=True): # Simulates through one episode
        """
        Simulate through one episode and store the statistics.
        """
        # Simulate through one episode
        next_obs_g, next_priv_obs_g, dones, episode_infos, eval_infos, eval_metrics = self.rollout(is_training=is_training)
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
                    self.wandb_logger.log({f'rewards_train/{key}': episode_infos[key]}, step=it)
                if 'tracking_lin_vel' in key:
                    self._rew_track_lin_vel = episode_infos[key]

    def agent_eval_step(self, it, save_dir, is_training=False): # this function can be called via test or train

        self.algo.actor_critic.eval()
        stat, episode_infos, _,_ = self.simulate(it,is_training=is_training)
        if stat["avg_reward"] > self.high_score_avg_reward:
            self.save(os.path.join(save_dir, f'best.pt'))

        if self.wandb_logger:
            self.wandb_logger.log({'val/avg_traverse': stat["avg_traverse"],
                                   'val/avg_reward': stat["avg_reward"],
                                   'val/dones': stat["dones"],
                                   'val/done_rate': stat["done_rate"],
                                   }, step=it)
            # ---------------- logging reward ------------------------------#
            for key in episode_infos.keys():
                self.wandb_logger.log({f'rewards_val/{key}': episode_infos[key]}, step=it)
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0., 0., 0.]))
        self.algo.storage.clear()

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

        self.env = self.init_env(self.cfg.scene_xml)
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)
        if self.cfg.curriculum:
            self.initial_xy = jp.array([-2., -2.5])

        num_total_iteration = num_learning_iterations + self.current_learning_iteration
        for it in range(self.current_learning_iteration, num_total_iteration):
            print(f"Epoch: {it}")
            self.agent_train_step(it)
            
            self.current_learning_iteration += 1
            
            if (it % self.eval_interval == 0) and it > 0:
                self.save(os.path.join(save_dir, f'model_{it}.pt'))
                print(f"Evaluation at epoch: {it}")
                self.agent_eval_step(it, save_dir,is_training=False)

            # Change terrain
            # if (it == 6000):
            #     self.env = self.init_env('unitree_go2/terrain_gaussian.xml')
            #     self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)

        self.current_learning_iteration = num_total_iteration
        self.save(os.path.join(save_dir, f'last.pt'))

    def test_agent(self, num_iterations, ckpt_path=None):
        """
        Test loop for the agent.
        """

        if ckpt_path:
            self.load(ckpt_path, load_optimizer=False)
        if self.cfg.viz and self.cfg.record_video:
            self.render_mode = "rgb_array"
        self.algo.actor_critic.eval()
        if self.cfg.env.manual_control.enable and self.cfg.env.manual_control.task == 'heading_directions':
            self.test_heading_directions(num_iterations)
        elif self.cfg.env.manual_control and self.cfg.env.manual_control.task == 'xy_random':
            self.test_xy_random(num_iterations)
        elif self.cfg.env.manual_control and self.cfg.env.manual_control.task == 'track_trajectory':
            self.test_tracking_traj(num_iterations)
        elif self.cfg.env.manual_control and self.cfg.env.manual_control.task == 'force_push':
            self.test_force_push_random_1(num_iterations)
        elif self.cfg.env.manual_control and self.cfg.env.manual_control.task == 'escape_pyramids':
            self.test_escape_pyramid(num_iterations)
        elif self.cfg.env.manual_control.task == 'auto':
            self.test_auto(num_iterations)
        elif self.cfg.env.manual_control.task == 'stiffness':
            self.test_stiffness(num_iterations)
        elif self.cfg.env.manual_control.task == 'default':
            self.test_default(num_iterations)
            
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
    
    def get_model_name(self):
        if self.cfg.ckpt_path is None:
            return None
        if 'checkpoints' in self.cfg.ckpt_path:
            parent_dir = os.path.dirname(os.path.dirname(self.cfg.ckpt_path))
            parent_dir_name = os.path.basename(parent_dir)
            grandparent_dir_name = os.path.basename(os.path.dirname(parent_dir))
            name = f'{grandparent_dir_name}_{parent_dir_name}'
        else:
            name = self.cfg.ckpt_path.split('/')[-1].split('.')[0]
        return name 
    
    # --------------------- Evaluation functions --------------------- #

    def test_auto(self, num_iterations):
        print(f"Starting auto evaluation")
        print(f"Starting heading directions evaluation:")
        self.test_heading_directions(num_iterations)
        print(f"Finihsed heading directions evaluation")
        print(f"Starting force push evaluation")
        # Change settings to force push 
        self.cfg.env.domain_rand.randomisation = False
        self.cfg.env.manual_control.enable = True
        self.cfg.env.manual_control.cmd_x = 0.3
        self.cfg.timesteps_per_rollout = 50
        self.cfg.env.enable_force_kick = True
        self.cfg.env.force_kick_interval = 150
        self.cfg.env.impulse_force_kick= False
        self.cfg.env.kick_vel = 0.0
        self.cfg.rollouts_per_experiment = 5
        self.cfg.num_iterations = 21
        self.cfg.env.force_kick_duration = 0.2
        self.cfg.env.kick_force = [50.0, 300.0]
        self.cfg.env.sample_command_interval = 301
        self.test_force_push_random(self.cfg.num_iterations)
        print(f"Finished force push evaluation")
        # Change settings to random xy
        self.cfg.env.manual_control.enable = False
        self.cfg.env.control_range['cmd_x'] = [-1.5, 1.5]
        self.cfg.env.control_range['cmd_y'] = [-1.5, 1.5]
        self.cfg.env.control_range['cmd_ang'] = [0.0, 0.0]
        self.cfg.env.enable_force_kick = False
        print(f"Starting random xy evaluation")
        self.test_xy_random(self.cfg.num_iterations)

    def test_default(self, num_iterations):
        #from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position
        self.env = self.init_env(self.cfg.scene_xml, render_mode="human")
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)
        for it in range(num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            
            print(f"Episode info: {episode_info}")
            
            commands = eval_infos['cmd'][-1,:,:]
            cmd_norm = torch.linalg.norm(commands, dim=1).unsqueeze(1).cpu()
            theta = torch.atan2(commands[:,1],commands[:,0]).cpu() # in rad
            self.algo.storage.clear()
    
    def test_stiffness(self, num_iterations):
        self.env = self.init_env(self.cfg.scene_xml, render_mode="human")
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0.5, 0., 0.]))
        
        import numpy as np
        from utils.graphs_gen import  save_tensors_to_csv
        
        # Control over evaluation of stiffness test
        show_power = True
        
        # Enable interactive mode for live plotting
        plt.ion()
        
        # Set up the interactive plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.set_title("P Gains over Time")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("P Gains")
        if not show_power:
            ax2.set_title("Position Errors over Time")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Position Error")
        if show_power:
            ax2.set_title("Power over Time")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Power")
            
        # Lists to accumulate values for plotting each of the 3 p_gains and position errors
        results = {'p_gains_values':[], 'position_errors':[], 'power':[]}
        if self.cfg.viz and self.cfg.record_video:
            path = os.path.join(os.getcwd(), 'outputs', 'videos', f'stiffness_{self.cfg.result_tag}_{self.get_model_name()}.mp4')
            self.env.start_video_recording(path, fps=50)
        
        for it in range(num_iterations*self.cfg.timesteps_per_rollout):
            print(f"Iteration: {it}")
            
            # Run the simulation step and gather data
            
            next_obs_g, next_priv_obs_g, dones, info, metrics = self.step(self.obs, self.obs_priv, is_training=False)
            # Assuming `eval_metrics` contains `p_gains` and `dof_pos` for position errors
            if 'p_gains' in metrics and 'dof_pos' in metrics and 'target_dof_pos' in metrics:                
                # Extract the p_gain and error values for the selected indices
                current_p_gain = metrics['p_gains'][0, :] #.cpu().numpy())
                current_error = (metrics['dof_pos'][0, :] - metrics['target_dof_pos'][0, :]) #.cpu().numpy()

                # Append values for each of the p_gain components and errors
                results['p_gains_values'].append(current_p_gain)
                results['position_errors'].append(current_error)
                results['power'].append(metrics['power'])
                
                # Convert lists to numpy arrays for easier slicing and plotting               
                p_gains_array = torch.stack(results['p_gains_values']).cpu().numpy()             # Shape (num_iterations, batch_size, 3)
                position_errors_array = torch.stack(results['position_errors']).cpu().numpy()  # Shape (num_iterations, batch_size, 3)
                power_array = torch.stack(results['power']).cpu().numpy()  # Shape (num_iterations, batch_size, 3)
                
                # Clear and update the plots
                ax1.clear()
                ax1.plot(p_gains_array[-50:, 0].flatten(), label="P Gain FR", color="blue")
                ax1.plot(p_gains_array[-50:, 3].flatten(), label="P Gain FL", color="green")
                ax1.plot(p_gains_array[-50:, 6].flatten(), label="P Gain RR", color="red")
                ax1.plot(p_gains_array[-50:, 9].flatten(), label="P Gain RL", color="purple")
                ax1.legend(loc="upper right")
                
                if show_power:
                    ax2.clear()
                    ax2.plot(np.array(power_array).flatten(), label="Power", color="blue")
                    ax2.legend(loc="upper right")
                if not show_power:
                    ax2.clear()
                    ax2.plot(np.abs(position_errors_array[-50:,0:3]).mean(axis=1).flatten(), label="Position Error FR", color="blue")
                    ax2.plot(np.abs(position_errors_array[-50:,3:6]).mean(axis=1).flatten(), label="Position Error FL", color="green")
                    ax2.plot(np.abs(position_errors_array[-50:,6:9]).mean(axis=1).flatten(), label="Position Error RR", color="red")
                    ax2.plot(np.abs(position_errors_array[-50:,9:12]).mean(axis=1).flatten(), label="Position Error RL", color="purple")
                    ax2.legend(loc="upper right")
                
                # Add a small pause to update the plots
                plt.pause(0.01)
                # update observation
                self.obs = next_obs_g
                self.priv_obs = next_priv_obs_g
                self.algo.storage.clear() # we don't need to store the data for training
                    
        # Turn off interactive mode and show the final plot
        plt.ioff()
        if self.cfg.viz and self.cfg.record_video:
            self.env.stop_video_recording()
            
        print(f"Pgains shape in results: {torch.stack(results['p_gains_values']).shape}")
        print(f"Position errors shape in results: {torch.stack(results['position_errors']).shape}")
        print(f"Power shape in results: {torch.stack(results['power']).shape}")
        
        name = self.get_model_name()
            
        for key in results.keys():
            if len(results[key]) != 0:      
                results[key] = torch.stack(results[key], dim=0).cpu()
            else:
                results[key] = torch.tensor([0.0])
        save_tensors_to_csv([results['p_gains_values'], results['position_errors'], results['power']],['p_gains_values', 'position_errors', 'power'], \
                             f'stiffness_{self.cfg.result_tag}_{name}.csv')        
        
    def test_force_push_random(self, num_iterations):
        name = self.get_model_name()
        self.env = self.init_env(self.cfg.scene_xml, render_mode="human")
        if self.cfg.viz and self.cfg.record_video:
            path = os.path.join(os.getcwd(), 'outputs', 'videos', f'force_push_{self.cfg.result_tag}_{self.get_model_name()}.mp4')
            self.env.start_video_recording(path, fps=50)
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0.5, 0., 0.]))
        from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position
        # For using this script please make sure the push force interval is set to 150 and the timesteps per rollout to 50 and the 
        if (self.cfg.env.force_kick_interval != 150) or (self.cfg.timesteps_per_rollout != 50) or (self.cfg.rollouts_per_experiment != 5) or (self.cfg.env.enable_force_kick != True):
            print("Please set the force kick interval to 150, timesteps per rollout to 50 and rollouts per experiment to 5 and enable force kick to true")
            return

        # This way the agent is pushes 
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([self.cfg.env.manual_control.cmd_x, 0., 0.]))
        results = {'success': [], 'kick_theta':[], 'kick_force_magnitude':[], 'recovery_time':[]}
        recovery_time = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.float32)
        threshold = 0.2
        start_temp, end_temp =None, torch.ones_like(recovery_time)*self.cfg.rollouts_per_experiment*self.cfg.timesteps_per_rollout
        for it in range(1, num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)

            tracking_error = torch.linalg.norm(eval_metrics['local_v'][:,:,:2] - eval_infos['cmd'][:,:,:2], dim=2)
            tracking_error_normalised = tracking_error/(torch.linalg.norm(eval_infos['cmd'][:,:,:2], dim=2)+1e-6)
            recovered = torch.logical_and(tracking_error_normalised < threshold , eval_infos['steps'] > self.cfg.env.force_kick_interval+4)
            # Find indices of non-zero elements, along axis=0 and extract the values
            non_zero_indices = eval_infos['kick_force_magnitude'].nonzero(as_tuple=True)
            non_zero_kick_force_magnitude = eval_infos['kick_force_magnitude'][non_zero_indices[0]]
            #start_recovery = eval_infos['steps'][non_zero_indices]
            #recovered_indices = recovered.nonzero(as_tuple=True)
            # masked_time = torch.where(recovered, eval_infos['steps'], torch.full_like(eval_infos['steps'], 500))
            # end_recovery = torch.min(masked_time, dim=0).values
            # if len(start_recovery) !=0 and start_temp == None:
            #     start_temp = start_recovery
            # if torch.any(end_recovery !=500) and torch.all(end_temp == 500):
            #     end_temp = end_recovery
            #print(f"Step of pushed: {eval_infos['steps'][non_zero_indices]}, {eval_infos['steps'][non_zero_indices].shape}")
            
            non_zero_kick_theta = eval_infos['kick_theta'][non_zero_indices[0]]
            magnitudes = torch.unique(non_zero_kick_force_magnitude, dim=0)
            thetas = torch.unique(non_zero_kick_theta, dim=0)
            print(f"Magnitudes: {magnitudes.numel()}")
            if (magnitudes.numel() != 0): # and ((it+1) % 5 == 0):
                magnitudes = torch.where(magnitudes.sum(dim=0)!=0, magnitudes.sum(dim=0), torch.tensor(0.))
                thetas = torch.where(thetas.sum(dim=0)!=0, thetas.sum(dim=0), torch.tensor(0.))
                results['kick_theta'].append(thetas.unsqueeze(0))
                results['kick_force_magnitude'].append(magnitudes.unsqueeze(0))
                # results['recovery_time'].append(end_temp-start_temp)
                print(f"Thetas: {thetas.shape}")
                print(f"Magnitudes: {magnitudes.shape}")
                #end_time_temp = None

            if it % 5 == 0 and it>0:
                print(f"Finished experiment {it//5} in total: {it//5*self.cfg.num_envs}")
                success = eval_infos['steps'][-1,:] >= 5*self.cfg.timesteps_per_rollout
                print(f"Success rate: {success.shape}")
                results['success'].append(success)
                self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([self.cfg.env.manual_control.cmd_x, 0., 0.]))
                start_temp, end_temp = None, torch.ones_like(recovery_time)*500
            self.algo.storage.clear()
        
        for key in results.keys():
            if len(results[key]) != 0:      
                results[key] = torch.stack(results[key], dim=0).cpu()
            else:
                results[key] = torch.tensor([0.0])

        if self.cfg.viz and self.cfg.record_video:
            self.env.stop_video_recording()
        print(f"Storing results in: force_push_results_{name}.csv")
        print(f"Average recovery {torch.mean(results['recovery_time'][torch.where(results['recovery_time']<349)])*0.02}")
        save_tensors_to_csv([results['success'], results['kick_force_magnitude'], results['kick_theta'], results['recovery_time']],['success_rate', 'kick_force_magnitude', 'kick_theta', 'recovery_time'], \
                             f'force_push_{self.cfg.result_tag}_{name}.csv')
        
    def test_force_push_random_1(self, num_iterations):
        name = self.get_model_name()
        self.env = self.init_env(self.cfg.scene_xml, render_mode="human")
        if self.cfg.viz and self.cfg.record_video:
            path = os.path.join(os.getcwd(), 'outputs', 'videos', f'force_push_{self.cfg.result_tag}_{self.get_model_name()}.mp4')
            self.env.start_video_recording(path, fps=50)
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0.5, 0., 0.]))
        from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position
        # For using this script please make sure the push force interval is set to 150 and the timesteps per rollout to 50 and the 
        if (self.cfg.timesteps_per_rollout != 50) or (self.cfg.rollouts_per_experiment != 5) or (self.cfg.env.enable_force_kick != True):
            print("Please set the force kick interval to 175, timesteps per rollout to 50 and rollouts per experiment to 5 and enable force kick to true")
            return
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([self.cfg.env.manual_control.cmd_x, 0., 0.]))
        
        results = {'success': torch.zeros(self.cfg.num_envs*4, dtype=torch.bool), 'kick_theta':torch.zeros(self.cfg.num_envs*4), 'kick_force_magnitude':torch.zeros(self.cfg.num_envs*4), 'recovery_time':torch.zeros(self.cfg.num_envs*4)}
        experiment = 0
        for it in range(1, num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            pushed_envs = eval_infos['kick_force_magnitude'].nonzero(as_tuple = True)[1]            
            results['kick_force_magnitude'][experiment*self.cfg.num_envs + pushed_envs.cpu()] = eval_infos['kick_force_magnitude'][eval_infos['kick_force_magnitude']!=0].cpu()
            results['kick_theta'][experiment*self.cfg.num_envs + pushed_envs.cpu()] = eval_infos['kick_theta'][eval_infos['kick_force_magnitude']!=0].cpu()
            if it % 5 == 0 and it>0:
                print(f"Finished experiment {experiment+1} in total: {(experiment+1)*self.cfg.num_envs}")
                pushed_envs = results['kick_force_magnitude'][experiment*self.cfg.num_envs:].nonzero()[:,0]
                print(f"pushed envs: {len(pushed_envs.cpu())}")
                results['success'][experiment*self.cfg.num_envs + pushed_envs.cpu()] = eval_infos['steps'][-1,:].cpu() >= 5*self.cfg.timesteps_per_rollout
                self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([self.cfg.env.manual_control.cmd_x, 0., 0.]))
                experiment += 1
            self.algo.storage.clear()
        
        print(f"Storing results in: force_push_{self.cfg.result_tag}_{name}.csv")

        save_tensors_to_csv([results['success'], results['kick_force_magnitude'], results['kick_theta'], results['recovery_time']],['success_rate', 'kick_force_magnitude', 'kick_theta', 'recovery_time'], \
                             f'force_push_{self.cfg.result_tag}_{name}.csv')
         
    def test_escape_pyramid(self, num_iterations):
        self.env = self.init_env(f'unitree_go2/terrain_pyramid_l0.xml')
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0.5, 0., 0.]))
        from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position

        results = {'success_rate': []}
        eval_data = {'total_dist':[], 'successful_envs': None}
        success_dist = 4.5
        level = 1
        for it in range(num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            # Calculate successrate: 
            print(f" Dones: {stat['dones']}")
            eval_data['total_dist'].append(eval_metrics['total_dist'])
            

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

            if it % self.cfg.rollouts_per_experiment == 0 and it >0 :
                if eval_data['successful_envs'] is not None:
                    results['success_rate'].append(torch.tensor([len(eval_data['successful_envs'])/self.cfg.num_envs]))
                else:
                    results['success_rate'].append(torch.tensor([0.0]))
                print(f"Finished experiment {it//self.cfg.rollouts_per_experiment} with success rate: {results['success_rate'][-1]}")
                eval_data['successful_envs'] = None
                if level < 4:
                    level += 1
                    self.env = self.init_env(f'unitree_go2/terrain_pyramid_l{level}.xml')
                    self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0.5, 0., 0.]))

            self.algo.storage.clear()

        name = self.cfg.ckpt_path.split('/')[-1].split('.')[0]
        for key in results.keys():      
            results[key] = torch.stack(results[key]).cpu()
        save_tensors_to_csv([results['success_rate']], 
                        [f'success_rate', ], f'pyramid_{self.cfg.result_tag}_{name}.csv')

    def test_tracking_traj(self, num_iterations):
        # eval data stores for every timestep, results are then the used metrics for overall comparison
        trajectory = create_combined_command(0.4, 0.25, 10.0, 50, 0.8, 0.0)[:,1:]
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0., 0., 0.]),traj=trajectory)

        for it in range(num_iterations):
            print(f"iteration: {it} ")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            self.algo.storage.clear()
        
    def test_heading_directions(self, num_iterations):
        # Get checkpoint name
        name = self.get_model_name()
        
        self.env = self.init_env(self.cfg.scene_xml)
        # Start recording if wanted
        if self.cfg.viz and self.cfg.record_video:
            path = os.path.join(os.getcwd(), 'outputs', 'videos', f'heading_{self.cfg.result_tag}_{self.get_model_name()}.mp4')
            self.env.start_video_recording(path, fps=50)
        
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)
        from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position
        if (self.cfg.env.manual_control.enable!=True) or (self.cfg.rollouts_per_experiment != 8) \
            or (self.cfg.timesteps_per_rollout != 50) or (self.cfg.num_iterations!=65) or (self.cfg.env.enable_force_kick ==True) \
                or (self.cfg.env.kick_vel !=0.0):
            print("Please set the manual control to true, rollouts per experiment to 8, timesteps per rollout to 50 and num iterations to 65 and disable force/vel. kick")
            return
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
            

            COT = torch.mean( torch.abs(eval_metrics['power']/( eval_metrics['m_total']*9.81*torch.linalg.norm(eval_metrics['local_v'], dim=2) ) ))
            
            # Gather the eval_data data with a dictionary
            eval_data['power'].append(torch.mean(eval_metrics['power']))
            eval_data['energy'].append(torch.sum(torch.mean(eval_metrics['power']*0.02, dim=1)))
            eval_data['COT'].append(COT)
            eval_data['local_v'].append(torch.linalg.norm(eval_metrics['local_v'],dim=2))
            eval_data['total_dist'].append(eval_metrics['total_dist']) 
            eval_data['p_gains'].append(eval_metrics['p_gains'])

            #print(f"Eval metric tot distance: {eval_metrics['total_dist']}")
            #print(f"Total distance: {eval_metrics['total_dist'][-1,:]}")

            if torch.any(eval_data['total_dist'][-1] > success_dist):
                indices = torch.where(eval_data['total_dist'][-1]>success_dist)
                #print(f"Finished envs: {indices[1]}")
                finished_env = torch.unique(indices[1])
                #print(f"Finished envs: {finished_env}")
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

            if (self.cfg.plot_details==True):
                # Plot foot tracking trajectories + command tracking 
                time_graph([eval_metrics['dof_pos'][:,0,2], eval_metrics['target_dof_pos'][:,0,2]], ['DOF pos', 'Target DOF pos'], f"details/dof_pos_track{it}", timestep=0.02)
                error = eval_metrics['dof_pos'][:,0,2] - eval_metrics['target_dof_pos'][:,0,2]
                time_graph([error, eval_metrics['p_gains'][:,0,2]], ['Error', 'P gains'], f"details/error_track{it}", timestep=0.02)
                # Recording foot trajectories 
                save_tensors_to_csv([eval_metrics['foot_pos_z'].cpu(), COT.cpu()], [f'foot trajectories {it}', f'Cost of Transport {it}'], f'details/data_run_{it}.csv')
                time_graph([eval_metrics['foot_pos_z'][:,0,0], eval_metrics['foot_pos_z'][:,0,1], 
                            eval_metrics['foot_pos_z'][:,0,2], eval_metrics['foot_pos_z'][:,0,3]], 
                            ['FR_foot','FL_foot','RR_foot','RL_foot'], f'details/Foot z position test run {it}', 0.02)
            
            self.algo.storage.clear()
        if self.cfg.viz and self.cfg.record_video:
            self.env.stop_video_recording()
        # Concatinating all recorded trajectories
        global_trajectory = torch.cat(eval_data['trajectory'], 0)
        print(f"Global trajectory: {global_trajectory.shape}")
        #plot_xy_position(global_trajectory[:,0,:], "Global Position Trajectory")

        COT = torch.mean(torch.stack(results['COT']))
        for key in results.keys():
            if len(results[key]) != 0:      
                results[key] = torch.stack(results[key], dim=0).cpu()
            else:
                results[key] = torch.tensor([0.0])
        print(f"||  Results ||: Mean Power[W]: {results['power']}, Energy overall[Ws]: {results['energy']}, COT mean: {results['COT']}")
        
        
        
        save_tensors_to_csv([results['power'], results['energy'], results['local_v'], results['success_rate'], results['COT']], 
                            [f'power', f'energy', 'local_v', 'success_rate', 'COT'], f'heading_directions_{self.cfg.result_tag}_{name}.csv')
        
    def test_xy_random(self, num_iterations):
        name = self.get_model_name()
        
        self.env = self.init_env(self.cfg.scene_xml)
        if self.cfg.viz and self.cfg.record_video:
            path = os.path.join(os.getcwd(), 'outputs', 'videos', f'cmd_rando_{self.cfg.result_tag}_{self.get_model_name()}.mp4')
            self.env.start_video_recording(path, fps=50)
        self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=self.manual_cmd)
        
        from utils.graphs_gen import time_graph, create_multiple_box_plots, create_power_energy_bar_chart, save_tensors_to_csv, load_tensor_from_csv, plot_xy_position
        if (self.cfg.env.manual_control.enable==True) or (self.cfg.env.control_range['cmd_ang'] !=[0.,0.]) \
            or (self.cfg.timesteps_per_rollout != 50) or (self.cfg.env.enable_force_kick ==True) or \
                (self.cfg.env.sample_command_interval<self.cfg.timesteps_per_rollout*self.cfg.rollouts_per_experiment) \
                or (self.cfg.env.kick_vel !=0.0):
            print("Please set to manual control to false, ang. control range to 0, timesteps per rollout to 50 and disable force/vel. kick")
            return

        results ={ 'success':[], 'cmd_theta':[], 'cmd_norm':[], 'cmd': [], 'local_v':[] }
        mean_error = []
        
        for it in range(1,num_iterations):
            print(f"iteration: {it}")
            stat, episode_info, eval_infos, eval_metrics = self.simulate(it,is_training=False)
            
            commands = eval_infos['cmd'][1,:,:2] # num_timesteps, num_envs, num_cmds
            cmd_norm = torch.linalg.norm(commands, dim=1).unsqueeze(1)
            
            # Do only use the first two indices of the command and count the error according to the command
            track_error = torch.mean(torch.linalg.norm(eval_metrics['local_v'][:,:,:2] - eval_infos['cmd'][:,:,:2], dim=2), dim=0)
            results['cmd'].append(eval_infos['cmd'][:,:,:2])
            results['local_v'].append(eval_metrics['local_v'][:,:,:2])
            mean_error.append(track_error)
            theta = torch.atan2(commands[:,1],commands[:,0]) # in rad
            target_x, target_y = (torch.cos(theta)*cmd_norm.T*self.cfg.rollouts_per_experiment, torch.sin(theta)*cmd_norm.T*self.cfg.rollouts_per_experiment)
            distance = torch.sqrt((eval_metrics['glob_pos'][-1,:,0]-target_x)**2 + (eval_metrics['glob_pos'][-1,:,1]-target_y)**2)

            if it % self.cfg.rollouts_per_experiment == 0 and it >0:
                mean_error = torch.mean(torch.stack(mean_error[-4:], dim=0),dim=0)
                success = (mean_error /(cmd_norm[:,0]+1.e-8) < 0.2).cpu().unsqueeze(0)
                results['success'].append(success)
                results['cmd_theta'].append(theta)
                results['cmd_norm'].append(cmd_norm)# *self.cfg.rollouts_per_experiment)
                print(f"Finished xy experiment {it//self.cfg.rollouts_per_experiment} with success rate: {torch.sum(success)/self.cfg.num_envs}")
                self.obs, self.obs_priv = self.env.reset(initial_xy=self.initial_xy, manual_cmd=jp.array([0., 0., 0.]))
                mean_error = []
            self.algo.storage.clear()
        
        for key in results.keys():
            if len(results[key]) != 0:      
                results[key] = torch.stack(results[key], dim=0).cpu()
            else:
                results[key] = torch.tensor([0.0])

        if self.cfg.viz and self.cfg.record_video:
            self.env.stop_video_recording()
        save_tensors_to_csv([results['success'], results['cmd_norm'], results['cmd_theta'], results['cmd'], results['local_v']],['success', 'cmd_norm', 'cmd_theta', 'cmd', 'local_v'], \
                             f'cmd_rando_xy_{self.cfg.result_tag}_{name}.csv')  


