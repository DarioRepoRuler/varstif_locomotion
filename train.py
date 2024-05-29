import os
import hydra
import wandb
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper
from tasks.PPOTaskBase import PPOTaskBase




def _create_env(env, num_envs, device, viz=False):
    """
    Create the environment with the specified number of environments and device.
    VmapWrapper->AutoResetWrapper->TorchWrapper(->RenderWrapper)

    Args:
        env (GO2Env): The environment to create.
        num_envs (int): The number of environments to create.
        device (str): The device to use for computation.
        viz (bool): Whether to render the environment.
    """
    env = VmapWrapper(env, batch_size=num_envs)
    env = AutoResetWrapper(env)
    if device == 'cpu':
        env = TorchWrapper(env, device=device, backend='cpu')
    else:
        env = TorchWrapper(env, device=device, backend='gpu')
    if viz:
        env = RenderWrapper(env, render_mode='rgb_array')
        env = RenderWrapper(env, render_mode='rgb_array')
    return env


# Define the configuration according to the schema in config/train.yaml
@hydra.main(config_path='config', config_name='train', version_base="1.2")
def train(cfg: DictConfig):
    """
    Train the model using the provided configuration.

    Args:
        cfg (DictConfig): The configuration for training.

    Returns:
        None
    """
    # Create the environment
    # Print config properties
    print(f"Config: {cfg}")
    
    env = _create_env(GO2Env(cfg.env), num_envs=cfg.num_envs, device=cfg.device, viz=cfg.viz)

    # Set up logging using wandb
    log = cfg.log
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    wandb_logger = None
    if log:
        wandb_logger = wandb.init(project=cfg.project,
                                  group=cfg.group,
                                  name=log_name)
        
    # Create the task and save directory
    task = PPOTaskBase(cfg=cfg, env=env, wandb_logger=wandb_logger)
    save_dir = os.path.join(os.getcwd(), 'outputs', log_name, 'checkpoints')
    #save_dir = os.path.join(Path.home(), cfg.log_dir, log_name, 'checkpoints')

    # Train the model
    # if cfg.ckpt_path is not None:
    #     task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
    #                 save_dir=save_dir, ckpt_path=cfg.ckpt_path)
    # else:
    #     task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
    #                 save_dir=save_dir)
    task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
                    save_dir=save_dir, ckpt_path=cfg.ckpt_path)
    
    # Finish logging
    if log:
        wandb.finish()


if __name__ == '__main__':
    train()
