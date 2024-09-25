import os
import hydra
import wandb
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import wrap, domain_randomize
from tasks.PPOTaskBase import PPOTaskBase

from omegaconf import OmegaConf



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

    # Set up logging using wandb
    log = cfg.log
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(f"Configuration {type(cfg)}")
    wandb_logger = None
    if log:
        wandb_logger = wandb.init(
                                  config=cfg_dict, 
                                  project=cfg.project,
                                  group=cfg.group,
                                  name=log_name,
                                  settings=wandb.Settings(code_dir=".")
                                  )
        
    # Create the task and save directory
    task = PPOTaskBase(cfg=cfg,  wandb_logger=wandb_logger)
    save_dir = os.path.join(os.getcwd(), 'outputs', log_name, 'checkpoints')

    # Train the model
    if cfg.ckpt_path is not None:
        # Interpreting as relative path
        ckpt_path = os.path.join(os.getcwd(),cfg.ckpt_path)
        task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
                    save_dir=save_dir, ckpt_path=ckpt_path)
    else:
        task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
                    save_dir=save_dir)

    
    # Finish logging
    if log:
        wandb.finish()


if __name__ == '__main__':
    train()
