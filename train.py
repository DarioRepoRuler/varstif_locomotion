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
    env = VmapWrapper(env, batch_size=num_envs)
    env = AutoResetWrapper(env)
    if device == 'cpu':
        env = TorchWrapper(env, device=device, backend='cpu')
    else:
        env = TorchWrapper(env, device=device, backend='gpu')
    if viz:
        env = RenderWrapper(env, render_mode='human')

    return env


@hydra.main(config_path='config', config_name='train', version_base="1.2")


def train(cfg: DictConfig):
    
    env = _create_env(GO2Env(cfg.env), num_envs=cfg.num_envs, device=cfg.device, viz=cfg.viz)

    log = cfg.log
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    wandb_logger = None
    if log:
        wandb_logger = wandb.init(project=cfg.project,
                                  group=cfg.group,
                                  name=log_name)

    task = PPOTaskBase(cfg=cfg, env=env, wandb_logger=wandb_logger)
    save_dir = os.path.join(Path.home(), cfg.log_dir, log_name, 'checkpoints')

    task.train_loop(num_learning_iterations=cfg.num_learning_iterations,
                    save_dir=save_dir,
                    ckpt_path=None)
    if log:
        wandb.finish()


if __name__ == '__main__':
    train()
