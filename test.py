import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper import _create_env
from tasks.PPOTaskBase import PPOTaskBase


@hydra.main(config_path='config', config_name='test', version_base="1.2")
def test(cfg: DictConfig):
    # Create the environment    
    task = PPOTaskBase(cfg=cfg)

    if cfg.ckpt_path is not None:
        # Get model path
        ckpt_path = os.path.join(os.getcwd(),cfg.ckpt_path)
        task.test_agent(num_iterations=cfg.num_iterations, ckpt_path=ckpt_path)
    else:
        task.test_agent(num_iterations=cfg.num_iterations)

if __name__ == '__main__':
    test()