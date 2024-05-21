import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper
from tasks.PPOTaskBase import PPOTaskBase


@hydra.main(config_path='config', config_name='test', version_base="1.2")
def test(cfg: DictConfig):
    env = GO2Env(cfg.env)
    env = VmapWrapper(env, batch_size=1)
    env = AutoResetWrapper(env)
    env = TorchWrapper(env, device='cuda:0', backend='gpu')
    env = RenderWrapper(env, render_mode='human')

    task = PPOTaskBase(cfg=cfg, env=env)
    task.test_agent(num_iterations=10, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    test()