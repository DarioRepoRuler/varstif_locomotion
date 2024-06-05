import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper
from tasks.PPOTaskBase import PPOTaskBase
import numpy as np

@hydra.main(config_path='config', config_name='test', version_base="1.2")

def main(cfg: DictConfig):
    env = GO2Env(cfg.env,scene_xml=cfg.scene_xml)
    env = VmapWrapper(env, batch_size=1)
    env = AutoResetWrapper(env)
    env = TorchWrapper(env, device='cuda:0', backend='gpu')
    env = RenderWrapper(env, render_mode='human')
    
    for i in range(1,100):
        env.step(np.zeros(12))

if __name__ == '__main__':
    main()