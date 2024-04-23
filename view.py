import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper
import mujoco
import mujoco.viewer
import time


@hydra.main(config_path='config', config_name='test', version_base="1.2")

def train(cfg: DictConfig):
    env0 = GO2Env(cfg.env)
    env = VmapWrapper(env0, batch_size=1)
    env = AutoResetWrapper(env)
    env = TorchWrapper(env, device='cpu', backend='cpu')

    m = env0.model
    d = env0.data
    obs = env.reset()
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            start = time.time()
            obs, reward, done, info = env.step(torch.zeros((1, 12), device='cpu'))
            qpos = env.state.pipeline_state.qpos
            qvel = env.state.pipeline_state.qvel
            d.qpos, d.qvel = qpos[0], qvel[0]
            mujoco.mj_forward(m, d)
            v.sync()
            # elapsed = time.time() - start
            # if elapsed < 0.1:
            #     time.sleep(0.1 - elapsed)

if __name__ == '__main__':
    train()