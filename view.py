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

# Define the configuration for training, later used.
@hydra.main(config_path='config', config_name='test', version_base="1.2")


def train(cfg: DictConfig):
    """
    Train the model using the provided configuration.

    Args:
        cfg (DictConfig): The configuration for training.
    """

    # Create the environment
    env0 = GO2Env(cfg.env)
    env = VmapWrapper(env0, batch_size=1) # alter batch size
    env = AutoResetWrapper(env) # reset env
    env = TorchWrapper(env, device=cfg.device, backend='gpu') # set device

    # Setup environment body(mujoco.MjModel) and data mujoco.MjData(mj_model)
    m = env0.model
    d = env0.data

    # Reset environment
    obs = env.reset()

    # Launch and step through the environment
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            # Utilize time tracking for logging and termination
            start = time.time()
            # Step through the environment
            obs, reward, done, info = env.step(torch.zeros((1, 12), device=cfg.device))
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