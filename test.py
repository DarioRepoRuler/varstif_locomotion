import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper, wrap, domain_randomize
from tasks.PPOTaskBase import PPOTaskBase

def _create_env(env, num_envs, device, viz=False, randomisation=True):
    """
    Create the environment with the specified number of environments and device.
    VmapWrapper->AutoResetWrapper->TorchWrapper(->RenderWrapper)

    Args:
        env (GO2Env): The environment to create.
        num_envs (int): The number of environments to create.
        device (str): The device to use for computation.
        viz (bool): Whether to render the environment.
    """
    #env = VmapWrapper(env, batch_size=num_envs)
    #env = AutoResetWrapper(env)
    if randomisation:
        env=wrap(env, num_envs=num_envs, randomization_fn=domain_randomize)
    else:
        env = wrap(env, num_envs=num_envs)
        
    if device == 'cpu':
        env = TorchWrapper(env, device=device, backend='cpu')
    else:
        env = TorchWrapper(env, device=device, backend='gpu')
    if viz:
        env = RenderWrapper(env, render_mode='human')
    return env



@hydra.main(config_path='config', config_name='test', version_base="1.2")
def test(cfg: DictConfig):
    # Create the environment    
    env = _create_env(GO2Env(cfg.env, scene_xml=cfg.scene_xml), num_envs=cfg.num_envs, device=cfg.device, viz=True, randomisation=cfg.randomisation)

    task = PPOTaskBase(cfg=cfg, env=env)
    

    if cfg.ckpt_path is not None:
        # Get model path
        ckpt_path = os.path.join(os.getcwd(),cfg.ckpt_path)
        task.test_agent(num_iterations=cfg.num_iterations, ckpt_path=ckpt_path)
    else:
        task.test_agent(num_iterations=cfg.num_iterations)

if __name__ == '__main__':
    test()