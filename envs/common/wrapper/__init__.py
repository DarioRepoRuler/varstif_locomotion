
from envs.robots.go2_env import GO2Env
from envs.common.wrapper.torch_wrapper import TorchWrapper
from envs.common.wrapper.render_wrapper import RenderWrapper
from envs.common.wrapper.training_wrapper import VmapWrapper, AutoResetWrapper, wrap, domain_randomize


def _create_env(env, num_envs, device, viz=False, domain_cfg=None):
    """
    Create the environment with the specified number of environments and device.
    VmapWrapper->AutoResetWrapper->TorchWrapper(->RenderWrapper)

    Args:
        env (GO2Env): The environment to create.
        num_envs (int): The number of environments to create.
        device (str): The device to use for computation.
        viz (bool): Whether to render the environment.
    """

    if domain_cfg.randomisation:
        env=wrap(env, num_envs=num_envs, randomization_fn=domain_randomize, randomization_args=domain_cfg)
    else:
        env = wrap(env, num_envs=num_envs)
    if device == 'cpu':
        env = TorchWrapper(env, device=device, backend='cpu')
    else:
        env = TorchWrapper(env, device=device, backend='gpu')
    if viz:
        env = RenderWrapper(env, render_mode='human')
    return env