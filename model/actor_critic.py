import numpy as np
import torch
import torch.nn as nn
from model.networks import MLP
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self,
                 config,
                 num_obs,
                 num_actions):
        super().__init__()

        self.actor = MLP(in_features=num_obs,
                         hidden_features=config.hidden_dim,
                         out_features=num_actions,
                         n_layers=config.n_layers,
                         act=nn.ELU(),
                         output_act=nn.Tanh(),
                         using_norm=False)

        self.critic = MLP(in_features=num_obs,
                          hidden_features=config.hidden_dim,
                          out_features=1,
                          n_layers=config.n_layers,
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)

        # Action distribution
        self.std = nn.Parameter(config.init_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)

        self.distribution = Normal(mean, torch.abs(self.std))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            actions_mean = self.actor(observations)

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
