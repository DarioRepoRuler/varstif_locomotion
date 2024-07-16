from model.replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
from model.actor_critic import ActorCritic


class PPO(nn.Module):
    def __init__(self,
                 cfg,
                 num_envs,
                 episode_length,
                 num_actions,
                 num_single_obs,
                 num_env_obs,
                 num_priv_obs,
                 num_robots=1,
                 device='cpu'
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.episode_length = episode_length
        self.use_encoder_decoder = cfg.use_encoder_decoder

        # PPO compoments
        self.actor_critic = ActorCritic(self.cfg,
                                        num_single_obs=num_single_obs,
                                        num_obs=num_env_obs,
                                        num_priv_obs=num_priv_obs,
                                        num_actions=num_actions
                                        ).to(self.device)

        self.storage = ReplayBuffer(num_envs=num_envs,
                                    num_transitions_per_env=episode_length * num_robots,
                                    num_obs=num_env_obs,
                                    num_priv_obs=num_priv_obs,
                                    num_actions=num_actions,
                                    num_robots=num_robots,
                                    device=self.device)
        self.transition = ReplayBuffer.Transition()
        params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.cfg.lr)
        
        if self.use_encoder_decoder:
            params = list(self.actor_critic.encoder.parameters()) + list(self.actor_critic.decoder.parameters())
            self.encode_decoder_optimizer = optim.Adam(params, lr=self.cfg.lr_encoder)

    def act(self, obs_g, priv_obs_g):
        # Compute the actions and values
        if self.use_encoder_decoder:
            self.transition.actions, _ = self.actor_critic.act(obs_g)
            #self.transition.priv_estimations = self.transition.priv_estimations.detach()

        else:
            self.transition.actions= self.actor_critic.act(obs_g)
        self.transition.actions = self.transition.actions.detach()

        #print(f"Actions shape: {self.transition.actions.shape}")
        self.transition.values = self.actor_critic.evaluate(priv_obs_g).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.actions

    def act_eval(self, obs_g, priv_obs_g):
        if self.use_encoder_decoder:
            self.transition.actions, _ = self.actor_critic.act(obs_g)
            #self.transition.priv_estimations = self.transition.priv_estimations.detach()
        else:
            self.transition.actions= self.actor_critic.act(obs_g)
        self.transition.actions = self.transition.actions.detach()
        self.transition.values = self.actor_critic.evaluate(priv_obs_g).detach() # calls just the critic
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.action_mean

    def inference(self, obs_g):
        return self.actor_critic.act_inference(obs_g)
    

    def process_env_step(self, obs, priviledged_obs_g,rewards, dones, infos):
        self.transition.observations = obs.detach()
        self.transition.priv_obs = priviledged_obs_g.detach()
        self.transition.rewards = rewards.detach()
        self.transition.dones = dones.detach()
        self.transition.progress = infos["step"].detach() / self.episode_length

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_obs_g):
        last_values = self.actor_critic.evaluate(last_obs_g.detach().clone()).detach()
        return self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lamb)

    def update(self):
        mean_value_loss = 0
        mean_actor_loss = 0
        mean_denoising_loss = 0

        generator = self.storage.mini_batch_generator(num_batches=self.cfg.num_batches, num_epochs=self.cfg.num_epochs)

        num_update = 1
        for obs_batch, priv_obs_batch, actions_batch, target_values_batch , \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:

            if self.use_encoder_decoder:
                self.actor_critic.act(obs_batch)
                #latent_batch, state_estimation_batch = self.actor_critic.get_states(obs_batch)
            else:
                self.actor_critic.act(obs_batch)
                state_estimation_batch = None
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(priv_obs_batch)
        
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.cfg.desired_kl != None and self.cfg.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                                   (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                                   (2.0 * torch.square(sigma_batch)) - 0.5,
                                   dim=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.cfg.lr = max(1e-5, self.cfg.lr / 1.5)
                    elif (self.cfg.desired_kl / 2.0) > kl_mean > 0.0:
                        self.cfg.lr = min(1e-2, self.cfg.lr * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.cfg.lr

            # Actor loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

            actor_loss = -torch.squeeze(advantages_batch) * ratio
            actor_loss_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg.clip_param,
                                                                                1.0 + self.cfg.clip_param)
            actor_loss = torch.max(actor_loss, actor_loss_clipped).mean()

            if self.cfg.use_clipped_value_loss:
                value_clipped = (target_values_batch +
                                 (value_batch - target_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param))
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            #print(f"Value loss dim: {(returns_batch - value_batch).pow(2).shape}")
            # print(f"entropy batch dim: {entropy_batch.shape}")
            loss = actor_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy_batch.mean()

            # Gradient step
            params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
            self.optimizer.step()    


            # State estimation loss
            if self.use_encoder_decoder:
                for epoch in range(self.cfg.num_epochs_encoder):
                    latent_batch, state_estimation_batch = self.actor_critic.get_states(obs_batch)
                    denoising_loss = self.cfg.denoise_loss_coef *(state_estimation_batch - priv_obs_batch).pow(2).sum(axis=1).mean() + self.cfg.latent_loss_coef * torch.abs(latent_batch).sum(axis=1).mean()

                    params = list(self.actor_critic.encoder.parameters()) + list(self.actor_critic.decoder.parameters())

                    self.encode_decoder_optimizer.zero_grad()
                    denoising_loss.backward()
                    nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                    self.encode_decoder_optimizer.step()

                    #loss = loss + denoising_loss
                    mean_denoising_loss += denoising_loss.item()
                    

            mean_value_loss += value_loss.item()
            mean_actor_loss += actor_loss.item()
            
            num_update += 1

        mean_value_loss /= num_update
        mean_actor_loss /= num_update

        mean_denoising_loss /= num_update

        self.storage.clear()

        return mean_value_loss, mean_actor_loss, mean_denoising_loss