import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import utils
from .replay_buffer import PrioritizedReplayBuffer


class Encoder(nn.Module):
    """Encodes the observation to feed into respective environment networks"""
    def __init__(self, input_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.fc = utils.mlp(input_shape, hidden_dim, hidden_dim, hidden_depth)
        self.feature_dim = hidden_dim

        self.outputs = dict()

    def forward(self, obs):
        obs = obs.reshape(obs.shape[0], -1)
        self.outputs['obs'] = obs

        out = self.fc(obs)
        self.outputs['fc'] = out

        return out

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
        
        for i, m in enumerate(self.fc):
            if type(m) is nn.Linear:
                logger.log_param(f'train_encoder/fc_{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(
        self, observation_shape, action_shape, num_env_paths, encoder_hidden_dim,
        encoder_hidden_depth, critic_hidden_dim, critic_hidden_depth, dueling
    ):
        super().__init__()

        self.num_env_paths = num_env_paths
        self.dueling = dueling
        self.action_shape = action_shape

        self.encoder = Encoder(observation_shape, encoder_hidden_dim, encoder_hidden_depth)

        if dueling:
            # Dueling DQN: define the value and the advantage network
            self.V = self._create_env_mlp(critic_hidden_dim, critic_hidden_depth)
            self.A = self._create_env_mlp(critic_hidden_dim, critic_hidden_depth)
        else:
            self.Q = self._create_env_mlp(critic_hidden_dim, critic_hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)
    
    def _create_env_mlp(self, hidden_dim, hidden_depth):
        return nn.ModuleList([
            utils.mlp(self.encoder.feature_dim, hidden_dim, self.action_shape, hidden_depth)
            for _ in range(self.num_env_paths)
        ])

    def forward(self, obs, env_path):

        obs = self.encoder(obs)

        if self.dueling:
            # Dueling DQN: compute the q value from the value and advantage network
            v = self.V[env_path](obs)
            a = self.A[env_path](obs)
            q = v + a - a.mean()
        else:
            q = self.Q[env_path](obs)

        self.outputs['q'] = q

        return q

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        M = self.A if self.dueling else self.Q
        for i, m in enumerate(M):
            if type(m) is nn.Linear:
                logger.log_param(f'train_critic/q_fc{i}', m, step)


class DRQLAgent(object):
    """Data regularized Q-learning: Deep Q-learning."""
    def __init__(
        self, observation_shape, action_shape, num_env_paths, encoder_config, critic_config,
        device, discount, learning_rate, beta_1, beta_2, weight_decay, adam_eps, max_grad_norm,
        critic_tau, critic_target_update_frequency, batch_size, multistep_return, eval_eps,
        double_q, prioritized_replay_beta0, prioritized_replay_beta_steps
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.action_shape = action_shape
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.eval_eps = eval_eps
        self.max_grad_norm = max_grad_norm
        self.multistep_return = multistep_return
        self.double_q = double_q
        assert prioritized_replay_beta0 <= 1.0
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.eps = 0

        self.critic = Critic(
            observation_shape, action_shape, num_env_paths, encoder_config.hidden_dim,
            encoder_config.hidden_depth, critic_config.hidden_dim, critic_config.hidden_depth,
            critic_config.dueling
        ).to(self.device)
        self.critic_target = Critic(
            observation_shape, action_shape, num_env_paths, encoder_config.hidden_dim,
            encoder_config.hidden_depth, critic_config.hidden_dim, critic_config.hidden_depth,
            critic_config.dueling
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            weight_decay=weight_decay,
            eps=adam_eps
        )

        self.train()
        self.critic_target.train()
    
    def _update_sample(self, update_value, sample):
        return update_value if sample is None else torch.cat([sample, update_value], dim=0)
    
    def _reorder_samples(self, obs, action, reward, next_obs, not_done, env_paths, weights=None, idxs=None):
        """Reorder the samples on the basis of env_paths"""
        obs_order, action_order, reward_order, next_obs_order, not_done_order, weights_order, idxs_order = [], None, None, [], None, None, None
        if idxs is not None:
            idxs = torch.tensor(idxs, device=self.device)
        for env_idx in range(self.critic.num_env_paths):
            path_idxs = (env_paths == env_idx).nonzero()[:, 0]
            if len(path_idxs) == 0:
                continue
            obs_order.append(obs[path_idxs])
            action_order = self._update_sample(action[path_idxs], action_order)
            reward_order = self._update_sample(reward[path_idxs], reward_order)
            next_obs_order.append(next_obs[path_idxs])
            not_done_order = self._update_sample(not_done[path_idxs], not_done_order)

            if weights is not None:
                weights_order = self._update_sample(weights[path_idxs], weights_order)
            if idxs is not None:
                idxs_order = self._update_sample(idxs[path_idxs], idxs_order)
        
        if idxs_order is not None:
            idxs_order = idxs_order.detach().cpu().numpy()
        return obs_order, action_order, reward_order, next_obs_order, not_done_order, weights_order, idxs_order
    
    def _critic_batch(self, critic, obs):
        outputs = critic(obs[0], 0)
        for env_idx in range(1, len(obs)):
            outputs = torch.cat((outputs, critic(obs[env_idx], env_idx)), dim=0)
        return outputs

    def train(self, training=True):
        self.training = training
        self.critic.train(training)
    
    def load(self, checkpoint):
        self.critic.load_state_dict(torch.load(checkpoint))
        self.critic_target.load_state_dict(torch.load(checkpoint))

    def act(self, obs, env_path):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0).contiguous()
            q = self.critic(obs, env_path)
            action = q.max(dim=1)[1].item()
        return action

    def update_critic(self, obs, action, reward, next_obs, not_done, weights, logger, step):
        with torch.no_grad():
            discount = self.discount**self.multistep_return
            if self.double_q:
                # Double Q Learning
                # Find the target Q value based on the critic
                # and the critic target networks to find the right
                # value of target_Q
                next_Q_critic = self._critic_batch(self.critic, next_obs)
                next_action_critic = next_Q_critic.max(dim=1)[1].unsqueeze(1)

                next_Q = self._critic_batch(self.critic_target, next_obs)
                next_Q = next_Q.gather(1, next_action_critic)
                target_Q = reward + (not_done * discount * next_Q)
            else:
                next_Q = self._critic_batch(self.critic_target, next_obs)
                next_Q = next_Q.max(dim=1)[0].unsqueeze(1)
                target_Q = reward + (not_done * discount * next_Q)

        # get current Q estimates
        current_Q = self._critic_batch(self.critic, obs)
        current_Q = current_Q.gather(1, action)

        td_errors = current_Q - target_Q
        critic_losses = F.smooth_l1_loss(current_Q, target_Q, reduction='none')
        if weights is not None:
            critic_losses *= weights

        critic_loss = critic_losses.mean()

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.critic.log(logger, step)

        return td_errors.squeeze(dim=1).detach().cpu().numpy()

    def update(self, replay_buffer, logger, step):

        prioritized_replay = type(replay_buffer) == PrioritizedReplayBuffer

        if prioritized_replay:
            fraction = min(step / self.prioritized_replay_beta_steps, 1.0)
            beta = self.prioritized_replay_beta0 + fraction * (1.0 - self.prioritized_replay_beta0)
            obs, action, reward, next_obs, not_done, envs, weights, idxs = replay_buffer.sample_multistep(
                self.batch_size, beta, self.discount, self.multistep_return
            )
        else:
            obs, action, reward, next_obs, not_done, envs = replay_buffer.sample_multistep(
                self.batch_size, self.discount, self.multistep_return
            )
            weights = None
            idxs = None
        
        # Reorder the samples on the basis of env_paths
        obs, action, reward, next_obs, not_done, weights, idxs = self._reorder_samples(
            obs, action, reward, next_obs, not_done, envs, weights, idxs
        )

        logger.log('train/batch_reward', reward.mean(), step)

        td_errors = self.update_critic(obs, action, reward, next_obs, not_done, weights, logger, step)

        if prioritized_replay:
            # Prioritized Replay Buffer: update the priorities in the replay buffer using td_errors
            proportional_priority = np.abs(td_errors) + 1e-6
            replay_buffer.update_priorities(idxs, proportional_priority)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


def create_agent(config, env, device, num_envs=0):
    return DRQLAgent(
        int(np.prod(env.observation_space.shape)), env.action_space.n,
        env.num_envs if num_envs == 0 else num_envs,
        config.encoder, config.critic, device, **config.agent
    )
