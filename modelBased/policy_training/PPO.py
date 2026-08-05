import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    # def act(self, state):

    #     if self.has_continuous_action_space:
    #         action_mean = self.actor(state)
    #         cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    #         dist = MultivariateNormal(action_mean, cov_mat)
    #     else:
    #         action_probs = self.actor(state)
    #         dist = Categorical(action_probs)

    #     action = dist.sample()
    #     action_logprob = dist.log_prob(action)
    #     state_val = self.critic(state)

    #     return action.detach(), action_logprob.detach(), state_val.detach()


    def _discrete_distribution(self, state, epsilon=0.1, forward_bias=0.6):
        """Build the behavior distribution used by both rollout and PPO update."""
        action_probs = self.actor(state)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        epsilon = min(max(float(epsilon), 0.0), 1.0)
        if epsilon == 0.0:
            return Categorical(action_probs)

        # Keep exploration in the final action dimension. For a single state,
        # action_probs has shape (1, action_dim), so len(action_probs) is the
        # batch size and must not be used as the number of actions.
        num_actions = action_probs.shape[-1]
        if num_actions <= 1:
            explore_probs = torch.ones_like(action_probs)
        else:
            forward_bias = min(max(float(forward_bias), 0.0), 1.0)
            explore_probs = torch.full_like(
                action_probs,
                (1.0 - forward_bias) / (num_actions - 1),
            )
            forward_action_index = min(2, num_actions - 1)
            explore_probs[..., forward_action_index] = forward_bias

        # Sampling from the marginal mixture makes epsilon exploration part of
        # the PPO behavior policy. evaluate() uses the same distribution, so
        # stored and recomputed log-probabilities remain consistent.
        behavior_probs = (1.0 - epsilon) * action_probs + epsilon * explore_probs
        behavior_probs = behavior_probs / behavior_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        return Categorical(behavior_probs)

    def act(self, state, epsilon=0.1, forward_bias=0.6):
        state = state.to(device)  # Ensure state is on GPU

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
        else:
            dist = self._discrete_distribution(
                state,
                epsilon=epsilon,
                forward_bias=forward_bias,
            )
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()




    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            dist = self._discrete_distribution(state)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        entropy_coef=0.01,
        normalize_advantages=False,
        normalize_returns=True,
        max_grad_norm=0.0,
    ):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = float(entropy_coef)
        self.normalize_advantages = bool(normalize_advantages)
        self.normalize_returns = bool(normalize_returns)
        self.max_grad_norm = float(max_grad_norm)

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def reset_actor_critic(self):
        """
        Reinitialize PPO after its state representation changes.

        Both actor and critic consume the world-model feature vector, so both
        networks and Adam's accumulated moments must be reset together. The
        rollout buffer is also invalid under the new representation.
        """
        def _reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.buffer.clear()
        self.policy.apply(_reset_module)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            return action.detach().cpu().numpy().flatten(), state, action, action_logprob, state_val

        else:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)
            return action.item(), state, action, action_logprob, state_val

    def save_buffer(self, state=None, action=None, logprob=None, state_value=None, reward=None, is_terminal=None):
        def _buffer_tensor(x, ensure_1d=False):
            if torch.is_tensor(x):
                x = x.detach()
            else:
                x = torch.as_tensor(x)
            if ensure_1d and x.ndim == 0:
                x = x.unsqueeze(0)
            # Keep the rollout buffer on one device. Policy inference may
            # receive states from either the CPU collector or a CUDA world
            # model, so retaining the source device makes torch.stack fail.
            return x.cpu()

        self.buffer.states.append(_buffer_tensor(state))
        self.buffer.actions.append(_buffer_tensor(action, ensure_1d=True))
        self.buffer.logprobs.append(_buffer_tensor(logprob, ensure_1d=True))
        self.buffer.state_values.append(_buffer_tensor(state_value, ensure_1d=True))
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)
        

    def estimate_old_value(self, state):
        """Estimate V(s) with the behavior policy used for rollout collection."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            value = self.policy_old.critic(state.to(device))
        return float(value.reshape(-1)[0].detach().cpu().item())

    def update(self, bootstrap_value=0.0):
        """
        Update PPO from the current rollout buffer.

        ``bootstrap_value`` is V(next_state) for a non-terminal, fixed-horizon
        truncation. Real terminal markers inside the buffer still reset the
        return to zero. This lets P2E update within one environment without
        pretending that the environment ended at each online update boundary.
        """
        # Monte Carlo estimate of returns, optionally bootstrapped at a
        # non-terminal rollout boundary.
        rewards = []
        discounted_reward = float(bootstrap_value)
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Return normalization is retained by default for backwards
        # compatibility. P2E disables it so critic values and bootstrap values
        # remain on the same intrinsic-return scale.
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if self.normalize_returns:
            rewards = (
                rewards - rewards.mean()
            ) / (rewards.std(unbiased=False) + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - self.entropy_coef * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def preprocess_observation(obs):
    obs = obs / np.array([10, 5, 2])
    return torch.from_numpy(obs.flatten()).float().to(device)
