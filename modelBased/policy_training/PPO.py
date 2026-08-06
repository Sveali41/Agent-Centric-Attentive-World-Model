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

    def transition_count(self):
        """Return transitions, not merely the number of temporal batches."""
        total = 0
        for reward in self.rewards:
            total += int(reward.numel()) if torch.is_tensor(reward) else 1
        return total


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


    def _discrete_distribution(self, state):
        """Build the categorical policy used by rollout and PPO update.

        Sampling from ``Categorical`` already provides exploration.  Do not
        mix in a separate epsilon/forward-biased distribution here: rollout
        log-probabilities and the probabilities recomputed by PPO must describe
        exactly the same policy.
        """
        action_probs = self.actor(state)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return Categorical(action_probs)

    def act(self, state):
        state = state.to(device)  # Ensure state is on GPU

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
        else:
            dist = self._discrete_distribution(state)
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
        # Number of successful optimizer updates.  This is intentionally kept
        # on the agent so callers can verify that imagined rollouts really
        # changed the policy, rather than only seeing environment timesteps.
        self.update_count = 0

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
        self.update_count = 0

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

    def select_action_batch(self, states):
        """Sample one action for every state in a vectorized environment."""
        if states.ndim != 2:
            raise ValueError(
                f"Expected batched states shaped (B, state_dim), got {tuple(states.shape)}"
            )
        with torch.no_grad():
            actions, action_logprobs, state_values = self.policy_old.act(states)
        return (
            actions.detach(),
            states.detach(),
            actions.detach(),
            action_logprobs.detach(),
            state_values.detach(),
        )

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

    def save_buffer_batch(
        self,
        states,
        actions,
        logprobs,
        state_values,
        rewards,
        is_terminals,
    ):
        """Save one temporal slice from B parallel trajectories.

        Entries retain their batch dimension in the buffer as ``[T, B, ...]``.
        ``update`` computes returns independently along T for every environment
        before flattening T and B for the standard PPO loss.
        """
        batch_size = int(states.shape[0])
        tensors = {
            "actions": actions,
            "logprobs": logprobs,
            "state_values": state_values,
            "rewards": rewards,
            "is_terminals": is_terminals,
        }
        for name, value in tensors.items():
            value = torch.as_tensor(value)
            if value.reshape(-1).numel() != batch_size:
                raise ValueError(
                    f"Parallel PPO {name} has {value.reshape(-1).numel()} values; "
                    f"expected batch size {batch_size}"
                )

        buffer_device = states.device
        self.buffer.states.append(states.detach())
        self.buffer.actions.append(actions.detach().reshape(batch_size))
        self.buffer.logprobs.append(logprobs.detach().reshape(batch_size))
        self.buffer.state_values.append(
            state_values.detach().reshape(batch_size)
        )
        self.buffer.rewards.append(
            torch.as_tensor(
                rewards, dtype=torch.float32, device=buffer_device
            ).reshape(batch_size)
        )
        self.buffer.is_terminals.append(
            torch.as_tensor(
                is_terminals, dtype=torch.bool, device=buffer_device
            ).reshape(batch_size)
        )
        

    def estimate_old_value(self, state):
        """Estimate V(s) with the behavior policy used for rollout collection."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            value = self.policy_old.critic(state.to(device))
        return float(value.reshape(-1)[0].detach().cpu().item())

    def estimate_old_values_batch(self, states):
        """Return V(s) for every state in a vectorized environment."""
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            values = self.policy_old.critic(states.to(device))
        return values.detach().reshape(-1).cpu()

    def update(self, bootstrap_value=0.0):
        """
        Update PPO from the current rollout buffer.

        ``bootstrap_value`` is V(next_state) for a non-terminal, fixed-horizon
        truncation. Real terminal markers inside the buffer still reset the
        return to zero. This lets P2E update within one environment without
        pretending that the environment ended at each online update boundary.
        """
        rollout_size = self.buffer.transition_count()
        if rollout_size == 0:
            return {
                "updated": False,
                "update_count": self.update_count,
                "rollout_size": 0,
                "reason": "empty_buffer",
            }

        parallel_rollout = torch.is_tensor(self.buffer.rewards[0])

        # Monte Carlo returns, optionally bootstrapped at a non-terminal
        # rollout boundary. For vectorized environments, every column is an
        # independent trajectory and terminal markers reset only that column.
        if parallel_rollout:
            rewards_tb = torch.stack(self.buffer.rewards, dim=0).float()
            terminals_tb = torch.stack(self.buffer.is_terminals, dim=0).bool()
            batch_size = rewards_tb.shape[1]
            discounted_reward = torch.as_tensor(
                bootstrap_value,
                dtype=torch.float32,
                device=rewards_tb.device,
            ).reshape(-1)
            if discounted_reward.numel() == 1:
                discounted_reward = discounted_reward.expand(batch_size).clone()
            elif discounted_reward.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} bootstrap values, got "
                    f"{discounted_reward.numel()}"
                )
            returns = []
            for reward_t, terminal_t in zip(
                reversed(rewards_tb), reversed(terminals_tb)
            ):
                discounted_reward = torch.where(
                    terminal_t,
                    torch.zeros_like(discounted_reward),
                    discounted_reward,
                )
                discounted_reward = reward_t + self.gamma * discounted_reward
                returns.append(discounted_reward)
            rewards = torch.stack(list(reversed(returns)), dim=0).reshape(-1).to(device)
        else:
            rewards = []
            discounted_reward = float(bootstrap_value)
            for reward, is_terminal in zip(
                reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
            ):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # Return normalization is retained by default for backwards
        # compatibility. P2E disables it so critic values and bootstrap values
        # remain on the same intrinsic-return scale.
        if self.normalize_returns:
            rewards = (
                rewards - rewards.mean()
            ) / (rewards.std(unbiased=False) + 1e-7)

        # convert list to tensor
        if parallel_rollout:
            old_states = torch.stack(self.buffer.states, dim=0).reshape(
                rollout_size, -1
            ).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).reshape(-1).detach().to(device)
            old_logprobs = torch.stack(self.buffer.logprobs, dim=0).reshape(-1).detach().to(device)
            old_state_values = torch.stack(
                self.buffer.state_values, dim=0
            ).reshape(-1).detach().to(device)
        else:
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

        # Track the parameters across the complete K-epoch update.  A positive
        # delta is a direct confirmation that the policy weights changed.
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in self.policy.named_parameters()
        }
        last_loss = None
        last_grad_norm = 0.0

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
            loss_mean = loss.mean()
            loss_mean.backward()
            last_loss = float(loss_mean.detach().cpu().item())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            ) if self.max_grad_norm > 0.0 else None
            if self.max_grad_norm > 0.0:
                last_grad_norm = float(grad_norm.detach().cpu().item())
            else:
                grad_squared_norm = 0.0
                for parameter in self.policy.parameters():
                    if parameter.grad is not None:
                        grad_squared_norm += float(parameter.grad.detach().norm(2).item() ** 2)
                last_grad_norm = grad_squared_norm ** 0.5
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        parameter_delta = 0.0
        for name, parameter in self.policy.named_parameters():
            parameter_delta += float(
                (parameter.detach() - parameters_before[name]).norm(2).item() ** 2
            )
        parameter_delta = parameter_delta ** 0.5
        self.update_count += 1
        metrics = {
            "updated": True,
            "update_count": self.update_count,
            "rollout_size": rollout_size,
            "loss": last_loss,
            "grad_norm": last_grad_norm,
            "parameter_delta": parameter_delta,
        }
        print(
            f"[PPO UPDATE #{self.update_count}] rollout={rollout_size} "
            f"loss={last_loss:.6f} grad_norm={last_grad_norm:.6f} "
            f"parameter_delta={parameter_delta:.6e}"
        )

        # clear buffer
        self.buffer.clear()
        return metrics

    def save(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(str(checkpoint_path))
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def preprocess_observation(obs, obs_norm_values=(10, 5, 3)):
    """Standardized MiniGrid observation preprocessing:
    1. Ensure channels-first (C, H, W) format.
    2. Normalize per channel using obs_norm_values (default [10, 5, 3]).
    3. Flatten into 1D float tensor on device.
    """
    from modelBased.common.utils import ColRowCanl_to_CanlRowCol, normalize_obs
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[0] != 3:
        state = ColRowCanl_to_CanlRowCol(obs)
    elif torch.is_tensor(obs) and obs.ndim == 3 and obs.shape[0] != 3:
        state = ColRowCanl_to_CanlRowCol(obs)
    else:
        state = obs
    normalized = normalize_obs(state.copy() if hasattr(state, 'copy') else state.clone(), obs_norm_values)
    return torch.as_tensor(normalized.flatten(), dtype=torch.float32, device=device)
