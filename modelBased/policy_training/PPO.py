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
        self.valid_masks = []
        self._transition_count = 0
        # Batched imagined rollouts live on CUDA.  Keeping this accumulator on
        # the same device avoids a host synchronization for every environment
        # step.  ``_transition_count`` is the most recently synchronized exact
        # value, while ``_possible_transition_count`` is a cheap upper bound.
        self._valid_transition_count_tensor = None
        self._possible_transition_count = 0
        self._synced_possible_transition_count = 0

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.valid_masks[:]
        self._transition_count = 0
        self._valid_transition_count_tensor = None
        self._possible_transition_count = 0
        self._synced_possible_transition_count = 0

    def transition_count(self):
        """Return transitions, not merely the number of temporal batches."""
        self._synchronize_transition_count()
        return self._transition_count

    def _synchronize_transition_count(self):
        """Synchronize the exact batched valid-transition count only on demand."""
        if self._valid_transition_count_tensor is not None:
            self._transition_count = int(
                self._valid_transition_count_tensor.detach().cpu().item()
            )
            self._synced_possible_transition_count = self._possible_transition_count

    def has_at_least_transitions(self, threshold):
        """Precisely test a rollout threshold with no unnecessary CUDA sync.

        Before the Python upper bound reaches ``threshold`` the answer is
        known to be false.  Once it might be true, synchronize exactly once;
        after an unsuccessful check, the cached exact count plus the number of
        newly appended slots provides another safe no-sync fast path.
        """
        threshold = int(threshold)
        if threshold <= 0:
            return True
        if self._possible_transition_count < threshold:
            return False
        unsynced_possible = (
            self._possible_transition_count - self._synced_possible_transition_count
        )
        if self._transition_count + unsynced_possible < threshold:
            return False
        self._synchronize_transition_count()
        return self._transition_count >= threshold


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
        freeze_actor_until_first_reward=False,
        minibatch_size=None,
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
        # None or 0 preserves the historical full-rollout PPO update.
        self.minibatch_size = (
            0 if minibatch_size is None else max(0, int(minibatch_size))
        )
        self.freeze_actor_until_first_reward = bool(
            freeze_actor_until_first_reward
        )
        # A random, untrained critic is not evidence that one action is better
        # than another.  Sparse-reward runs keep the actor fixed until the
        # collector has observed at least one genuine reward signal.
        self.has_seen_reward_signal = False

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
        self.has_seen_reward_signal = False

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
        self.buffer.valid_masks.append(True)
        self.buffer._possible_transition_count += 1
        if self.buffer._valid_transition_count_tensor is None:
            self.buffer._transition_count += 1
            self.buffer._synced_possible_transition_count += 1
        else:
            # A mixed batch/single rollout keeps the GPU accumulator as the
            # source of truth; leave the cached Python count untouched until
            # the next exact synchronization.
            self.buffer._valid_transition_count_tensor = (
                self.buffer._valid_transition_count_tensor + 1
            )

    def save_buffer_batch(
        self,
        states,
        actions,
        logprobs,
        state_values,
        rewards,
        is_terminals,
        valid_mask=None,
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
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=states.device)
        tensors["valid_mask"] = valid_mask
        for name, value in tensors.items():
            value = torch.as_tensor(value)
            if value.reshape(-1).numel() != batch_size:
                raise ValueError(
                    f"Parallel PPO {name} has {value.reshape(-1).numel()} values; "
                    f"expected batch size {batch_size}"
                )

        buffer_device = states.device
        self.buffer.states.append(states.detach())
        self.buffer.actions.append(
            actions.detach().to(buffer_device).reshape(batch_size)
        )
        self.buffer.logprobs.append(
            logprobs.detach().to(buffer_device).reshape(batch_size)
        )
        self.buffer.state_values.append(
            state_values.detach().to(buffer_device).reshape(batch_size)
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
        self.buffer.valid_masks.append(
            torch.as_tensor(
                valid_mask, dtype=torch.bool, device=buffer_device
            ).reshape(batch_size)
        )
        self.buffer._possible_transition_count += batch_size
        valid_count = self.buffer.valid_masks[-1].sum()
        if self.buffer._valid_transition_count_tensor is None:
            self.buffer._valid_transition_count_tensor = (
                valid_count + torch.as_tensor(
                    self.buffer._transition_count,
                    device=valid_count.device,
                    dtype=valid_count.dtype,
                )
            )
        else:
            self.buffer._valid_transition_count_tensor = (
                self.buffer._valid_transition_count_tensor + valid_count
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

    def behavior_clone(
        self,
        states,
        actions,
        epochs=500,
        batch_size=256,
        learning_rate=3e-4,
        target_accuracy=0.995,
        target_probability=0.995,
    ):
        """Warm-start the discrete actor from verified planner transitions.

        This updates only the actor. The critic remains untrained until PPO
        sees environment returns, and ``has_seen_reward_signal`` deliberately
        remains false so the sparse-reward actor guard still applies.
        """
        if self.has_continuous_action_space:
            raise ValueError("Behavior cloning currently supports discrete PPO only")
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1)
        if states.ndim != 2:
            raise ValueError(
                f"BC states must have shape [N,state_dim], got {tuple(states.shape)}"
            )
        if len(states) != len(actions) or not len(states):
            raise ValueError(
                f"BC requires equally sized non-empty states/actions, got "
                f"{len(states)} and {len(actions)}"
            )
        if int(actions.min()) < 0 or int(actions.max()) >= self.policy.action_dim:
            raise ValueError("BC actions contain an ID outside the PPO action space")

        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in self.policy.actor.named_parameters()
        }
        optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=float(learning_rate)
        )
        batch_size = max(1, min(int(batch_size), len(states)))
        epochs_completed = 0

        def evaluate_actor():
            with torch.no_grad():
                probabilities = self.policy.actor(states)
                chosen = probabilities.gather(1, actions[:, None]).squeeze(1)
                accuracy = (probabilities.argmax(dim=1) == actions).float().mean()
                loss = -chosen.clamp_min(1e-8).log().mean()
            return (
                float(loss.item()),
                float(accuracy.item()),
                float(chosen.mean().item()),
                float(chosen.min().item()),
            )

        for epoch in range(max(1, int(epochs))):
            permutation = torch.randperm(len(states), device=device)
            for start in range(0, len(states), batch_size):
                indices = permutation[start : start + batch_size]
                probabilities = self.policy.actor(states[indices])
                selected = probabilities.gather(
                    1, actions[indices, None]
                ).squeeze(1)
                loss = -selected.clamp_min(1e-8).log().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            epochs_completed = epoch + 1
            if epochs_completed % 10 == 0 or epochs_completed == int(epochs):
                _, accuracy, mean_probability, _ = evaluate_actor()
                if (
                    accuracy >= float(target_accuracy)
                    and mean_probability >= float(target_probability)
                ):
                    break

        loss, accuracy, mean_probability, min_probability = evaluate_actor()
        actor_delta = 0.0
        for name, parameter in self.policy.actor.named_parameters():
            actor_delta += float(
                (parameter.detach() - actor_before[name]).norm(2).item() ** 2
            )
        actor_delta **= 0.5

        # Rollout sampling always uses policy_old. Synchronize only the actor;
        # the original random critic is intentionally untouched.
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())
        # If this method is used after previous training, stale Adam moments
        # must not immediately undo the supervised initialization.
        actor_parameters = set(self.policy.actor.parameters())
        for parameter in list(self.optimizer.state):
            if parameter in actor_parameters:
                del self.optimizer.state[parameter]

        metrics = {
            "examples": len(states),
            "epochs": epochs_completed,
            "loss": loss,
            "accuracy": accuracy,
            "mean_target_probability": mean_probability,
            "min_target_probability": min_probability,
            "actor_parameter_delta": actor_delta,
        }
        print(
            "[PPO BC] "
            f"examples={len(states)} epochs={epochs_completed} "
            f"loss={loss:.6f} accuracy={accuracy:.2%} "
            f"mean_target_prob={mean_probability:.4f} "
            f"min_target_prob={min_probability:.4f} "
            f"actor_delta={actor_delta:.6e}"
        )
        return metrics

    def update(self, bootstrap_value=0.0, collect_metrics=True):
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

        if parallel_rollout:
            rewards_tb = torch.stack(self.buffer.rewards, dim=0).float()
            valid_tb = torch.stack(self.buffer.valid_masks, dim=0).to(
                rewards_tb.device
            ).bool()
            reward_signal_in_rollout = bool(
                rewards_tb[valid_tb].detach().abs().gt(1e-12).any().item()
            )
        else:
            reward_signal_in_rollout = any(
                abs(float(torch.as_tensor(reward).detach().reshape(-1)[0].item()))
                > 1e-12
                for reward in self.buffer.rewards
            )
        if reward_signal_in_rollout:
            self.has_seen_reward_signal = True
        actor_update_enabled = (
            not self.freeze_actor_until_first_reward
            or self.has_seen_reward_signal
        )
        # Before any real reward has been observed, a bootstrap value comes
        # only from the random critic itself.  Do not train the critic to
        # reproduce that arbitrary initial value; use the known zero-return
        # target until an external reward grounds the value function.
        zero_unrewarded_bootstrap = (
            self.freeze_actor_until_first_reward
            and not self.has_seen_reward_signal
        )

        # Monte Carlo returns, optionally bootstrapped at a non-terminal
        # rollout boundary. For vectorized environments, every column is an
        # independent trajectory and terminal markers reset only that column.
        if parallel_rollout:
            terminals_tb = torch.stack(self.buffer.is_terminals, dim=0).to(
                rewards_tb.device
            ).bool()
            state_values_tb = torch.stack(
                self.buffer.state_values, dim=0
            ).to(rewards_tb.device).float()
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
            if zero_unrewarded_bootstrap:
                discounted_reward.zero_()
            returns = []
            for reward_t, terminal_t, valid_t, state_value_t in zip(
                reversed(rewards_tb),
                reversed(terminals_tb),
                reversed(valid_tb),
                reversed(state_values_tb),
            ):
                invalid_bootstrap = (
                    torch.zeros_like(state_value_t)
                    if zero_unrewarded_bootstrap
                    else state_value_t.detach()
                )
                discounted_reward = torch.where(
                    valid_t,
                    discounted_reward,
                    invalid_bootstrap,
                )
                discounted_reward = torch.where(
                    terminal_t & valid_t,
                    torch.zeros_like(discounted_reward),
                    discounted_reward,
                )
                valid_return = reward_t + self.gamma * discounted_reward
                discounted_reward = torch.where(
                    valid_t,
                    valid_return,
                    discounted_reward,
                )
                returns.append(discounted_reward)
            valid_flat = valid_tb.reshape(-1)
            rewards = torch.stack(list(reversed(returns)), dim=0).reshape(-1)
            rewards = rewards[valid_flat].to(device)
        else:
            rewards = []
            discounted_reward = (
                0.0 if zero_unrewarded_bootstrap else float(bootstrap_value)
            )
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
                -1, self.buffer.states[0].shape[-1]
            )[valid_flat].detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).reshape(-1)[valid_flat].detach().to(device)
            old_logprobs = torch.stack(self.buffer.logprobs, dim=0).reshape(-1)[valid_flat].detach().to(device)
            old_state_values = torch.stack(
                self.buffer.state_values, dim=0
            ).reshape(-1)[valid_flat].detach().to(device)
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

        collect_metrics = bool(collect_metrics)
        # Parameter copies and scalar host transfers are diagnostic-only.  Do
        # not perform them on updates whose metrics will not be consumed.
        parameters_before = (
            {
                name: parameter.detach().clone()
                for name, parameter in self.policy.named_parameters()
            }
            if collect_metrics
            else None
        )
        last_loss = last_grad_norm = last_actor_loss = None
        last_critic_loss = last_entropy = None
        num_samples = int(old_states.shape[0])
        minibatch_size = self.minibatch_size or num_samples
        minibatch_size = min(minibatch_size, num_samples)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Preserve the historical full-batch execution order when the
            # feature is disabled. Minibatch PPO shuffles independently at
            # every epoch as required by the standard update.
            permutation = (
                torch.randperm(num_samples, device=device)
                if self.minibatch_size
                else torch.arange(num_samples, device=device)
            )
            if collect_metrics:
                metric_weight = torch.zeros((), device=device)
                metric_loss = torch.zeros((), device=device)
                metric_actor_loss = torch.zeros((), device=device)
                metric_critic_loss = torch.zeros((), device=device)
                metric_entropy = torch.zeros((), device=device)
                metric_grad_norm = torch.zeros((), device=device)
            for start in range(0, num_samples, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                batch_states = old_states[indices]
                batch_actions = old_actions[indices]
                batch_logprobs = old_logprobs[indices]
                batch_rewards = rewards[indices]
                batch_advantages = advantages[indices]

                # Evaluating old actions and values for this shuffled minibatch.
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - batch_logprobs.detach())

                critic_loss = 0.5 * self.MseLoss(state_values, batch_rewards)
                if actor_update_enabled:
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(
                        ratios, 1 - self.eps_clip, 1 + self.eps_clip
                    ) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist_entropy.mean()
                    loss_mean = (
                        actor_loss + critic_loss - self.entropy_coef * entropy
                    )
                else:
                    # Actor and critic are disjoint modules. Omitting both the
                    # surrogate and entropy terms prevents an ungrounded actor
                    # update before a genuine reward has been observed.
                    actor_loss = torch.zeros((), device=state_values.device)
                    entropy = torch.zeros((), device=state_values.device)
                    loss_mean = critic_loss

                self.optimizer.zero_grad()
                loss_mean.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                ) if self.max_grad_norm > 0.0 else None
                if collect_metrics:
                    if self.max_grad_norm > 0.0:
                        grad_norm_value = grad_norm.detach()
                    else:
                        grad_squared_norm = torch.zeros((), device=device)
                        for parameter in self.policy.parameters():
                            if parameter.grad is not None:
                                grad_squared_norm = grad_squared_norm + parameter.grad.detach().square().sum()
                        grad_norm_value = grad_squared_norm.sqrt()
                self.optimizer.step()
                if collect_metrics:
                    weight = torch.as_tensor(len(indices), device=device)
                    metric_weight += weight
                    metric_loss += loss_mean.detach() * weight
                    metric_actor_loss += actor_loss.detach() * weight
                    metric_critic_loss += critic_loss.detach() * weight
                    metric_entropy += entropy.detach() * weight
                    metric_grad_norm += grad_norm_value * weight

            # Report a sample-weighted mean across all optimizer steps in the
            # final epoch, rather than an arbitrary last minibatch.
            if collect_metrics:
                # Only one host synchronization per reported PPO update.
                metrics_tensor = torch.stack((
                    metric_loss / metric_weight,
                    metric_actor_loss / metric_weight,
                    metric_critic_loss / metric_weight,
                    metric_entropy / metric_weight,
                    metric_grad_norm / metric_weight,
                ))
                (
                    last_loss,
                    last_actor_loss,
                    last_critic_loss,
                    last_entropy,
                    last_grad_norm,
                ) = metrics_tensor.detach().cpu().tolist()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        parameter_delta = actor_parameter_delta = critic_parameter_delta = None
        if collect_metrics:
            parameter_delta_sq = torch.zeros((), device=device)
            actor_delta_sq = torch.zeros((), device=device)
            critic_delta_sq = torch.zeros((), device=device)
            for name, parameter in self.policy.named_parameters():
                squared_delta = (parameter.detach() - parameters_before[name]).square().sum()
                parameter_delta_sq += squared_delta
                if name.startswith("actor."):
                    actor_delta_sq += squared_delta
                elif name.startswith("critic."):
                    critic_delta_sq += squared_delta
            (
                parameter_delta,
                actor_parameter_delta,
                critic_parameter_delta,
            ) = torch.stack((
                parameter_delta_sq.sqrt(), actor_delta_sq.sqrt(), critic_delta_sq.sqrt()
            )).detach().cpu().tolist()
        self.update_count += 1
        metrics = {
            "updated": True,
            "update_count": self.update_count,
            "rollout_size": rollout_size,
            "loss": last_loss,
            "grad_norm": last_grad_norm,
            "parameter_delta": parameter_delta,
            "actor_parameter_delta": actor_parameter_delta,
            "critic_parameter_delta": critic_parameter_delta,
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "entropy": last_entropy,
            "actor_update_enabled": actor_update_enabled,
            "reward_signal_in_rollout": reward_signal_in_rollout,
            "has_seen_reward_signal": self.has_seen_reward_signal,
        }
        if collect_metrics:
            print(
                f"[PPO UPDATE #{self.update_count}] rollout={rollout_size} "
                f"loss={last_loss:.6f} grad_norm={last_grad_norm:.6f} "
                f"actor_delta={actor_parameter_delta:.6e} "
                f"critic_delta={critic_parameter_delta:.6e} "
                f"actor_update={actor_update_enabled} "
                f"reward_signal={reward_signal_in_rollout}"
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
        # A loaded policy is assumed to have already been trained on grounded
        # returns. Evaluation is unaffected, and resumed training must not
        # unexpectedly re-freeze a previously learned actor.
        self.has_seen_reward_signal = True


def preprocess_observation(
    obs,
    obs_norm_values=(10, 5, 3),
    inventory_token=None,
    inventory_classes=7,
):
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
    state_tensor = torch.as_tensor(
        normalized.flatten(), dtype=torch.float32, device=device
    )
    if inventory_token is not None:
        inventory = torch.nn.functional.one_hot(
            torch.as_tensor(inventory_token, device=device).long(),
            num_classes=int(inventory_classes),
        ).float().reshape(-1)
        state_tensor = torch.cat((state_tensor, inventory), dim=0)
    return state_tensor
