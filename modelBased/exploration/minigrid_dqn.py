"""Spatial DQN backend for the fully-observed MiniGrid explorer.

This module intentionally keeps the PPO explorer untouched.  The DQN class
implements the same collector lifecycle and observation construction, so it
can be used as an optimizer control for reward diagnostics.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from modelBased.exploration.minigrid_rmax import (
    _DEVICE,
    _SpatialActorCritic,
    MiniGridRMaxExplorer,
    _pos,
)


_DQN_ARCH = "minigrid_spatial_dqn_v1"


class _SpatialQNetwork(_SpatialActorCritic):
    """Reuse the spatial encoder while interpreting the six outputs as Q-values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A random Q-head bias makes an unvisited action look optimal forever
        # under a short sparse-reward run.  Zero initialization gives all six
        # actions equal initial value; _greedy_action then breaks ties randomly.
        nn.init.zeros_(self.actor_head[-1].weight)
        nn.init.zeros_(self.actor_head[-1].bias)

    def forward(self, states):
        feature = self.encode(states)
        return self.actor_head(feature)


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def append(self, state, action, reward, next_state, terminal):
        self.data.append(
            (
                state.detach().cpu(),
                int(action),
                float(reward),
                next_state.detach().cpu(),
                bool(terminal),
            )
        )

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int, generator: torch.Generator):
        count = min(int(batch_size), len(self.data))
        indices = torch.randperm(len(self.data), generator=generator, device=_DEVICE)
        indices = indices[:count].detach().cpu().tolist()
        batch = [self.data[index] for index in indices]
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (
            torch.stack(states).to(_DEVICE),
            torch.as_tensor(actions, dtype=torch.long, device=_DEVICE),
            torch.as_tensor(rewards, dtype=torch.float32, device=_DEVICE),
            torch.stack(next_states).to(_DEVICE),
            torch.as_tensor(terminals, dtype=torch.bool, device=_DEVICE),
        )

    def state_dict(self):
        return list(self.data)

    def load_state_dict(self, values):
        self.data.clear()
        for state, action, reward, next_state, terminal in values:
            self.data.append(
                (
                    torch.as_tensor(state).detach().cpu(),
                    int(action),
                    float(reward),
                    torch.as_tensor(next_state).detach().cpu(),
                    bool(terminal),
                )
            )


class MiniGridDQNExplorer(MiniGridRMaxExplorer):
    """DQN replacement for :class:`MiniGridRMaxExplorer`.

    Map-local visited/frontier memory follows the PPO explorer.  The replay
    buffer and Q-network persist across maps, which makes this backend suitable
    for both fixed-map reward diagnostics and continual training.
    """

    def __init__(self, cfg):
        # The parent owns the observation encoder construction and lifecycle
        # invariants.  Replace its PPO object immediately after initialization;
        # no PPO buffer or optimizer is used by this backend.
        super().__init__(cfg)
        r = cfg.domains.minigrid.rmax_like
        self.ppo = None
        self.gamma = float(getattr(r, "gamma", 0.99))
        self.max_grad_norm = float(getattr(r, "max_grad_norm", 0.5))
        self._dqn_batch_size = int(getattr(r, "dqn_batch_size", 64))
        self._dqn_replay_capacity = int(getattr(r, "dqn_replay_capacity", 50_000))
        self._dqn_updates_per_iteration = int(
            getattr(r, "dqn_updates_per_iteration", 32)
        )
        self._dqn_target_update_interval = int(
            getattr(r, "dqn_target_update_interval", 32)
        )
        self._dqn_learning_rate = float(getattr(r, "dqn_learning_rate", 3e-4))
        self._dqn_epsilon_start = float(getattr(r, "dqn_epsilon_start", 1.0))
        self._dqn_epsilon_end = float(getattr(r, "dqn_epsilon_end", 0.05))
        self._dqn_epsilon_decay = max(
            1.0, float(getattr(r, "dqn_epsilon_decay", 10_000.0))
        )
        self._dqn_warmup = int(getattr(r, "dqn_warmup", 64))
        self._dqn_optimizer_steps = 0
        self._dqn_update_count = 0
        self._dqn_environment_steps = 0
        self._dqn_iteration_transitions = 0
        self._dqn_pending = None

        p = self._make_network(cfg)
        self.dqn_policy = p
        self.dqn_target = self._make_network(cfg)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        self.dqn_optimizer = torch.optim.Adam(
            self.dqn_policy.parameters(), lr=self._dqn_learning_rate
        )
        self.dqn_replay = _ReplayBuffer(self._dqn_replay_capacity)

    def _make_network(self, cfg):
        r = cfg.domains.minigrid.rmax_like
        return _SpatialQNetwork(
            feature_dim=int(getattr(r, "feature_dim", 128)),
            actions=6,
            object_dim=int(getattr(r, "object_embedding_dim", 8)),
            color_dim=int(getattr(r, "color_embedding_dim", 4)),
            state_dim=int(getattr(r, "state_embedding_dim", 4)),
            channels=int(getattr(r, "conv_channels", 64)),
        ).to(_DEVICE)

    @property
    def epsilon(self):
        progress = min(1.0, self._dqn_environment_steps / self._dqn_epsilon_decay)
        return self._dqn_epsilon_start + progress * (
            self._dqn_epsilon_end - self._dqn_epsilon_start
        )

    def set_training(self, enabled):
        self.training_enabled = bool(enabled)
        self._pending_transition = None
        self._dqn_pending = None
        self.dqn_policy.train(bool(enabled))
        self.dqn_target.eval()

    def begin_iteration(self):
        if self._iteration_active or self._dqn_pending is not None:
            raise RuntimeError("Cannot begin DQN iteration with pending transition")
        self._iteration_active = True
        self._dqn_iteration_transitions = 0

    def _greedy_action(self, state):
        with torch.no_grad():
            values = self.dqn_policy(state.unsqueeze(0)).squeeze(0)
            best = torch.nonzero(values == values.max(), as_tuple=False).flatten()
            if len(best) == 1:
                return int(best.item())
            choice = torch.randint(
                len(best), (), generator=self._rng, device=_DEVICE
            )
            return int(best[choice].item())

    def select_action(self, obs):
        self._mark(obs)
        state = self._policy_state(obs, self.current_inventory_token)
        explore = self.training_enabled and float(
            torch.rand((), generator=self._rng, device=_DEVICE).item()
        ) < self.epsilon
        if explore:
            action = int(
                torch.randint(6, (), generator=self._rng, device=_DEVICE).item()
            )
        else:
            action = self._greedy_action(state)
        self._dqn_pending = (state.detach(), action)
        self._dqn_environment_steps += 1
        return action

    def estimate(self, state):
        with torch.no_grad():
            return float(self.dqn_target(state.unsqueeze(0)).max().item())

    def record_transition(
        self,
        reward,
        done,
        *,
        obs_next=None,
        terminated=None,
        truncated=None,
        **_,
    ):
        if not self.training_enabled:
            self._dqn_pending = None
            return
        if not self._iteration_active:
            raise RuntimeError("Training transitions require begin_iteration")
        if self._dqn_pending is None:
            raise RuntimeError("MiniGrid transition has no sampled DQN action")
        state, action = self._dqn_pending
        terminal = bool(done) if terminated is None else bool(terminated)
        if obs_next is None:
            next_state = torch.zeros_like(state)
            terminal = True
        else:
            next_state = self._policy_state(obs_next, self.next_inventory_token)
        self.dqn_replay.append(state, action, float(reward), next_state, terminal)
        self._dqn_iteration_transitions += 1
        self._dqn_pending = None

    def mark_rollout_boundary(self):
        # A collector rollout ends the current map.  Do not let a Q target
        # bootstrap from the next map's unrelated initial state.
        if self.dqn_replay:
            state, action, reward, next_state, _ = self.dqn_replay.data[-1]
            self.dqn_replay.data[-1] = (
                state,
                action,
                reward,
                next_state,
                True,
            )
        self._dqn_pending = None

    def _update_once(self):
        if len(self.dqn_replay) < max(self._dqn_warmup, self._dqn_batch_size):
            return None
        states, actions, rewards, next_states, terminals = self.dqn_replay.sample(
            self._dqn_batch_size, self._rng
        )
        q_values = self.dqn_policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double-DQN target: select with the online network and evaluate
            # with the target network to reduce early over-estimation of an
            # arbitrary non-movement action under sparse rewards.
            next_actions = self.dqn_policy(next_states).argmax(1)
            next_values = self.dqn_target(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_values * (~terminals).float()
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.dqn_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.dqn_policy.parameters(), self.max_grad_norm)
        self.dqn_optimizer.step()
        self._dqn_optimizer_steps += 1
        if self._dqn_optimizer_steps % self._dqn_target_update_interval == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        return float(loss.detach().item())

    def end_iteration(self):
        if not self._iteration_active:
            raise RuntimeError("end_iteration requires begin_iteration")
        losses = [
            loss
            for _ in range(self._dqn_updates_per_iteration)
            if (loss := self._update_once()) is not None
        ]
        self._iteration_active = False
        self._dqn_update_count += 1
        return {
            "updated": bool(losses),
            "update_count": self._dqn_update_count,
            "transition_count": self._dqn_iteration_transitions,
            "rollout_size": self._dqn_iteration_transitions,
            "replay_size": len(self.dqn_replay),
            "optimizer_steps": len(losses),
            "dqn_loss": float(np.mean(losses)) if losses else float("nan"),
            "epsilon": float(self.epsilon),
        }

    @property
    def update_count(self):
        return self._dqn_update_count

    def _metadata(self):
        p = self.dqn_policy
        return {
            "architecture": _DQN_ARCH,
            "feature_dim": p.feature_dim,
            "object_embedding_dim": p.object_embedding.embedding_dim,
            "color_embedding_dim": p.color_embedding.embedding_dim,
            "state_embedding_dim": p.state_embedding.embedding_dim,
            "conv_channels": p.conv[0].out_channels,
            "action_dim": p.actor_head[-1].out_features,
            "object_vocab_size": p.object_embedding.num_embeddings,
            "color_vocab_size": p.color_embedding.num_embeddings,
            "state_vocab_size": p.state_embedding.num_embeddings,
            "object_vocab": sorted(OBJECT_TO_IDX.items()),
            "color_vocab": sorted(COLOR_TO_IDX.items()),
            "state_vocab": sorted(STATE_TO_IDX.items()),
            "inventory_schema": ["empty"]
            + [name for name, _ in sorted(COLOR_TO_IDX.items(), key=lambda item: item[1])],
            "action_schema": ["left", "right", "forward", "pickup", "toggle", "drop"],
            "backend": "dqn",
        }

    def save_checkpoint(self, path=None, completed_iteration=None):
        if self._iteration_active or self._dqn_pending is not None:
            raise RuntimeError("DQN checkpoints may only be saved at iteration boundary")
        target = self._checkpoint_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "metadata": self._metadata(),
                "policy": self.dqn_policy.state_dict(),
                "target": self.dqn_target.state_dict(),
                "optimizer": self.dqn_optimizer.state_dict(),
                "replay": self.dqn_replay.state_dict(),
                "update_count": self._dqn_update_count,
                "optimizer_steps": self._dqn_optimizer_steps,
                "environment_steps": self._dqn_environment_steps,
                "completed_iteration": self.completed_iteration
                if completed_iteration is None
                else int(completed_iteration),
                "rng_state": self._rng.get_state(),
            },
            temporary,
        )
        os.replace(temporary, target)

    def load_checkpoint(self, path=None):
        source = self._checkpoint_file(path)
        if not source.is_file():
            raise FileNotFoundError(f"Explorer checkpoint not found: {source}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata") != self._metadata():
            raise ValueError("Explorer checkpoint is incompatible with DQN spatial policy")
        self.dqn_policy.load_state_dict(checkpoint["policy"])
        self.dqn_target.load_state_dict(checkpoint["target"])
        self.dqn_optimizer.load_state_dict(checkpoint["optimizer"])
        self.dqn_replay.load_state_dict(checkpoint.get("replay", []))
        self._dqn_update_count = int(checkpoint.get("update_count", 0))
        self._dqn_optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        self._dqn_environment_steps = int(checkpoint.get("environment_steps", 0))
        self.completed_iteration = int(checkpoint.get("completed_iteration", 0))
        self._rng.set_state(checkpoint["rng_state"])
        return self.completed_iteration

    def copy_weights_from(self, source):
        if not isinstance(source, MiniGridDQNExplorer):
            raise TypeError("DQN evaluation clone requires a MiniGridDQNExplorer")
        self.dqn_policy.load_state_dict(source.dqn_policy.state_dict())
        self.dqn_target.load_state_dict(source.dqn_target.state_dict())
        self._dqn_update_count = source._dqn_update_count
        self._dqn_environment_steps = source._dqn_environment_steps
