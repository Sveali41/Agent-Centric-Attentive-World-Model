"""Plan2Explore-style disagreement exploration for the attentive WM.

This module ports the core P2E logic from ``Curriculum_world_model_learning``:
an independently bootstrapped dynamics ensemble supplies intrinsic reward and
a PPO actor learns to seek transitions on which the ensemble disagrees.

The Curriculum implementation used only the pooled map latent for Crafter.
AGEBT's WM also predicts inventory deltas, so the state used by both the
ensemble and explorer appends the normalized 16-slot inventory.  This keeps
crafting/resource uncertainty visible to P2E without changing the WM itself.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from domain.minigrid import minigrid_support
from modelBased.common.crafter_imagination import imagined_crafter_step_batch


# Small task-aware terms keep curiosity from preferring states that are merely
# hard to predict. Disagreement remains the dominant P2E signal.
_PROGRESSION_WEIGHTS = {
    4: 0.10,  # wood
    5: 1.00,  # stone
    6: 1.25,  # coal
    7: 1.50,  # iron
    8: 2.00,  # diamond
    10: 0.50,  # wood_pickaxe
    11: 1.25,  # stone_pickaxe
    12: 1.75,  # iron_pickaxe
    13: 0.25,  # wood_sword
    14: 0.50,  # stone_sword
    15: 0.75,  # iron_sword
}


class P2EEnsemble(nn.Module):
    """Bootstrapped latent-dynamics heads used for epistemic disagreement."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        num_models: int = 10,
        learning_rate: float = 1e-4,
        target_type: str = "delta",
        bootstrap_heads: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.target_type = str(target_type).strip().lower()
        if self.target_type not in {"latent", "delta"}:
            raise ValueError("P2E target_type must be 'latent' or 'delta'")
        if int(num_models) < 2:
            raise ValueError("P2E disagreement requires at least two ensemble heads")
        self.bootstrap_heads = bool(bootstrap_heads)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        input_dim = self.state_dim + self.action_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, self.state_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.state_dim * 2, self.state_dim),
                )
                for _ in range(int(num_models))
            ]
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(learning_rate)
        )

    def _state(self, value) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected P2E states shaped (B, {self.state_dim}), got {tuple(value.shape)}"
            )
        return value

    def _actions(self, action) -> torch.Tensor:
        action = torch.as_tensor(action, device=self.device)
        if action.ndim == 0:
            action = action.unsqueeze(0)
        action = action.long().reshape(-1)
        return F.one_hot(action, num_classes=self.action_dim).float()

    def predictions(self, state, action) -> torch.Tensor:
        """Return all head predictions with shape ``(K, B, state_dim)``."""
        state = self._state(state)
        action_one_hot = self._actions(action)
        if action_one_hot.shape[0] == 1 and state.shape[0] != 1:
            action_one_hot = action_one_hot.expand(state.shape[0], -1)
        if action_one_hot.shape[0] != state.shape[0]:
            raise ValueError("P2E state/action batch sizes do not match")
        inputs = torch.cat([state, action_one_hot], dim=-1)
        return torch.stack([head(inputs) for head in self.heads], dim=0)

    @torch.no_grad()
    def intrinsic_reward(self, state, action) -> torch.Tensor:
        """Standard-deviation disagreement averaged over state dimensions."""
        return self.predictions(state, action).std(dim=0, unbiased=False).mean(dim=-1)

    def train_step(
        self,
        state,
        action,
        next_state,
        *,
        epochs: int = 1,
        batch_size: int = 256,
    ) -> float:
        """Fit each head on an independent bootstrap of observed transitions."""
        state = self._state(state).detach()
        next_state = self._state(next_state).detach()
        action_one_hot = self._actions(action)
        if not (len(state) == len(next_state) == len(action_one_hot)):
            raise ValueError("P2E transition arrays must have the same length")
        inputs = torch.cat([state, action_one_hot], dim=-1)
        targets = next_state - state if self.target_type == "delta" else next_state
        count = int(inputs.shape[0])
        if count == 0:
            return 0.0

        total_loss = 0.0
        updates = 0
        batch_size = max(1, int(batch_size))
        self.train()
        for _ in range(max(1, int(epochs))):
            # Each head gets an independent bootstrap permutation for the
            # complete training set. Resampling only within a shared minibatch
            # makes the heads see nearly identical data and weakens disagreement.
            orders = [
                torch.randint(0, count, (count,), device=self.device)
                if self.bootstrap_heads
                else torch.randperm(count, device=self.device)
                for _ in self.heads
            ]
            for start in range(0, count, batch_size):
                loss = torch.zeros((), device=self.device)
                for head_index, head in enumerate(self.heads):
                    index = orders[head_index][start : start + batch_size]
                    head_inputs = inputs[index]
                    head_targets = targets[index]
                    loss = loss + F.mse_loss(head(head_inputs), head_targets)
                loss = loss / len(self.heads)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                updates += 1
        self.eval()
        return total_loss / max(updates, 1)


class CrafterP2EExplorer:
    """Connect a Crafter WM encoder and disagreement ensemble to PPO."""

    expects_raw_obs = True
    expects_observation_dict = True

    def __init__(self, ppo, world_model, ensemble: P2EEnsemble, cfg) -> None:
        self.ppo = ppo
        self.world_model = world_model
        self.ensemble = ensemble
        self.cfg = cfg
        self.inventory_dim = int(getattr(cfg.domains.crafter, "inventory_dim", 16))
        self.inventory_scale = float(getattr(cfg.p2e, "inventory_scale", 9.0))
        self.reward_scale = float(getattr(cfg.p2e, "intrinsic_reward_scale", 1.0))
        self.disagreement_weight = float(
            getattr(cfg.p2e, "intrinsic_disagreement_weight", 1.0)
        )
        self.progression_weight = float(
            getattr(cfg.p2e, "intrinsic_progression_weight", 0.25)
        )
        self.survival_weight = float(
            getattr(cfg.p2e, "intrinsic_survival_weight", 0.05)
        )
        self.death_penalty = float(
            getattr(cfg.p2e, "intrinsic_death_penalty", 0.5)
        )
        self.disagreement_clip = float(
            getattr(cfg.p2e, "intrinsic_disagreement_clip", 5.0)
        )
        self._disagreement_ema: float | None = None
        expected_dim = int(cfg.attention_model.embed_dim) + self.inventory_dim
        if self.ensemble.state_dim != expected_dim:
            raise ValueError(
                f"Crafter P2E ensemble state_dim={self.ensemble.state_dim}; "
                f"expected {expected_dim}"
            )
        self.current_step_context = None
        self._representation_version = 0
        self._intrinsic_rewards: list[float] = []
        self._actions: list[int] = []
        self._anchor_images = None
        self._anchor_inventories = None
        self._anchor_state = None

    def set_world_model(self, world_model) -> None:
        """Attach the latest online WM representation without resetting PPO."""
        self.world_model = world_model
        self._representation_version += 1

    def set_representation_anchor(self, images, inventories) -> None:
        """Keep a fixed raw anchor batch for measuring encoder drift."""
        self._anchor_images = np.asarray(images).copy()
        self._anchor_inventories = np.asarray(inventories).copy()
        self._anchor_state = self.encode_batch(
            self._anchor_images, self._anchor_inventories
        ).detach()

    def measure_representation_drift(self) -> dict[str, float]:
        if self._anchor_state is None:
            return {"representation_cosine": 1.0, "representation_l2": 0.0}
        current = self.encode_batch(
            self._anchor_images, self._anchor_inventories
        ).detach()
        previous = self._anchor_state.to(current.device)
        cosine = F.cosine_similarity(current, previous, dim=-1).mean()
        l2 = (current - previous).pow(2).mean().sqrt()
        self._anchor_state = current
        return {
            "representation_cosine": float(cosine.cpu()),
            "representation_l2": float(l2.cpu()),
        }

    @staticmethod
    def _parts(observation) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping):
            raise TypeError(
                "Crafter P2E requires the full observation dict with image and inventory"
            )
        return (
            np.asarray(observation["image"]),
            np.asarray(observation["inventory"], dtype=np.float32),
        )

    def _masked_images(self, images) -> torch.Tensor:
        device = next(self.world_model.parameters()).device
        images = torch.as_tensor(images, device=device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected Crafter image batch, got {tuple(images.shape)}")
        if images.shape[1] != 2 and images.shape[-1] == 2:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] != 2:
            raise ValueError(f"Expected two Crafter map channels, got {tuple(images.shape)}")
        positions = minigrid_support.get_agent_position(images, player_id=13)
        masked = minigrid_support.extract_masked_state(
            images, int(self.world_model.mask_size), positions
        )
        return masked.long()

    @torch.no_grad()
    def encode_batch(self, images, inventories) -> torch.Tensor:
        """Encode maps and inventory into the P2E state representation."""
        masked = self._masked_images(images)
        features = self.world_model.encode(masked).mean(dim=1)
        inventory = torch.as_tensor(
            inventories, dtype=torch.float32, device=features.device
        )
        if inventory.ndim == 1:
            inventory = inventory.unsqueeze(0)
        inventory = inventory[:, : self.inventory_dim] / max(self.inventory_scale, 1e-6)
        return torch.cat([features.float(), inventory], dim=-1)

    @torch.no_grad()
    def encode_observation(self, observation) -> torch.Tensor:
        image, inventory = self._parts(observation)
        return self.encode_batch(image, inventory)

    @staticmethod
    def _object_channel(images, device: torch.device) -> torch.Tensor:
        value = torch.as_tensor(images, device=device)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError(f"Expected Crafter image batch, got {tuple(value.shape)}")
        if value.shape[-1] == 2:
            return value[..., 0]
        if value.shape[1] == 2:
            return value[:, 0]
        raise ValueError(f"Expected two Crafter map channels, got {tuple(value.shape)}")

    @torch.no_grad()
    def _intrinsic_components(
        self,
        state: torch.Tensor,
        action,
        inventories=None,
        next_inventories=None,
        images=None,
        next_images=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the shared real/imagined task-aware P2E reward."""
        disagreement = self.ensemble.intrinsic_reward(state, action).reshape(-1)
        batch_mean = float(disagreement.mean().detach().cpu())
        if self._disagreement_ema is None:
            self._disagreement_ema = max(batch_mean, 1e-6)
        else:
            self._disagreement_ema = (
                0.99 * self._disagreement_ema + 0.01 * max(batch_mean, 0.0)
            )
        normalized = disagreement / max(self._disagreement_ema, 1e-6)
        normalized = normalized.clamp(0.0, self.disagreement_clip)
        reward = self.disagreement_weight * normalized

        progression = torch.zeros_like(reward)
        survival_delta = torch.zeros_like(reward)
        death = torch.zeros_like(reward)
        if inventories is not None and next_inventories is not None:
            current = torch.as_tensor(inventories, device=reward.device, dtype=torch.float32)
            following = torch.as_tensor(
                next_inventories, device=reward.device, dtype=torch.float32
            )
            if current.ndim == 1:
                current = current.unsqueeze(0)
            if following.ndim == 1:
                following = following.unsqueeze(0)
            positive_delta = (following[:, : self.inventory_dim] - current[:, : self.inventory_dim]).clamp_min(0.0).clamp_max(1.0)
            weights = torch.zeros(self.inventory_dim, device=reward.device)
            for index, weight in _PROGRESSION_WEIGHTS.items():
                if index < self.inventory_dim:
                    weights[index] = weight
            progression = (positive_delta * weights).sum(dim=-1)
            survival_delta = ((following[:, 0] - current[:, 0]) / 9.0).clamp(-1.0, 1.0)
            death = (following[:, 0] <= 0.0).float()

            if images is not None and next_images is not None:
                objects = self._object_channel(images, reward.device)
                next_objects = self._object_channel(next_images, reward.device)
                table = ((next_objects == 11) & (objects != 11)).any(dim=(1, 2))
                furnace = ((next_objects == 12) & (objects != 12)).any(dim=(1, 2))
                progression = progression + table.float() * 0.20 + furnace.float() * 0.30

        reward = reward + self.progression_weight * progression
        reward = reward + self.survival_weight * survival_delta
        reward = reward - self.death_penalty * death
        # A terminal death must never become attractive just because the WM is
        # uncertain about that transition.
        reward = torch.where(
            death > 0.0,
            torch.full_like(reward, -self.death_penalty),
            reward,
        )
        return reward, {
            "disagreement": normalized,
            "progression": progression,
            "survival_delta": survival_delta,
            "death": death,
        }

    def begin_rollout(self, transition_budget) -> None:
        self.current_step_context = None

    def select_action(self, observation) -> int:
        state = self.encode_observation(observation)
        policy_device = next(self.ppo.policy_old.parameters()).device
        state = state.to(policy_device)
        action_idx, _, _, _, _ = self.ppo.select_action(state)
        return int(action_idx)

    def compute_intrinsic_reward(self, observation, action, next_observation) -> float:
        image, inventory = self._parts(observation)
        next_image, next_inventory = self._parts(next_observation)
        state = self.encode_batch(image, inventory)
        reward, _ = self._intrinsic_components(
            state,
            int(action),
            inventory,
            next_inventory,
            image,
            next_image,
        )
        return float(reward[0].detach().cpu()) * self.reward_scale

    def record_transition(
        self,
        reward,
        is_terminal,
        env_reward=0.0,
        obs=None,
        obs_next=None,
        action=None,
        is_success=False,
    ) -> None:
        del env_reward, obs, obs_next, is_success
        self._intrinsic_rewards.append(float(reward))
        if action is not None:
            self._actions.append(int(action))
        self.current_step_context = None

    def mark_rollout_boundary(self) -> None:
        self.current_step_context = None

    def update_ppo_if_ready(self, *, force: bool = False) -> bool:
        del force
        return False

    def adapt_actor_imagined(self, start_images, start_inventories) -> dict:
        """Train PPO on short imagined rollouts from real replay states."""
        images = np.asarray(start_images)
        inventories = np.asarray(start_inventories, dtype=np.float32)
        if len(images) == 0:
            return {"imagined_transitions": 0, "imagined_updates": 0}
        device = next(self.world_model.parameters()).device
        policy_device = next(self.ppo.policy_old.parameters()).device
        target_mode = "absolute"
        for field in getattr(self.cfg.domains.crafter, "observation_schema", []):
            if str(field.get("name", "")) == "inventory":
                target_mode = str(field.get("target_mode", "absolute"))
        horizon = max(1, int(getattr(self.cfg.p2e, "imagined_horizon", 15)))
        batch_size = max(1, int(getattr(self.cfg.p2e, "imagined_batch_size", 64)))
        batches = max(1, int(getattr(self.cfg.p2e, "imagined_batches_per_cycle", 4)))
        reward_values = []
        entropy_values = []
        update_metrics = []
        self.world_model.eval()
        self.ensemble.eval()
        for _ in range(batches):
            indices = np.random.randint(0, len(images), size=batch_size)
            states = torch.as_tensor(images[indices], device=device)
            if states.ndim != 4:
                raise ValueError(f"Expected imagined maps shaped (B,H,W,2), got {tuple(states.shape)}")
            if states.shape[-1] == 2:
                states = states.permute(0, 3, 1, 2)
            states = states.float()
            inventory = torch.as_tensor(inventories[indices], device=device, dtype=torch.float32)
            for step in range(horizon):
                latent = self.encode_batch(states, inventory)
                policy_latent = latent.to(policy_device)
                actions, state_buffer, action_buffer, logprobs, values = self.ppo.select_action_batch(policy_latent)
                with torch.no_grad():
                    probs = self.ppo.policy_old.actor(policy_latent)
                    entropy_values.append(float(torch.distributions.Categorical(probs).entropy().mean().cpu()))
                    next_states, next_inventory = imagined_crafter_step_batch(
                        self.world_model,
                        states,
                        actions.to(device),
                        inventory,
                        int(self.world_model.mask_size),
                        target_mode,
                    )
                    intrinsic, _components = self._intrinsic_components(
                        latent,
                        actions.to(self.ensemble.device),
                        inventory,
                        next_inventory,
                        states,
                        next_states,
                    )
                    intrinsic = intrinsic.to(policy_device) * self.reward_scale
                    terminals = next_inventory[:, 0] <= 0.0
                self.ppo.save_buffer_batch(
                    state_buffer, action_buffer, logprobs, values, intrinsic, terminals
                )
                reward_values.extend(intrinsic.detach().cpu().tolist())
                states, inventory = next_states, next_inventory
            with torch.no_grad():
                next_latent = self.encode_batch(states, inventory).to(policy_device)
                bootstrap = self.ppo.estimate_old_values_batch(next_latent)
                bootstrap[terminals.detach().cpu()] = 0.0
            update_metrics.append(self.ppo.update(bootstrap_value=bootstrap))
        return {
            "imagined_transitions": int(len(reward_values)),
            "imagined_updates": int(len(update_metrics)),
            "imagined_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
            "imagined_reward_std": float(np.std(reward_values)) if reward_values else 0.0,
            "imagined_reward_p10": float(np.percentile(reward_values, 10)) if reward_values else 0.0,
            "imagined_reward_p50": float(np.percentile(reward_values, 50)) if reward_values else 0.0,
            "imagined_reward_p90": float(np.percentile(reward_values, 90)) if reward_values else 0.0,
            "imagined_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "imagined_parameter_delta": float(sum(m.get("parameter_delta", 0.0) for m in update_metrics)),
            "imagined_grad_norm": float(np.mean([m.get("grad_norm", 0.0) for m in update_metrics])) if update_metrics else 0.0,
        }

    def metrics(self, *, reset: bool = False) -> dict[str, float | int]:
        rewards = np.asarray(self._intrinsic_rewards, dtype=np.float32)
        actions = np.asarray(self._actions, dtype=np.int64)
        if actions.size:
            probabilities = np.bincount(actions, minlength=17).astype(np.float64)
            probabilities /= probabilities.sum()
            probabilities = probabilities[probabilities > 0]
            action_entropy = float(-(probabilities * np.log(probabilities)).sum())
        else:
            action_entropy = 0.0
        result = {
            "intrinsic_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "intrinsic_reward_std": float(rewards.std()) if rewards.size else 0.0,
            "action_coverage_ratio": float(len(np.unique(actions)) / 17.0)
            if actions.size
            else 0.0,
            "real_action_entropy": action_entropy,
            "transitions": int(rewards.size),
            "representation_version": int(self._representation_version),
        }
        if reset:
            self._intrinsic_rewards.clear()
            self._actions.clear()
        return result
