"""Official DreamerV2 Plan2Explore components for the Crafter adapter."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from domain.minigrid import minigrid_support
from modelBased.world_model.crafter_dynamics import imagined_crafter_step_batch


def _mlp(in_dim: int, out_dim: int, *, layers: int = 4, units: int = 400) -> nn.Sequential:
    blocks: list[nn.Module] = []
    current = int(in_dim)
    for _ in range(int(layers)):
        blocks.extend((nn.Linear(current, units), nn.ELU()))
        current = units
    blocks.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*blocks)


class P2EEnsemble(nn.Module):
    """DreamerV2 disagreement ensemble: absolute next-state targets."""

    def __init__(self, state_dim: int, action_dim: int, *, num_models: int = 10,
                 learning_rate: float = 3e-4, device: torch.device | str | None = None,
                 **_ignored: Any) -> None:
        super().__init__()
        if int(num_models) < 2:
            raise ValueError("P2E requires at least two ensemble heads")
        self.state_dim, self.action_dim, self.num_models = int(state_dim), int(action_dim), int(num_models)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.heads = nn.ModuleList([_mlp(self.state_dim + self.action_dim, self.state_dim) for _ in range(self.num_models)])
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate), eps=1e-5, weight_decay=1e-6)

    def _state(self, value: Any) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value[None]
        if value.ndim != 2 or value.shape[-1] != self.state_dim:
            raise ValueError(f"Expected (B,{self.state_dim}) state, got {tuple(value.shape)}")
        return value

    def _actions(self, action: Any, batch: int | None = None) -> torch.Tensor:
        action = torch.as_tensor(action, dtype=torch.long, device=self.device).reshape(-1)
        if batch is not None and action.numel() == 1 and batch != 1:
            action = action.expand(batch)
        if batch is not None and action.numel() != batch:
            raise ValueError("P2E state/action batch sizes do not match")
        return F.one_hot(action, num_classes=self.action_dim).float()

    def predictions(self, state: Any, action: Any) -> torch.Tensor:
        state = self._state(state)
        inputs = torch.cat((state, self._actions(action, state.shape[0])), dim=-1)
        return torch.stack([head(inputs) for head in self.heads])

    @torch.no_grad()
    def intrinsic_reward(self, state: Any, action: Any) -> torch.Tensor:
        return self.predictions(state, action).std(dim=0, unbiased=False).mean(dim=-1)

    def train_batch(self, state: Any, action: Any, next_state: Any) -> float:
        state, next_state = self._state(state).detach(), self._state(next_state).detach()
        inputs = torch.cat((state, self._actions(action, state.shape[0])), dim=-1)
        target = next_state.detach()
        self.train(); self.optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([F.mse_loss(head(inputs), target) for head in self.heads]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0); self.optimizer.step(); self.eval()
        return float(loss.detach().cpu())

    def train_step(self, state, action, next_state, **_kwargs) -> float:
        return self.train_batch(state, action, next_state)


def td_lambda_targets(rewards: torch.Tensor, values_next: torch.Tensor, continuation: torch.Tensor,
                      gamma: float, lam: float) -> torch.Tensor:
    """Backward TD(lambda), with continuation already containing gamma."""
    rewards, values_next, continuation = [x.float() for x in (rewards, values_next, continuation)]
    targets = torch.zeros_like(rewards)
    running = values_next[..., -1]
    for index in range(rewards.shape[-1] - 1, -1, -1):
        bootstrap = values_next[..., index]
        running = rewards[..., index] + continuation[..., index] * ((1.0 - lam) * bootstrap + lam * running)
        targets[..., index] = running
    return targets


class DreamerP2EActorCritic(nn.Module):
    """Discrete Dreamer actor-critic for imagined exploration trajectories."""

    def __init__(self, state_dim: int, action_dim: int, *, actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4, entropy_coef: float = 3e-3, gamma: float = .999,
                 discount_lambda: float = .95, horizon: int = 15, slow_target_update: int = 100,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.entropy_coef, self.gamma, self.discount_lambda = float(entropy_coef), float(gamma), float(discount_lambda)
        self.horizon, self.slow_target_update = int(horizon), int(slow_target_update)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor, self.critic, self.target_critic = _mlp(self.state_dim, self.action_dim), _mlp(self.state_dim, 1), _mlp(self.state_dim, 1)
        self.to(self.device); self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr), eps=1e-5)
        self.update_count = 0

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(state))

    @torch.no_grad()
    def act(self, state: Any) -> int:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1: state = state[None]
        return int(self.distribution(state).sample()[0].cpu())

    def imagine_update(self, start_states: torch.Tensor, start_images: torch.Tensor, start_inventories: torch.Tensor,
                       wm_adapter, ensemble: P2EEnsemble, *, target_mode: str = "categorical_gate",
                       start_terminal: torch.Tensor | None = None) -> dict[str, float]:
        del start_terminal
        device = self.device
        images = start_images.to(next(wm_adapter.world_model.parameters()).device)
        inventory = start_inventories.to(images.device).float()
        latent = start_states.to(device).detach()
        logps, entropies, rewards, conts, values, next_values = [], [], [], [], [], []
        for _ in range(self.horizon):
            # Keep policy log-probabilities attached for REINFORCE.  The WM,
            # ensemble and imagined states are explicitly detached below.
            dist = self.distribution(latent); action = dist.sample()
            logps.append(dist.log_prob(action)); entropies.append(dist.entropy()); values.append(self.critic(latent).squeeze(-1))
            with torch.no_grad():
                next_images, next_inventory = wm_adapter.imagine_step(images, action, inventory, target_mode)
                next_latent = wm_adapter.encode_states(next_images, next_inventory).to(device).detach()
                rewards.append(ensemble.intrinsic_reward(latent.detach(), action).to(device))
                next_values.append(self.target_critic(next_latent).squeeze(-1))
                conts.append((next_inventory[:, 0] > 0).float().to(device) * self.gamma)
            latent, images, inventory = next_latent, next_images, next_inventory
        rew, cont = torch.stack(rewards, 1), torch.stack(conts, 1)
        vals, nxt = torch.stack(values, 1), torch.stack(next_values, 1)
        targets = td_lambda_targets(rew, nxt, cont, self.gamma, self.discount_lambda)
        weights = torch.ones_like(cont)
        if self.horizon > 1: weights[:, 1:] = torch.cumprod(cont[:, :-1], dim=1)
        advantage = (targets - vals).detach()
        actor_loss = -(weights * (torch.stack(logps, 1) * advantage + self.entropy_coef * torch.stack(entropies, 1))).mean()
        critic_loss = (weights * (vals - targets.detach()).pow(2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0); self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0); self.critic_optimizer.step()
        self.update_count += 1
        if self.update_count % self.slow_target_update == 0: self.target_critic.load_state_dict(self.critic.state_dict())
        reward_flat = rew.detach().reshape(-1).cpu()
        return {"actor_loss": float(actor_loss.detach().cpu()), "critic_loss": float(critic_loss.detach().cpu()),
                "imagined_reward_mean": float(reward_flat.mean()),
                "imagined_reward_p50": float(torch.quantile(reward_flat, .50)),
                "imagined_reward_p90": float(torch.quantile(reward_flat, .90)),
                "imagined_reward_max": float(reward_flat.max()),
                "imagined_entropy": float(torch.stack(entropies).mean().cpu()), "imagined_transitions": float(rew.numel())}


class AttentionWMP2EAdapter:
    """Attention WM encoding, native supervised update, and one-step rollout."""

    def __init__(self, world_model, *, inventory_dim: int = 16, inventory_scale: float = 9.0, amp: bool = False):
        self.world_model, self.inventory_dim, self.inventory_scale = world_model, int(inventory_dim), float(inventory_scale)
        self.amp = bool(amp and next(world_model.parameters()).is_cuda)
        configured = world_model.configure_optimizers(); self.optimizer = configured["optimizer"] if isinstance(configured, dict) else configured

    def _masked(self, images: Any) -> torch.Tensor:
        device = next(self.world_model.parameters()).device; x = torch.as_tensor(images, device=device)
        if x.ndim == 3: x = x[None]
        if x.shape[-1] == 2 and x.shape[1] != 2: x = x.permute(0, 3, 1, 2)
        pos = minigrid_support.get_agent_position(x, player_id=13)
        return minigrid_support.extract_masked_state(x, int(self.world_model.mask_size), pos).long()

    # no_grad (rather than inference_mode) is intentional: the resulting
    # latent is consumed by the actor/critic autograd graph. Inference-mode
    # tensors cannot safely be used as autograd inputs.
    @torch.no_grad()
    def encode_states(self, images: Any, inventories: Any) -> torch.Tensor:
        # The collector calls train_batch() explicitly before each update, so
        # keeping eval active avoids thousands of train/eval
        # toggles during one imagined rollout.
        self.world_model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            latent = self.world_model.encode(self._masked(images)).mean(1).float()
        inv = torch.as_tensor(inventories, dtype=torch.float32, device=latent.device)
        if inv.ndim == 1: inv = inv[None]
        return torch.cat((latent, inv[:, :self.inventory_dim] / self.inventory_scale), dim=-1)

    @torch.inference_mode()
    def imagine_step(self, images, action, inventory, target_mode: str):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            return imagined_crafter_step_batch(
                self.world_model, images, action, inventory,
                int(self.world_model.mask_size), target_mode,
                inventory_value_mode=getattr(
                    self.world_model, "crafter_inventory_value_mode",
                    "categorical_absolute",
                ),
            )

    def train_batch(self, batch: dict[str, Any]) -> float:
        device = next(self.world_model.parameters()).device; self.world_model.train()
        prepared = {key: (torch.as_tensor(value, device=device) if value is not None else None) for key, value in batch.items()}
        for key in ("obs", "obs_next"):
            if prepared.get(key) is not None and prepared[key].ndim == 4 and prepared[key].shape[-1] == 2:
                prepared[key] = prepared[key].permute(0, 3, 1, 2).contiguous()
        obs, act, obs_next, info, _mask, _pos, inv, inv_next = self.world_model.preprocess_batch(prepared, True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred, _attn, aux = self.world_model(obs, act, info, inv=inv)
            loss, _fields = self.world_model.observation_loss(pred, obs_next, obs, aux, inv, inv_next)
        self.optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0); self.optimizer.step()
        return float(loss.detach().cpu())


class CrafterP2EExplorer:
    """Policy facade for generic collectors; reward remains pure disagreement."""
    expects_raw_obs = True; expects_observation_dict = True
    def __init__(self, actor_critic, adapter: AttentionWMP2EAdapter, ensemble: P2EEnsemble): self.actor_critic, self.adapter, self.ensemble = actor_critic, adapter, ensemble
    def select_action(self, observation) -> int: return self.actor_critic.act(self.adapter.encode_states(observation["image"], observation["inventory"]))
    def compute_intrinsic_reward(self, observation, action, next_observation) -> float:
        del next_observation
        return float(self.ensemble.intrinsic_reward(self.adapter.encode_states(observation["image"], observation["inventory"]), action)[0].cpu())


class EpisodeReplay:
    """Uniform episodic replay preserving sequence and terminal boundaries."""
    def __init__(self, capacity: int = 2_000_000): self.capacity, self.episodes, self.current, self.size = int(capacity), deque(), [], 0
    def add(self, obs, obs_next, action, reward, done, info, inv, inv_next):
        self.current.append({"obs": np.asarray(obs), "obs_next": np.asarray(obs_next), "act": int(action), "rew": float(reward), "done": bool(done), "is_first": len(self.current) == 0, "info": info, "inv": np.asarray(inv, dtype=np.float32), "inv_next": np.asarray(inv_next, dtype=np.float32)}); self.size += 1
        if done: self.episodes.append(self.current); self.current = []
        while self.size > self.capacity and self.episodes: self.size -= len(self.episodes.popleft())
    def __len__(self): return self.size
    @property
    def ready(self): return bool(self.episodes)
    def sample(self, batch_size: int, sequence_length: int, rng=None) -> dict[str, np.ndarray]:
        rng = rng or np.random.default_rng(); eps = list(self.episodes)
        if not eps: raise ValueError("Cannot sample empty replay")
        rows=[]
        for _ in range(int(batch_size)):
            ep=eps[int(rng.integers(len(eps)))]; start=int(rng.integers(max(1, len(ep)-sequence_length+1))) if len(ep)>=sequence_length else 0
            seq=ep[start:start+sequence_length]
            if len(seq)<sequence_length: seq=[seq[i % len(seq)] for i in range(sequence_length)]
            rows.append(seq)
        result={k: np.asarray([[item[k] for item in row] for row in rows]) for k in ("obs","obs_next","act","rew","done","inv","inv_next","is_first")}; result["is_terminal"] = result["done"]; return result
    def export_dict(self):
        rows=[item for ep in self.episodes for item in ep]+list(self.current); return {k: np.asarray([item[k] for item in rows]) for k in ("obs","obs_next","act","rew","done","inv","inv_next","info")}
    def save(self, path: str | Path): Path(path).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path, **self.export_dict())
    @classmethod
    def load(cls, path: str | Path, capacity=2_000_000):
        replay=cls(capacity); data=np.load(path,allow_pickle=True)
        for i in range(len(data["act"])): replay.add(*(data[k][i] for k in ("obs","obs_next","act","rew","done","info","inv","inv_next")))
        return replay
