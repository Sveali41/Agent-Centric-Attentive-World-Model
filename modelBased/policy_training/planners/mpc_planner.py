"""Online MiniGrid MPC: search only in the learned world model.

Each control cycle evaluates action sequences in the WM, executes a bounded
prefix in the real environment, then replans from the new real observation.
The prefix is cut short when the real state disagrees with the WM prediction.
Imagined transitions are logged separately from ``env.step``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper
from omegaconf import DictConfig

from domain.minigrid.action_codec import (
    COMPACT_ACTION_NAMES,
    MODEL_ACTION_COUNT,
    carrying_token_from_env,
    compact_to_native,
)
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from domain.minigrid import minigrid_support as minigrid_utils
from modelBased.common import utils
from modelBased.common.artifacts import world_model_checkpoint_path
from modelBased.policy_training.common.minigrid_dense_reward import (
    build_door_topology,
    build_goal_distance_map,
    build_goal_region_mask,
    main_dense_rewards,
    reward_settings,
)
from modelBased.policy_training.common.minigrid_wm_rollout import rollout_minigrid_wm
from modelBased.world_model import AttentionWM_support
from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import ACHIEVEMENT_NAMES, native_reward_batch
from domain.crafter.crafter_support import CRAFTER_ACTION_NAMES, load_crafter_planning_model
from modelBased.world_model.crafter_dynamics import SURVIVAL_SLOTS, crafter_player_counts, imagined_crafter_step_batch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_goal(state: np.ndarray) -> tuple[int, int]:
    matches = np.argwhere(np.asarray(state)[0] == 8)
    if not len(matches):
        raise ValueError("MiniGrid MPC requires a goal tile (object ID 8)")
    return tuple(map(int, matches[0]))


def _realized_goal_guide_bonus(
    start_distance: float,
    end_distance: float,
    weight: float,
    gamma: float,
    executed_steps: int,
    failed: bool,
) -> float:
    """Measure the planner's endpoint distance bonus on an executed action block."""
    if failed or executed_steps < 1 or not np.isfinite(start_distance) or not np.isfinite(end_distance):
        return 0.0
    progress = np.clip(
        (start_distance - end_distance) / max(1.0, start_distance), -1.0, 1.0
    )
    return float(weight * progress * (gamma ** (executed_steps - 1)))


class DenseRewardContext:
    """Episode reward history shared by real steps and imagined candidates."""

    _UPDATED_FIELDS = {
        "best_goal_distances": "new_best_goal_distance",
        "progress_milestone_seen": "progress_milestone_seen",
        "pending_door_ids": "pending_door_ids",
        "pending_door_origins": "pending_door_origins",
        "rewarded_door_crossings": "rewarded_door_crossings",
        "rewarded_goal_region_key_positions": "rewarded_goal_region_key_positions",
        "rewarded_goal_region_door_positions": "rewarded_goal_region_door_positions",
    }

    def __init__(self, initial_state: torch.Tensor, goal_yx: tuple[int, int], settings: dict):
        state = torch.as_tensor(initial_state, device=DEVICE)
        self.settings = settings
        self.goal_distance_map = build_goal_distance_map(state, goal_yx)
        self.goal_region_mask = build_goal_region_mask(state, goal_yx)
        self.door_topology = build_door_topology(state, goal_yx)
        position = utils.get_agent_position_torch(state)
        distance = self.goal_distance_map[position[0], position[1]].reshape(1)
        height, width = state.shape[-2:]
        self.history = {
            "goal_region_seen": torch.zeros(1, device=DEVICE, dtype=torch.bool),
            "best_goal_distances": distance.clone(),
            "initial_goal_distances": distance.clone(),
            "progress_milestone_seen": torch.zeros(1, device=DEVICE, dtype=torch.bool),
            "pending_door_ids": torch.full((1,), -1, device=DEVICE, dtype=torch.long),
            "pending_door_origins": torch.full((1,), -1, device=DEVICE, dtype=torch.long),
            "rewarded_door_crossings": torch.zeros(
                (1, self.door_topology.num_doors), device=DEVICE, dtype=torch.bool
            ),
            "rewarded_goal_region_key_positions": torch.zeros(
                (1, height, width), device=DEVICE, dtype=torch.bool
            ),
            "rewarded_goal_region_door_positions": torch.zeros(
                (1, height, width), device=DEVICE, dtype=torch.bool
            ),
        }

    def for_population(self, population: int) -> "DenseRewardContext":
        context = object.__new__(DenseRewardContext)
        context.settings = self.settings
        context.goal_distance_map = self.goal_distance_map
        context.goal_region_mask = self.goal_region_mask
        context.door_topology = self.door_topology
        context.history = {
            name: value.expand(population, *value.shape[1:]).clone()
            for name, value in self.history.items()
        }
        return context

    def step(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        actions: torch.Tensor,
        previous_inventory: torch.Tensor,
        current_inventory: torch.Tensor,
        success: torch.Tensor,
        lava_terminated: torch.Tensor,
        transition_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rewards, goal_region_seen, events = main_dense_rewards(
            previous,
            current,
            actions,
            self.goal_distance_map,
            self.goal_region_mask,
            self.history["goal_region_seen"],
            success,
            lava_terminated,
            best_goal_distances=self.history["best_goal_distances"],
            initial_goal_distances=self.history["initial_goal_distances"],
            progress_milestone_seen=self.history["progress_milestone_seen"],
            previous_carrying_tokens=previous_inventory,
            current_carrying_tokens=current_inventory,
            door_topology=self.door_topology,
            pending_door_ids=self.history["pending_door_ids"],
            pending_door_origins=self.history["pending_door_origins"],
            rewarded_door_crossings=self.history["rewarded_door_crossings"],
            rewarded_goal_region_key_positions=self.history["rewarded_goal_region_key_positions"],
            rewarded_goal_region_door_positions=self.history["rewarded_goal_region_door_positions"],
            transition_valid=transition_valid,
            **self.settings,
        )
        self.history["goal_region_seen"] = goal_region_seen
        for name, event_name in self._UPDATED_FIELDS.items():
            self.history[name] = events[event_name]
        return rewards


@dataclass
class MPCPlan:
    actions: list[int]
    score: float
    predicted_positions: list[tuple[int, int]]
    predicted_inventories: list[int] | list[list[float]]
    diagnostics: dict[str, float | int | str]


class WorldModelMPC:
    """Random-shooting MPC with elite categorical resampling for MiniGrid."""

    def __init__(self, model, attention_mask_size: int, config, ppo_config):
        self.model = model
        self.attention_mask_size = int(attention_mask_size)
        self.horizon = int(config.horizon)
        self.population = int(config.population)
        self.elite_count = int(config.elite_count)
        self.iterations = int(config.iterations)
        self.gamma = float(config.gamma)
        self.reward_mode = str(
            getattr(config, "minigrid_reward_mode", "native_goal")
        ).lower()
        if self.reward_mode not in {"legacy_dense", "native_goal"}:
            raise ValueError(
                "PPO.mpc.minigrid_reward_mode must be 'legacy_dense' or 'native_goal'"
            )
        self.goal_guide_weight = float(
            getattr(config, "minigrid_goal_guide_weight", 0.05)
        )
        if not np.isfinite(self.goal_guide_weight) or self.goal_guide_weight < 0:
            raise ValueError("PPO.mpc.minigrid_goal_guide_weight must be finite and nonnegative")
        self.reward_settings = reward_settings(ppo_config)
        self.fallback_action = int(config.fallback_action)
        self.capture_action_hashes = bool(getattr(config, "capture_action_hashes", False))
        self.return_attention_weights = not bool(
            getattr(config, "minigrid_skip_attention_weights", False)
        )
        self.capture_profile_timing = bool(getattr(config, "profile_timing", False))
        self.action_hashes: list[tuple[str, int, str]] = []
        self._forward_decode_events = []
        self.execute_steps = int(getattr(config, "minigrid_execute_steps", self.horizon))
        if self.horizon < 1 or self.population < 1 or self.iterations < 1:
            raise ValueError("MPC horizon, population, and iterations must be positive")
        if not 1 <= self.execute_steps <= self.horizon:
            raise ValueError(
                "MPC execute_steps must be in [1, horizon] "
                f"(got {self.execute_steps}, horizon={self.horizon})"
            )
        if not 1 <= self.elite_count <= self.population:
            raise ValueError("MPC elite_count must be in [1, population]")
        if not 0 <= self.fallback_action < MODEL_ACTION_COUNT:
            raise ValueError("MPC fallback_action must be a compact MiniGrid action")

    def _sample(self, probabilities: torch.Tensor | None) -> torch.Tensor:
        if probabilities is None:
            return torch.randint(
                MODEL_ACTION_COUNT,
                (self.population, self.horizon),
                device=DEVICE,
            )
        return torch.multinomial(
            probabilities.unsqueeze(0).expand(self.population, -1, -1).reshape(
                -1, MODEL_ACTION_COUNT
            ),
            1,
        ).reshape(self.population, self.horizon)

    def _evaluate(
        self,
        start_state: torch.Tensor,
        start_inventory: int,
        sequences: torch.Tensor,
        goal_yx: tuple[int, int],
        reward_context: DenseRewardContext,
        episode_step: int,
        max_episode_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
    ]:
        if episode_step < 0 or max_episode_steps < 1:
            raise ValueError("episode_step must be nonnegative and max_episode_steps positive")
        states = start_state.unsqueeze(0).expand(self.population, -1, -1, -1).clone()
        inventories = torch.full(
            (self.population,), int(start_inventory), device=DEVICE, dtype=torch.long
        )
        scores = torch.zeros(self.population, device=DEVICE)
        native_returns = torch.zeros_like(scores)
        failure_penalties = torch.zeros_like(scores)
        finished = torch.zeros(self.population, device=DEVICE, dtype=torch.bool)
        use_legacy_dense = self.reward_mode == "legacy_dense"
        imagined_reward = (
            reward_context.for_population(self.population) if use_legacy_dense else None
        )
        predicted_positions = torch.empty(
            (self.population, self.horizon, 2), device=DEVICE, dtype=torch.long
        )
        predicted_inventories = torch.empty(
            (self.population, self.horizon), device=DEVICE, dtype=torch.long
        )
        uncertainty_sum = torch.zeros((), device=DEVICE, dtype=torch.float64)
        invalid_sum = torch.zeros((), device=DEVICE, dtype=torch.long)
        end_positions = utils.get_agent_position_torch(states)
        current_positions = end_positions
        row_indices = torch.arange(self.population, device=DEVICE)
        deltas = torch.as_tensor(((0, 1), (1, 0), (0, -1), (-1, 0)), device=DEVICE)
        end_steps = torch.full(
            (self.population,), self.horizon - 1, device=DEVICE, dtype=torch.long
        )
        failure_terminal = torch.zeros(self.population, device=DEVICE, dtype=torch.bool)
        for step in range(self.horizon):
            actions = sequences[:, step]
            previous_states = states
            previous_inventories = inventories
            before_positions = current_positions
            directions = previous_states[
                row_indices,
                2,
                before_positions[:, 0],
                before_positions[:, 1],
            ].long()
            front = before_positions + deltas[directions.clamp(0, 3)]
            height, width = previous_states.shape[-2:]
            valid_front = (
                directions.ge(0) & directions.lt(4)
                & front[:, 0].ge(0) & front[:, 0].lt(height)
                & front[:, 1].ge(0) & front[:, 1].lt(width)
            )
            front_objects = previous_states[
                row_indices,
                0,
                front[:, 0].clamp(0, height - 1),
                front[:, 1].clamp(0, width - 1),
            ]
            lava_terminated = (
                ~finished & actions.eq(2) & valid_front & front_objects.eq(9)
            )
            if self.capture_profile_timing and DEVICE.type == "cuda":
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
            else:
                forward_started = time.perf_counter() if self.capture_profile_timing else 0.0
            states, inventories, transition = rollout_minigrid_wm(
                self.model, states, actions, inventories, self.attention_mask_size,
                agent_positions=before_positions,
                return_attention_weights=self.return_attention_weights,
            )
            if self.capture_profile_timing:
                if DEVICE.type == "cuda":
                    forward_end.record()
                    self._forward_decode_events.append((forward_start, forward_end))
                else:
                    self._forward_decode_seconds += time.perf_counter() - forward_started
            after_positions = transition["agent_positions_after"]
            current_positions = after_positions
            if use_legacy_dense:
                end_positions = after_positions
            predicted_positions[:, step] = after_positions
            predicted_inventories[:, step] = inventories
            at_goal = (after_positions[:, 0] == goal_yx[0]) & (
                after_positions[:, 1] == goal_yx[1]
            )
            new_goal = at_goal & ~finished & ~lava_terminated
            at_time_limit = episode_step + step + 1 >= max_episode_steps
            new_truncation = (
                (~finished & ~new_goal & ~lava_terminated) & at_time_limit
                if not use_legacy_dense
                else torch.zeros_like(finished)
            )
            unchanged = states.eq(previous_states).flatten(1).all(dim=1) & inventories.eq(previous_inventories)
            invalid = unchanged & (actions >= 2)
            discount = self.gamma ** step
            if use_legacy_dense:
                dense_reward = imagined_reward.step(
                    previous_states,
                    states,
                    actions,
                    previous_inventories,
                    inventories,
                    new_goal,
                    lava_terminated,
                    transition_valid=~finished,
                )
                scores += discount * torch.where(finished, 0.0, dense_reward)
                finished |= new_goal | lava_terminated
            else:
                success_step = episode_step + step + 1
                native_success_reward = 1.0 - 0.9 * (
                    float(success_step) / float(max_episode_steps)
                )
                native_returns += discount * new_goal.float() * native_success_reward
                lava_reward = float(self.reward_settings["lava_reward"])
                failure_penalties += discount * lava_terminated.float() * lava_reward
                active_position = (~finished) & ~lava_terminated & ~new_truncation
                end_positions = torch.where(
                    active_position.unsqueeze(-1), after_positions, end_positions
                )
                end_steps = torch.where(
                    (new_goal | lava_terminated | new_truncation),
                    torch.full_like(end_steps, step),
                    end_steps,
                )
                failure_terminal |= lava_terminated | new_truncation
                finished |= new_goal | lava_terminated | new_truncation
            uncertainty_sum += transition["uncertainty"].sum().to(torch.float64)
            invalid_sum += invalid.sum()

        goal_guide_scores = torch.zeros_like(scores)
        if not use_legacy_dense:
            distance_map = reward_context.goal_distance_map
            start_positions = utils.get_agent_position_torch(
                start_state.unsqueeze(0)
            ).expand(self.population, -1)
            start_distances = distance_map[
                start_positions[:, 0], start_positions[:, 1]
            ]
            end_distances = distance_map[end_positions[:, 0], end_positions[:, 1]]
            finite_distances = torch.isfinite(start_distances) & torch.isfinite(end_distances)
            normalized_progress = (
                (start_distances - end_distances) / start_distances.clamp(min=1.0)
            ).clamp(-1.0, 1.0)
            guide_valid = finite_distances & ~failure_terminal
            guide_discount = torch.pow(
                torch.full_like(scores, self.gamma), end_steps.float()
            )
            goal_guide_scores = (
                self.goal_guide_weight
                * normalized_progress
                * guide_discount
                * guide_valid.float()
            )
            scores = native_returns + failure_penalties + goal_guide_scores
        legacy_dense_returns = scores.clone() if use_legacy_dense else torch.zeros_like(scores)
        diagnostics = {
            "mean_rollout_uncertainty": float(uncertainty_sum.item()) / (self.population * self.horizon),
            "invalid_imagined_actions": int(invalid_sum.item()),
            "imagined_transitions": self.population * self.horizon,
        }
        return (
            scores,
            predicted_positions,
            predicted_inventories,
            native_returns,
            goal_guide_scores,
            failure_penalties,
            legacy_dense_returns,
            diagnostics,
        )

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        inventory_token: int,
        goal_yx: tuple[int, int],
        reward_context: DenseRewardContext,
        episode_step: int,
        max_episode_steps: int,
    ) -> MPCPlan:
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        if state.ndim != 3:
            raise ValueError(f"Expected state [3,H,W], got {tuple(state.shape)}")
        probabilities = None
        best = None
        total_imagined = 0
        total_invalid = 0
        uncertainty_values = []
        self.action_hashes = []
        self._forward_decode_events = []
        self._forward_decode_seconds = 0.0
        for iteration in range(self.iterations):
            sequences = self._sample(probabilities)
            if self.capture_action_hashes:
                digest = hashlib.sha256(sequences.contiguous().cpu().numpy().tobytes()).hexdigest()
                self.action_hashes.append(("candidate_sequences", iteration, digest))
            (
                scores,
                predicted_positions,
                predicted_inventories,
                native_returns,
                goal_guide_scores,
                failure_penalties,
                legacy_dense_returns,
                diagnostics,
            ) = self._evaluate(
                state, inventory_token, sequences, goal_yx, reward_context,
                episode_step, max_episode_steps,
            )
            total_imagined += int(diagnostics["imagined_transitions"])
            total_invalid += int(diagnostics["invalid_imagined_actions"])
            uncertainty_values.append(float(diagnostics["mean_rollout_uncertainty"]))
            if self.capture_action_hashes:
                ranking = scores.argsort(descending=True).contiguous()
                digest = hashlib.sha256(ranking.cpu().numpy().tobytes()).hexdigest()
                self.action_hashes.append(("candidate_ranking", iteration, digest))
            best_index = int(scores.argmax().item())
            if best is None or scores[best_index] > best[0]:
                best = (
                    scores[best_index].detach().clone(),
                    sequences[best_index].detach().clone(),
                    predicted_positions[best_index].detach().clone(),
                    predicted_inventories[best_index].detach().clone(),
                    native_returns[best_index].detach().clone(),
                    goal_guide_scores[best_index].detach().clone(),
                    failure_penalties[best_index].detach().clone(),
                    legacy_dense_returns[best_index].detach().clone(),
                )
            elite_indices = scores.topk(self.elite_count).indices
            elite_actions = sequences[elite_indices]
            counts = torch.nn.functional.one_hot(
                elite_actions, num_classes=MODEL_ACTION_COUNT
            ).float().sum(dim=0)
            probabilities = (counts + 1.0) / (self.elite_count + MODEL_ACTION_COUNT)

        assert best is not None
        (
            score,
            actions,
            predicted_positions,
            predicted_inventories,
            native_return,
            goal_guide_score,
            failure_penalty,
            legacy_dense_return,
        ) = best
        forward_decode_seconds = self._forward_decode_seconds
        if self.capture_profile_timing and DEVICE.type == "cuda":
            torch.cuda.synchronize()
            forward_decode_seconds = sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._forward_decode_events
            )
        if self.capture_action_hashes:
            digest = hashlib.sha256(actions.contiguous().cpu().numpy().tobytes()).hexdigest()
            self.action_hashes.append(("selected_action_list", -1, digest))
        return MPCPlan(
            actions=[int(action) for action in actions.detach().cpu().tolist()],
            score=float(score.item()),
            predicted_positions=[
                tuple(map(int, position))
                for position in predicted_positions.cpu().tolist()
            ],
            predicted_inventories=[
                int(inventory) for inventory in predicted_inventories.cpu().tolist()
            ],
            diagnostics={
                "iterations": self.iterations,
                "population": self.population,
                "horizon": self.horizon,
                "imagined_transitions": total_imagined,
                "invalid_imagined_actions": total_invalid,
                "mean_rollout_uncertainty": float(np.mean(uncertainty_values)),
                "native_return": float(native_return.item()),
                "goal_guide_score": float(goal_guide_score.item()),
                "failure_penalty": float(failure_penalty.item()),
                "legacy_dense_return": float(legacy_dense_return.item()),
                "reward_mode": self.reward_mode,
                "model_forward_decode_seconds": forward_decode_seconds,
            },
        )


def _load_world_model(cfg: DictConfig):
    hparams = cfg.attention_model
    if str(cfg.domain) != "minigrid" or str(hparams.model_type).lower() != "attention":
        raise ValueError("Online MPC currently supports Attention WM MiniGrid only")
    model = AttentionWM_support.AttentionModule(
        hparams.data_type,
        hparams.grid_shape,
        hparams.attention_mask_size,
        hparams.embed_dim,
        hparams.num_heads,
        env_type=hparams.env_type,
        frame_stack=hparams.frame_stack,
        minigrid_transition_mode=getattr(hparams, "minigrid_transition_mode", "effect"),
    ).to(DEVICE)
    configured = getattr(cfg.PPO, "checkpoint_path_wm", None)
    checkpoint = (
        Path(str(configured)).expanduser().resolve()
        if configured is not None and str(configured).strip().lower() not in {"", "null", "none"}
        else world_model_checkpoint_path(cfg, "minigrid")
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"World-model checkpoint not found: {checkpoint}")
    utils.load_model_weight(model, str(checkpoint))
    model.eval()
    return model, checkpoint


def _state_from_observation(observation) -> np.ndarray:
    return minigrid_utils.ColRowCanl_to_CanlRowCol(observation["image"])


def _crafter_resource_distances(objects: np.ndarray, resource_ids: tuple[int, ...]) -> np.ndarray:
    """Steps to a walkable cell adjacent to a visible resource."""
    height, width = objects.shape
    walkable = np.isin(objects, (0, 2, 4, 5, 13, 18))
    resources = np.isin(objects, resource_ids)
    distances = np.full((height, width), np.inf, dtype=np.float32)
    queue = deque()
    for y, x in np.argwhere(resources):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < height and 0 <= nx < width and walkable[ny, nx] and not np.isfinite(distances[ny, nx]):
                distances[ny, nx] = 0.0
                queue.append((ny, nx))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and walkable[ny, nx] and not np.isfinite(distances[ny, nx]):
                distances[ny, nx] = distances[y, x] + 1.0
                queue.append((ny, nx))
    return distances


def _crafter_guided_resource_prefix(
    objects: np.ndarray, position: tuple[int, int], direction: int,
    distances: np.ndarray, resource_id: int, horizon: int,
    interaction_action: int | None = 5,
) -> list[int]:
    """Follow a real-map distance gradient, then attempt an interaction."""
    y, x = position
    if not np.isfinite(distances[y, x]):
        return []
    # Action IDs and direction IDs use different orderings.
    moves = ((3, 1, -1, 0), (4, 2, 1, 0), (1, 3, 0, -1), (2, 4, 0, 1))
    prefix = []
    while distances[y, x] > 0 and len(prefix) < horizon:
        next_step = None
        for action, facing, dy, dx in moves:
            ny, nx = y + dy, x + dx
            if (0 <= ny < objects.shape[0] and 0 <= nx < objects.shape[1]
                    and distances[ny, nx] < distances[y, x]):
                next_step = (action, facing, ny, nx)
                break
        if next_step is None:
            return []
        action, direction, y, x = next_step
        prefix.append(action)
    if distances[y, x] > 0 or interaction_action is None:
        return prefix
    for action, facing, dy, dx in moves:
        ny, nx = y + dy, x + dx
        if (0 <= ny < objects.shape[0] and 0 <= nx < objects.shape[1]
                and objects[ny, nx] == resource_id):
            if direction != facing and len(prefix) < horizon:
                prefix.append(action)
            if len(prefix) < horizon:
                prefix.append(interaction_action)
            break
    return prefix


def _crafter_distance_progress(
    distances: torch.Tensor, positions: torch.Tensor, max_distance: float = 8.0
) -> torch.Tensor:
    """Convert a BFS distance to a bounded, small proximity potential."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    height, width = distances.shape
    valid = (
        positions[:, 0].ge(0) & positions[:, 0].lt(height)
        & positions[:, 1].ge(0) & positions[:, 1].lt(width)
    )
    y = positions[:, 0].clamp(0, height - 1)
    x = positions[:, 1].clamp(0, width - 1)
    distance = distances[y, x]
    finite = torch.isfinite(distance) & valid
    return torch.where(
        finite,
        (1.0 - distance / float(max_distance)).clamp(0.0, 1.0),
        torch.zeros_like(distance),
    )


def _crafter_tool_progress(
    objects: torch.Tensor,
    inventories: torch.Tensor,
    positions: torch.Tensor,
    tree_distances: torch.Tensor,
    stone_distances: torch.Tensor,
    table_distances: torch.Tensor,
    *,
    reserve_wood_for_first_table: bool = False,
) -> torch.Tensor:
    """Return four bounded potentials for the wood-to-stone tool chain.

    Columns are table preparation, wood-pickaxe readiness, stone acquisition,
    and stone-pickaxe readiness. Distances come from the current real map; an
    imagined table is recognized from the endpoint map and gets its distance
    map at the next real replanning boundary.
    """
    if objects.ndim == 2:
        objects = objects.unsqueeze(0)
    if inventories.ndim == 1:
        inventories = inventories.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if objects.ndim != 3 or inventories.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("Crafter tool progress expects batched objects, inventories, and positions")
    if inventories.shape[1] != 16:
        raise ValueError("Crafter inventory must contain 16 slots")

    wood = inventories[:, 4].clamp(0.0, 9.0)
    stone = inventories[:, 5].clamp(0.0, 9.0)
    wood_pickaxe = inventories[:, 10].ge(0.5)
    stone_pickaxe = inventories[:, 11].ge(0.5)
    table_exists = objects.eq(11).flatten(1).any(dim=1)

    tree_proximity = _crafter_distance_progress(tree_distances, positions)
    stone_proximity = _crafter_distance_progress(stone_distances, positions)
    table_proximity = _crafter_distance_progress(table_distances, positions)
    # A newly imagined table has no distance map yet. It is treated as usable
    # for this endpoint; the next real replan computes its BFS distance.
    has_observed_table = torch.isfinite(table_distances).any()
    observed_table_access = torch.where(
        table_exists, table_proximity, torch.zeros_like(table_proximity)
    )
    imagined_table_access = torch.where(
        table_exists, torch.ones_like(table_proximity), table_proximity
    )
    table_access = torch.where(
        has_observed_table, observed_table_access, imagined_table_access
    )
    height, width = table_distances.shape
    valid_position = (
        positions[:, 0].ge(0) & positions[:, 0].lt(height)
        & positions[:, 1].ge(0) & positions[:, 1].lt(width)
    )
    y = positions[:, 0].clamp(0, height - 1)
    x = positions[:, 1].clamp(0, width - 1)
    table_reachable = (
        table_exists & valid_position
        & (torch.isfinite(table_distances[y, x]) | ~has_observed_table)
    )

    table_material = (wood / 2.0).clamp(0.0, 1.0)
    table_progress = torch.where(
        table_reachable,
        torch.ones_like(table_material),
        table_material + 0.2 * (1.0 - table_material) * tree_proximity,
    )
    # The first wood pickaxe needs another wood after placing the table.
    # Keep tree proximity useful even while the table is farther away.
    wood_material = wood.clamp(max=1.0)
    wood_supply = wood_material + 0.2 * (1.0 - wood_material) * tree_proximity
    craft_access = torch.where(
        table_reachable,
        0.8 + 0.2 * table_access,
        torch.zeros_like(table_access),
    )
    # Before the first table, reserve its two wood plus one for the pickaxe.
    # A third wood then advances the tool-chain potential before placement.
    reserve_material = (wood - 2.0).clamp(0.0, 1.0)
    reserve_progress = torch.where(
        wood.ge(2.0),
        reserve_material + 0.2 * (1.0 - reserve_material) * tree_proximity,
        torch.zeros_like(wood),
    )
    wood_pickaxe_progress = torch.where(
        wood_pickaxe,
        torch.ones_like(table_material),
        torch.where(
            ~table_exists, reserve_progress, craft_access * wood_supply
        ) if reserve_wood_for_first_table else craft_access * wood_supply,
    )
    stone_progress = torch.where(
        stone.ge(0.5) | stone_pickaxe,
        torch.ones_like(table_material),
        torch.where(wood_pickaxe, stone_proximity, torch.zeros_like(stone_proximity)),
    )
    stone_pickaxe_progress = torch.where(
        stone_pickaxe,
        torch.ones_like(table_material),
        craft_access * stone.clamp(max=1.0) * wood_supply,
    )
    return torch.stack(
        (table_progress, wood_pickaxe_progress, stone_progress, stone_pickaxe_progress),
        dim=1,
    )


def _crafter_inventory_progress_stage(objects: np.ndarray, inventory: np.ndarray) -> int:
    """Current real tool tier, with table placement as the first substage."""
    if inventory[12] >= 0.5:
        return 4  # iron pickaxe: seek diamond
    if inventory[11] >= 0.5:
        return 3  # stone pickaxe: prepare iron pickaxe
    if inventory[10] >= 0.5:
        return 2  # wood pickaxe: seek stone
    return 1 if np.any(objects == 11) else 0  # table placed or not


def _crafter_inventory_progress_targets(stage: int, furnace_exists: bool) -> tuple[tuple[int, float], ...]:
    """Only materials still needed for the next tool milestone earn progress."""
    if stage == 0:
        return ((4, 3.0),)  # two wood for table, one for wood pickaxe
    if stage == 1:
        return ((4, 1.0),)
    if stage == 2:
        return ((5, 1.0),)  # stone pickaxe
    if stage == 3:
        materials = ((6, 1.0), (7, 1.0))  # coal and iron
        return materials if furnace_exists else ((5, 4.0),) + materials
    if stage == 4:
        return ((8, 1.0),)  # diamond
    raise ValueError(f"Unknown Crafter inventory progress stage: {stage}")


def _crafter_inventory_progress_gain(
    inventories: torch.Tensor, peaks: torch.Tensor,
    targets: tuple[tuple[int, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count newly reached material thresholds once per stage and episode."""
    if not targets:
        return torch.zeros(inventories.shape[0], device=inventories.device), peaks
    slots = [slot for slot, _ in targets]
    caps = inventories.new_tensor([cap for _, cap in targets])
    observed = inventories[:, slots].clamp_min(0.0).minimum(caps)
    new_peaks = peaks.clone()
    new_peaks[:, slots] = torch.maximum(peaks[:, slots], observed)
    gain = (new_peaks[:, slots] - peaks[:, slots]).sum(dim=1)
    return gain, new_peaks


def _crafter_late_tool_progress(
    objects: torch.Tensor, inventories: torch.Tensor, positions: torch.Tensor,
    tree_distances: torch.Tensor, stone_distances: torch.Tensor,
    coal_distances: torch.Tensor, iron_distances: torch.Tensor,
    diamond_distances: torch.Tensor, table_distances: torch.Tensor,
) -> torch.Tensor:
    """Bounded preparation for iron pickaxe, then diamond after it is held."""
    if objects.ndim == 2:
        objects = objects.unsqueeze(0)
    if inventories.ndim == 1:
        inventories = inventories.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    stone_pickaxe = inventories[:, 11].ge(0.5)
    iron_pickaxe = inventories[:, 12].ge(0.5)
    active = stone_pickaxe | iron_pickaxe
    furnace = objects.eq(12).flatten(1).any(dim=1)

    def material(slot: int, distances: torch.Tensor) -> torch.Tensor:
        held = inventories[:, slot].clamp(0.0, 1.0)
        pending = held + 0.2 * (1.0 - held) * _crafter_distance_progress(distances, positions)
        return torch.where(iron_pickaxe, torch.ones_like(held),
                           torch.where(active, pending, torch.zeros_like(held)))

    wood = material(4, tree_distances)
    coal = material(6, coal_distances)
    iron = material(7, iron_distances)
    stone_fraction = (inventories[:, 5] / 4.0).clamp(0.0, 1.0)
    furnace_preparation = (
        0.7 * stone_fraction
        + 0.2 * _crafter_distance_progress(table_distances, positions)
        + 0.1 * (1.0 - stone_fraction) * _crafter_distance_progress(stone_distances, positions)
    ).clamp(0.0, 1.0)
    furnace_ready = torch.where(furnace | iron_pickaxe, torch.ones_like(stone_fraction),
                                furnace_preparation)
    furnace_ready = torch.where(active, furnace_ready, torch.zeros_like(furnace_ready))
    diamond_held = inventories[:, 8].ge(0.5)
    diamond = torch.where(
        iron_pickaxe,
        torch.where(diamond_held, torch.ones_like(stone_fraction),
                    0.2 * _crafter_distance_progress(diamond_distances, positions)),
        torch.zeros_like(stone_fraction),
    )
    return torch.stack((wood, coal, iron, furnace_ready, diamond), dim=1)


def _crafter_late_crafting_prefix(
    objects: np.ndarray, position: tuple[int, int], direction: int,
    horizon: int, craft_action: int,
) -> list[int]:
    """Reach a valid furnace placement or pickaxe crafting pose."""
    if craft_action not in (9, 11, 12, 13):
        raise ValueError("Crafter crafting route supports furnace and pickaxes")
    tables = np.argwhere(objects == 11)
    furnaces = np.argwhere(objects == 12)
    if not len(tables) or (craft_action == 13 and not len(furnaces)):
        return []
    moves = ((3, 1, -1, 0), (4, 2, 1, 0), (1, 3, 0, -1), (2, 4, 0, 1))
    offsets = {facing: (dy, dx) for _, facing, dy, dx in moves}
    walkable = (0, 2, 4, 5, 13, 18)

    def near(cells: np.ndarray, y: int, x: int) -> bool:
        return bool(np.any((np.abs(cells[:, 0] - y) <= 1)
                           & (np.abs(cells[:, 1] - x) <= 1)))

    def ready(y: int, x: int, facing: int) -> bool:
        if not near(tables, y, x):
            return False
        if craft_action in (11, 12):
            return True
        if craft_action == 13:
            return near(furnaces, y, x)
        dy, dx = offsets.get(facing, (0, 0))
        ny, nx = y + dy, x + dx
        return (facing in offsets and 0 <= ny < objects.shape[0]
                and 0 <= nx < objects.shape[1]
                and objects[ny, nx] in (2, 4, 5))

    start = (*position, direction)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (y, x, facing), route = queue.popleft()
        if ready(y, x, facing):
            return route + [craft_action]
        if len(route) + 1 >= horizon:
            continue
        for action, next_facing, dy, dx in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < objects.shape[0] and 0 <= nx < objects.shape[1]
                    and objects[ny, nx] in walkable):
                ny, nx = y, x
            next_state = (ny, nx, next_facing)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, route + [action]))
    return []


def _crafter_late_guided_prefixes(
    objects: np.ndarray, inventory: np.ndarray, position: tuple[int, int],
    direction: int, distances: dict[int, np.ndarray], horizon: int,
) -> list[tuple[list[int], bool]]:
    """Propose routes toward missing materials after a real pickaxe unlock."""
    stone_pickaxe = inventory[11] >= 0.5
    iron_pickaxe = inventory[12] >= 0.5
    if not (stone_pickaxe or iron_pickaxe):
        return []
    furnace_exists = bool((objects == 12).any())
    targets = []
    if iron_pickaxe:
        if inventory[8] < 0.5:
            targets.append(10)  # diamond
    else:
        if inventory[4] < 1:
            targets.append(6)  # tree
        if inventory[6] < 1:
            targets.append(8)  # coal
        if inventory[7] < 1:
            targets.append(9)  # iron
        if not furnace_exists and inventory[5] < 4:
            targets.append(3)  # stone for furnace
    y, x = position
    targets.sort(key=lambda resource: float(distances[resource][y, x]))
    proposals = []
    for resource in targets:
        prefix = _crafter_guided_resource_prefix(
            objects, position, direction, distances[resource], resource, horizon
        )
        if prefix:
            proposals.append((prefix, 5 in prefix[:horizon]))
    if stone_pickaxe and not iron_pickaxe:
        craft_action = None
        if not furnace_exists and inventory[5] >= 4:
            craft_action = 9  # place_furnace
        elif (furnace_exists and inventory[4] >= 1 and inventory[6] >= 1
              and inventory[7] >= 1):
            craft_action = 13  # make_iron_pickaxe
        if craft_action is not None:
            prefix = _crafter_late_crafting_prefix(
                objects, position, direction, horizon, craft_action
            )
            if prefix:
                proposals.append((prefix, True))
    return proposals


def _crafter_initial_visit_map(objects: np.ndarray) -> np.ndarray:
    """Start a per-episode visit mask with the current player cell marked."""
    player_yx = np.argwhere(np.asarray(objects) == 13)
    if len(player_yx) != 1:
        raise ValueError("Crafter MPC expects exactly one player on the current map")
    visited = np.zeros(np.asarray(objects).shape, dtype=np.bool_)
    visited[tuple(player_yx[0])] = True
    return visited


def _crafter_unvisited_distances(objects: np.ndarray, visited: np.ndarray) -> np.ndarray:
    """BFS distance to the nearest currently reachable unvisited walkable tile."""
    height, width = objects.shape
    if visited.shape != objects.shape:
        raise ValueError(f"Crafter visited map shape {visited.shape} does not match {objects.shape}")
    walkable = np.isin(objects, (0, 2, 4, 5, 13, 18))
    targets = walkable & ~visited
    distances = np.full((height, width), np.inf, dtype=np.float32)
    queue = deque()
    for y, x in np.argwhere(targets):
        distances[y, x] = 0.0
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        next_distance = distances[y, x] + 1.0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= ny < height and 0 <= nx < width and walkable[ny, nx]
                    and not np.isfinite(distances[ny, nx])):
                distances[ny, nx] = next_distance
                queue.append((ny, nx))
    return distances


def _crafter_exploration_bonus(
    predicted_positions: torch.Tensor,
    visited: torch.Tensor,
    frontier_distances: torch.Tensor,
    start_distance: float,
    alive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score unique imagined new cells and progress toward the nearest frontier."""
    population, horizon, _ = predicted_positions.shape
    height, width = visited.shape
    batch_ids = torch.arange(population, device=predicted_positions.device)
    imagined_seen = torch.zeros(
        (population, height, width), dtype=torch.bool, device=predicted_positions.device
    )
    novel_count = torch.zeros(population, dtype=torch.float32, device=predicted_positions.device)
    for step in range(horizon):
        positions = predicted_positions[:, step]
        valid_position = positions[:, 0].ge(0) & positions[:, 0].lt(height)
        valid_position &= positions[:, 1].ge(0) & positions[:, 1].lt(width) & alive
        y = positions[:, 0].clamp(0, height - 1)
        x = positions[:, 1].clamp(0, width - 1)
        first_visit = valid_position & ~visited[y, x] & ~imagined_seen[batch_ids, y, x]
        novel_count += first_visit.float()
        imagined_seen[batch_ids[valid_position], y[valid_position], x[valid_position]] = True

    bonus = (0.1 * novel_count).clamp(max=0.5)
    if np.isfinite(start_distance) and start_distance > 0:
        final_positions = predicted_positions[:, -1]
        valid_final = final_positions[:, 0].ge(0) & final_positions[:, 0].lt(height)
        valid_final &= final_positions[:, 1].ge(0) & final_positions[:, 1].lt(width) & alive
        y = final_positions[:, 0].clamp(0, height - 1)
        x = final_positions[:, 1].clamp(0, width - 1)
        end_distance = frontier_distances[y, x]
        finite_end = torch.isfinite(end_distance) & valid_final
        progress = ((start_distance - torch.where(
            finite_end, end_distance, torch.full_like(end_distance, start_distance)
        )) / 4.0).clamp(0.0, 1.0)
        bonus += 0.25 * progress
    reachable_frontier = bool(np.isfinite(start_distance) and start_distance > 0)
    if not reachable_frontier:
        bonus.zero_()
        novel_count.zero_()
    bonus = torch.where(alive, bonus, 0.0)
    novel_count = torch.where(alive, novel_count, 0.0)
    return bonus, novel_count


def _crafter_context(env: CustomCrafterEnv, state: torch.Tensor):
    """Synchronize hidden simulator progress at each planning boundary."""
    player = env.env._player
    unlocked = torch.tensor(
        [[bool(player.achievements.get(name, 0)) for name in ACHIEVEMENT_NAMES]],
        device=DEVICE, dtype=torch.bool,
    )
    life = {
        name: torch.tensor([getattr(player, attr)], device=DEVICE, dtype=dtype)
        for name, attr, dtype in (
            ("hunger", "_hunger", torch.float32),
            ("thirst", "_thirst", torch.float32),
            ("fatigue", "_fatigue", torch.float32),
            ("recover", "_recover", torch.float32),
            ("sleeping", "sleeping", torch.bool),
        )
    }
    hp = torch.full_like(state[0], -1.0)
    for entity in env.env._world.objects:
        if type(entity).__name__.lower() in {"cow", "zombie", "skeleton"}:
            x, y = map(int, entity.pos)
            if 0 <= y < hp.shape[0] and 0 <= x < hp.shape[1]:
                hp[y, x] = max(0.0, float(entity.health))
    return unlocked, life, hp.unsqueeze(0)


class CrafterWorldModelMPC:
    """Elite categorical search over Crafter's 17 native actions."""

    def __init__(self, model, spec, config):
        self.model, self.spec = model, spec
        self.horizon = int(config.horizon)
        self.population = int(config.population)
        self.elite_count = int(config.elite_count)
        self.iterations = int(config.iterations)
        self.execute_steps = int(config.execute_steps)
        self.gamma = float(config.gamma)
        self.fallback_action = int(config.fallback_action)
        self.terminal_penalty = float(config.crafter_terminal_penalty)
        self.resource_guidance_weight = float(config.crafter_resource_guidance_weight)
        self.exploration_weight = float(config.crafter_exploration_weight)
        self.tool_progress_weight = float(getattr(config, "crafter_tool_progress_weight", 0.25))
        self.require_three_wood_for_first_table = bool(
            getattr(config, "crafter_require_three_wood_for_first_table", False)
        )
        self.late_stage_guidance = bool(getattr(config, "crafter_late_stage_guidance", False))
        self.inventory_progress_weight = float(getattr(config, "crafter_inventory_progress_weight", 0.0))
        self.action_count = len(CRAFTER_ACTION_NAMES)
        if spec.action_count != self.action_count:
            raise ValueError(f"Crafter WM supports {spec.action_count} actions; expected {self.action_count}")
        if min(self.horizon, self.population, self.iterations) < 1:
            raise ValueError("MPC horizon, population, and iterations must be positive")
        if not 1 <= self.elite_count <= self.population:
            raise ValueError("MPC elite_count must be in [1, population]")
        if not 1 <= self.execute_steps <= self.horizon:
            raise ValueError("MPC execute_steps must be in [1, horizon]")
        if not 0 <= self.fallback_action < self.action_count:
            raise ValueError("MPC fallback_action must be a native Crafter action")
        if not np.isfinite(self.terminal_penalty) or self.terminal_penalty < 0:
            raise ValueError("PPO.mpc.crafter_terminal_penalty must be finite and nonnegative")
        if not np.isfinite(self.resource_guidance_weight) or self.resource_guidance_weight < 0:
            raise ValueError("PPO.mpc.crafter_resource_guidance_weight must be finite and nonnegative")
        if not np.isfinite(self.exploration_weight) or self.exploration_weight < 0:
            raise ValueError("PPO.mpc.crafter_exploration_weight must be finite and nonnegative")
        if not np.isfinite(self.tool_progress_weight) or self.tool_progress_weight < 0:
            raise ValueError("PPO.mpc.crafter_tool_progress_weight must be finite and nonnegative")
        if not np.isfinite(self.inventory_progress_weight) or self.inventory_progress_weight < 0:
            raise ValueError("PPO.mpc.crafter_inventory_progress_weight must be finite and nonnegative")

    @torch.no_grad()
    def plan(
        self, state: torch.Tensor, inventory: torch.Tensor, context,
        visited_positions: np.ndarray, inventory_progress_peaks: np.ndarray | None = None,
    ) -> MPCPlan:
        state = state.to(device=DEVICE, dtype=torch.float32)
        inventory = inventory.to(device=DEVICE, dtype=torch.float32)
        if state.ndim != 3 or state.shape[0] != 2 or inventory.shape != (16,):
            raise ValueError("Crafter MPC expects [2,H,W] grid and 16 inventory values")
        initial_unlocked, initial_life, initial_hp = context
        objects = state[0].long().cpu().numpy()
        player_yx = np.argwhere(objects == 13)
        if len(player_yx) != 1:
            raise ValueError("Crafter MPC expects exactly one player on the current map")
        start_y, start_x = map(int, player_yx[0])
        stage = _crafter_inventory_progress_stage(objects, inventory.cpu().numpy())
        stage_targets = _crafter_inventory_progress_targets(stage, bool((objects == 12).any()))
        if inventory_progress_peaks is None:
            stage_peaks = inventory.clone()
        else:
            history = np.asarray(inventory_progress_peaks, dtype=np.float32)
            if history.shape != (5, 16):
                raise ValueError("Crafter MPC inventory progress peaks must have shape [5, 16]")
            stage_peaks = torch.as_tensor(history[stage], device=DEVICE)
        visited_positions = np.asarray(visited_positions, dtype=np.bool_)
        if visited_positions.shape != objects.shape:
            raise ValueError(
                f"Crafter visited map shape {visited_positions.shape} does not match {objects.shape}"
            )
        visited = torch.as_tensor(visited_positions, device=DEVICE, dtype=torch.bool)
        frontier_values = _crafter_unvisited_distances(objects, visited_positions)
        frontier_distances = torch.as_tensor(frontier_values, device=DEVICE)
        start_frontier_distance = float(frontier_values[start_y, start_x])
        resource_maps = (
            _crafter_resource_distances(objects, (14,)),  # food: cow; plant ripeness is hidden
            _crafter_resource_distances(objects, (1,)),       # drink: water
        )
        resource_distances = tuple(torch.as_tensor(values, device=DEVICE) for values in resource_maps)
        start_distances = tuple(float(values[start_y, start_x]) for values in resource_maps)
        tool_maps = (
            _crafter_resource_distances(objects, (6,)),  # tree
            _crafter_resource_distances(objects, (3,)),  # stone
            _crafter_resource_distances(objects, (11,)),  # table
        )
        tool_distances = tuple(torch.as_tensor(values, device=DEVICE) for values in tool_maps)
        start_position = torch.tensor([start_y, start_x], device=DEVICE, dtype=torch.long)
        start_tool_progress = _crafter_tool_progress(
            torch.as_tensor(objects, device=DEVICE, dtype=torch.long),
            inventory, start_position, *tool_distances,
            reserve_wood_for_first_table=self.require_three_wood_for_first_table,
        )[0]
        late_active = self.late_stage_guidance and bool(
            inventory[11].ge(0.5) or inventory[12].ge(0.5)
        )
        late_distances = None
        if late_active:
            late_maps = {6: tool_maps[0], 3: tool_maps[1], 11: tool_maps[2]}
            for resource in (8, 9, 10, 12):
                late_maps[resource] = _crafter_resource_distances(objects, (resource,))
            late_distances = tuple(
                torch.as_tensor(late_maps[resource], device=DEVICE)
                for resource in (6, 3, 8, 9, 10, 11)
            )
            start_late_progress = _crafter_late_tool_progress(
                torch.as_tensor(objects, device=DEVICE, dtype=torch.long),
                inventory, start_position, *late_distances,
            )[0]
            guided_prefixes = _crafter_late_guided_prefixes(
                objects, inventory.cpu().numpy(), (start_y, start_x),
                int(state[1, start_y, start_x].item()), late_maps, self.horizon,
            )
        else:
            start_late_progress = torch.zeros(5, device=DEVICE)
            guided_prefixes = []
        if self.inventory_progress_weight > 0 and stage <= 2:
            direction = int(state[1, start_y, start_x].item())
            resource = 6 if stage <= 1 else 3
            slot, cap = stage_targets[0]
            if float(stage_peaks[slot]) < cap:
                distance = tool_maps[0 if resource == 6 else 1]
                prefix = _crafter_guided_resource_prefix(
                    objects, (start_y, start_x), direction,
                    distance, resource, self.horizon,
                )
                if prefix:
                    guided_prefixes.append((prefix, 5 in prefix))
            craft_action = None
            if stage == 1 and inventory[4] >= 1:
                craft_action = 11  # make_wood_pickaxe
            elif stage == 2 and inventory[4] >= 1 and inventory[5] >= 1:
                craft_action = 12  # make_stone_pickaxe
            if craft_action is not None:
                prefix = _crafter_late_crafting_prefix(
                    objects, (start_y, start_x), direction, self.horizon, craft_action
                )
                if prefix:
                    guided_prefixes.append((prefix, True))
        best = probabilities = None
        imagined = 0
        invalid = torch.zeros((), device=DEVICE, dtype=torch.int64)
        for _ in range(self.iterations):
            if probabilities is None:
                sequences = torch.randint(
                    self.action_count, (self.population, self.horizon), device=DEVICE
                )
            else:
                sequences = torch.multinomial(
                    probabilities.unsqueeze(0).expand(self.population, -1, -1).reshape(-1, self.action_count),
                    1,
                ).reshape(self.population, self.horizon)
            guided_bonus = torch.zeros(self.population, device=DEVICE)
            for index, (prefix, attempts_resource) in enumerate(guided_prefixes[:self.population]):
                sequences[index, :len(prefix)] = torch.as_tensor(prefix, device=DEVICE)
                if attempts_resource and len(prefix) <= self.execute_steps:
                    weight = self.inventory_progress_weight if stage <= 2 else self.tool_progress_weight
                    guided_bonus[index] = weight * self.gamma ** (len(prefix) - 1)
            states = state.unsqueeze(0).expand(self.population, -1, -1, -1).clone()
            inventories = inventory.unsqueeze(0).expand(self.population, -1).clone()
            unlocked = initial_unlocked.expand(self.population, -1).clone()
            life = {name: value.expand(self.population).clone() for name, value in initial_life.items()}
            hp = initial_hp.expand(self.population, -1, -1).clone()
            scores = torch.zeros(self.population, device=DEVICE)
            inventory_progress_contribution = torch.zeros_like(scores)
            candidate_peaks = stage_peaks.unsqueeze(0).expand(self.population, -1).clone()
            done = torch.zeros(self.population, device=DEVICE, dtype=torch.bool)
            rollout_lengths = torch.zeros(self.population, device=DEVICE, dtype=torch.float32)
            positions = torch.full(
                (self.population, self.horizon, 2), -1, device=DEVICE, dtype=torch.long
            )
            predicted_inventory = torch.empty(
                (self.population, self.horizon, 16), device=DEVICE
            )
            prefix_state = prefix_inventory = None
            prefix_length = 0
            for t in range(self.horizon):
                actions = sequences[:, t]
                if self.require_three_wood_for_first_table:
                    premature_table = (
                        actions.eq(CRAFTER_ACTION_NAMES.index("place_table"))
                        & inventories[:, 4].lt(3.0)
                        & inventories[:, 10].lt(0.5)
                        & ~states[:, 0].eq(11).flatten(1).any(dim=1)
                    )
                    # Replace only this impossible-for-the-goal proposal; CEM
                    # updates from the actions it actually imagined and returns.
                    actions = torch.where(premature_table, torch.zeros_like(actions), actions)
                    sequences[:, t] = actions
                next_states, next_inv = imagined_crafter_step_batch(
                    self.model, states, actions, inventories,
                    self.spec.attention_mask_size, self.spec.inventory_output_mode,
                    predict_survival=self.spec.predict_survival,
                    inventory_value_mode=self.spec.inventory_value_mode,
                )
                reward, _, next_unlocked, next_life, next_hp, next_inv = native_reward_batch(
                    states, inventories, actions, next_states, next_inv,
                    unlocked, life["sleeping"], hp, life,
                )
                valid = crafter_player_counts(next_states).eq(1)
                valid &= torch.isfinite(next_states).flatten(1).all(dim=1)
                valid &= torch.isfinite(next_inv).all(dim=1)
                active = ~done
                invalid += (active & ~valid).sum()
                rollout_lengths += (active & valid).float()
                predicted_death = active & valid & next_inv[:, 0].le(0.5)
                if self.inventory_progress_weight > 0:
                    gain, next_peaks = _crafter_inventory_progress_gain(
                        next_inv, candidate_peaks, stage_targets
                    )
                    eligible = active & valid & ~predicted_death
                    contribution = (self.gamma ** t) * self.inventory_progress_weight * torch.where(
                        eligible, gain, 0.0
                    )
                    scores += contribution
                    inventory_progress_contribution += contribution
                    candidate_peaks = torch.where(eligible.unsqueeze(1), next_peaks, candidate_peaks)
                scores += (self.gamma ** t) * torch.where(active & valid, reward, 0.0)
                scores -= (self.gamma ** t) * (active & ~valid).float()
                scores -= (self.gamma ** t) * predicted_death.float() * self.terminal_penalty
                done |= ~valid | predicted_death
                flat = next_states[:, 0].eq(13).flatten(1)
                index = flat.float().argmax(dim=1)
                width = next_states.shape[-1]
                positions[:, t, 0] = torch.where(valid, index // width, -1)
                positions[:, t, 1] = torch.where(valid, index % width, -1)
                predicted_inventory[:, t] = next_inv
                states, inventories = next_states, next_inv
                if self.tool_progress_weight > 0 and t < self.execute_steps:
                    prefix_state = next_states[:, 0].long().clone()
                    prefix_inventory = next_inv.clone()
                    prefix_length = t + 1
                unlocked, life, hp = next_unlocked, next_life, next_hp
                imagined += self.population
                if bool(done.all()):
                    positions[:, t + 1:] = positions[:, t:t + 1]
                    predicted_inventory[:, t + 1:] = next_inv.unsqueeze(1)
                    break
            # Partial progress toward food/water makes long routes discoverable
            # before a 16-step rollout can reach the resource itself.
            final_positions = positions[:, -1]
            final_y = final_positions[:, 0].clamp(0, state.shape[-2] - 1)
            final_x = final_positions[:, 1].clamp(0, state.shape[-1] - 1)
            guidance = torch.zeros_like(scores)
            for slot, distances, start_distance in zip((1, 2), resource_distances, start_distances):
                guidance += (inventories[:, slot] - inventory[slot]) / 9.0
                if np.isfinite(start_distance) and start_distance > 0:
                    end_distance = distances[final_y, final_x]
                    progress = ((start_distance - torch.where(
                        torch.isfinite(end_distance), end_distance, start_distance
                    )) / min(start_distance, 4.0)).clamp(-1.0, 1.0)
                    urgency = (9.0 - inventory[slot].clamp(0.0, 9.0)) / 9.0
                    guidance += urgency * progress
            scores += (self.gamma ** self.horizon) * self.resource_guidance_weight * torch.where(
                ~done, guidance, 0.0
            )
            exploration_bonus, novel_count = _crafter_exploration_bonus(
                positions, visited, frontier_distances, start_frontier_distance, ~done
            )
            exploration_contribution = (
                (self.gamma ** self.horizon) * self.exploration_weight * exploration_bonus
            )
            scores += exploration_contribution

            final_positions = positions[:, -1]
            final_progress = _crafter_tool_progress(
                states[:, 0].long(), inventories, final_positions, *tool_distances,
                reserve_wood_for_first_table=self.require_three_wood_for_first_table,
            )
            final_progress = torch.where(done.unsqueeze(1), torch.zeros_like(final_progress), final_progress)
            progress_discount = self.gamma ** rollout_lengths
            if self.tool_progress_weight > 0:
                prefix_progress = _crafter_tool_progress(
                    prefix_state, prefix_inventory,
                    positions[:, self.execute_steps - 1], *tool_distances,
                    reserve_wood_for_first_table=self.require_three_wood_for_first_table,
                )
                prefix_progress = torch.where(
                    done.unsqueeze(1), torch.zeros_like(prefix_progress), prefix_progress
                )
                prefix_discount = self.gamma ** prefix_length
                # Half the shaping signal must be reachable before the next replan.
                tool_progress_contribution = self.tool_progress_weight * (
                    0.5 * prefix_discount * prefix_progress.sum(dim=1)
                    + 0.5 * progress_discount * final_progress.sum(dim=1)
                    - start_tool_progress.sum()
                )
            else:
                prefix_progress = torch.zeros_like(final_progress)
                tool_progress_contribution = torch.zeros_like(scores)
            scores += tool_progress_contribution
            late_progress_contribution = torch.zeros_like(scores)
            if late_active:
                final_late_progress = _crafter_late_tool_progress(
                    states[:, 0].long(), inventories, final_positions, *late_distances,
                )
                prefix_late_progress = _crafter_late_tool_progress(
                    prefix_state, prefix_inventory,
                    positions[:, self.execute_steps - 1], *late_distances,
                ) if self.tool_progress_weight > 0 else torch.zeros_like(final_late_progress)
                final_late_progress = torch.where(
                    done.unsqueeze(1), torch.zeros_like(final_late_progress), final_late_progress
                )
                prefix_late_progress = torch.where(
                    done.unsqueeze(1), torch.zeros_like(prefix_late_progress), prefix_late_progress
                )
                if self.tool_progress_weight > 0:
                    late_progress_contribution = self.tool_progress_weight * (
                        0.5 * (self.gamma ** prefix_length) * prefix_late_progress.sum(dim=1)
                        + 0.5 * progress_discount * final_late_progress.sum(dim=1)
                        - start_late_progress.sum()
                    )
                scores += late_progress_contribution
            guided_bonus = torch.where(~done, guided_bonus, torch.zeros_like(guided_bonus))
            scores += guided_bonus

            winner = int(scores.argmax().item())
            if best is None or float(scores[winner]) > float(best[0]):
                best = (
                    scores[winner].clone(), sequences[winner].clone(),
                    positions[winner].clone(), predicted_inventory[winner].clone(),
                    exploration_contribution[winner].clone(), novel_count[winner].clone(),
                    tool_progress_contribution[winner].clone(),
                    prefix_progress[winner].clone(), final_progress[winner].clone(),
                    late_progress_contribution[winner].clone(), guided_bonus[winner].clone(),
                    inventory_progress_contribution[winner].clone(),
                )
            elite = sequences[scores.topk(self.elite_count).indices]
            counts = torch.nn.functional.one_hot(elite, self.action_count).float().sum(dim=0)
            probabilities = (counts + 1.0) / (self.elite_count + self.action_count)
        (
            score, actions, positions, predicted_inventory, exploration_score, novel_cells,
            tool_progress_score, prefix_tool_progress, end_tool_progress,
            late_progress_score, guided_attempt_score, inventory_progress_score,
        ) = best
        return MPCPlan(
            actions=actions.cpu().tolist(), score=float(score),
            predicted_positions=[tuple(map(int, pos)) for pos in positions.cpu().tolist()],
            predicted_inventories=predicted_inventory.cpu().tolist(),
            diagnostics={
                "iterations": self.iterations, "population": self.population,
                "horizon": self.horizon, "imagined_transitions": imagined,
                "invalid_imagined_actions": int(invalid.item()),
                "exploration_score": float(exploration_score.item()),
                "novel_cells": int(novel_cells.item()),
                "tool_progress_score": float(tool_progress_score.item()),
                "late_progress_score": float(late_progress_score.item()),
                "guided_attempt_score": float(guided_attempt_score.item()),
                "guided_candidate_count": min(len(guided_prefixes), self.population),
                "inventory_progress_stage": stage,
                "inventory_progress_score": float(inventory_progress_score.item()),
                "late_stage_active": late_active,
                "tool_progress_start_total": float(start_tool_progress.sum().item()),
                "tool_progress_prefix_total": float(prefix_tool_progress.sum().item()),
                "tool_progress_end_total": float(end_tool_progress.sum().item()),
                "tool_progress_start_table": float(start_tool_progress[0].item()),
                "tool_progress_start_wood_pickaxe": float(start_tool_progress[1].item()),
                "tool_progress_start_stone": float(start_tool_progress[2].item()),
                "tool_progress_start_stone_pickaxe": float(start_tool_progress[3].item()),
                "tool_progress_end_table": float(end_tool_progress[0].item()),
                "tool_progress_end_wood_pickaxe": float(end_tool_progress[1].item()),
                "tool_progress_end_stone": float(end_tool_progress[2].item()),
                "tool_progress_end_stone_pickaxe": float(end_tool_progress[3].item()),
            },
        )


def run_online_crafter_mpc(cfg: DictConfig) -> list[dict]:
    """Plan in the frozen WM and execute bounded prefixes in real Crafter."""
    if str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower() != "mpc":
        raise ValueError("Crafter MPC requires PPO.wm_control_mode=mpc")
    if bool(cfg.PPO.train_in_real_env):
        raise ValueError("MPC requires PPO.train_in_real_env=false")
    mpc_cfg = cfg.PPO.mpc
    if DEVICE.type == "cpu":
        threads = int(mpc_cfg.cpu_threads)
        if threads < 1:
            raise ValueError("PPO.mpc.cpu_threads must be positive")
        torch.set_num_threads(threads)
    _seed_everything(int(cfg.PPO.seed))
    configured = getattr(mpc_cfg, "checkpoint_path", None)
    if configured is None or str(configured).strip().lower() in {"", "null", "none"}:
        configured = mpc_cfg.crafter_checkpoint_path
    checkpoint = Path(str(configured)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Crafter MPC WM checkpoint not found: {checkpoint}")
    model, spec = load_crafter_planning_model(checkpoint)
    change_bias = getattr(mpc_cfg, "crafter_inventory_change_bias", None)
    if change_bias is not None:
        if not spec.inventory_event_residual_enabled:
            raise ValueError("Crafter inventory change bias requires an event-residual WM checkpoint")
        change_bias = float(change_bias)
        if not np.isfinite(change_bias):
            raise ValueError("PPO.mpc.crafter_inventory_change_bias must be finite")
        model.crafter_inventory_event_residual_change_bias = change_bias
    model.to(DEVICE).eval()
    planner = CrafterWorldModelMPC(model, spec, mpc_cfg)
    episodes, max_steps = int(mpc_cfg.episodes), int(cfg.PPO.max_ep_len)
    if episodes < 1 or max_steps < 1:
        raise ValueError("MPC episodes and max_ep_len must be positive")
    print_every = int(mpc_cfg.print_every_steps)
    if print_every < 1:
        raise ValueError("MPC print_every_steps must be positive")
    output_dir = Path(str(mpc_cfg.crafter_output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    step_path = output_dir / "wm_mpc_steps.csv"
    fields = (
        "episode", "seed", "environment_step", "plan_id", "plan_step",
        "replan_reason", "action", "action_name", "real_reward", "cumulative_real_reward",
        "plan_score", "planning_latency_ms", "imagined_transitions_this_plan",
        "imagined_transitions_total", "pose_match", "inventory_match",
        "survival_inventory_match", "item_inventory_match",
        "actual_item_delta", "predicted_item_delta",
        "newly_unlocked", "unique_visited_cells", "plan_exploration_score",
        "plan_tool_progress_score", "plan_late_progress_score",
        "plan_guided_attempt_score", "plan_guided_candidate_count",
        "plan_inventory_progress_stage", "plan_inventory_progress_score",
        "plan_tool_progress_start_total",
        "plan_tool_progress_prefix_total", "plan_tool_progress_end_total",
        "plan_tool_progress_start_table",
        "plan_tool_progress_start_wood_pickaxe", "plan_tool_progress_start_stone",
        "plan_tool_progress_start_stone_pickaxe", "plan_tool_progress_end_table",
        "plan_tool_progress_end_wood_pickaxe", "plan_tool_progress_end_stone",
        "plan_tool_progress_end_stone_pickaxe", "terminated", "truncated",
    )
    rows, traces = [], []
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    for episode in range(episodes):
        seed = int(cfg.PPO.seed) + episode
        env = CustomCrafterEnv(
            txt_file_path=str(cfg.PPO.env_path), max_steps=max_steps, seed=seed
        )
        try:
            observation, _ = env.reset()
            # CustomCrafterEnv builds a fresh World with RandomState(None).
            # Seed its objects here so MPC comparisons use the requested seed.
            world_rng = np.random.RandomState(seed)
            env.env._world.random = world_rng
            for entity in env.env._world._objects:
                if entity is not None:
                    entity.random = world_rng
            initial_objects = np.transpose(observation["image"], (2, 0, 1))[0]
            visited_positions = _crafter_initial_visit_map(initial_objects)
            inventory_progress_peaks = np.zeros((5, 16), dtype=np.float32)
            initial_stage = _crafter_inventory_progress_stage(
                initial_objects, np.asarray(observation["inventory"])
            )
            inventory_progress_peaks[initial_stage] = observation["inventory"]
            terminated = truncated = False
            episode_steps = 0
            real_return = 0.0
            plan_calls = early_replans = imagined_total = 0
            episode_trace = []
            unlocked = set()
            while not (terminated or truncated):
                state = torch.as_tensor(
                    np.transpose(observation["image"], (2, 0, 1)), device=DEVICE, dtype=torch.float32
                )
                inventory = torch.as_tensor(observation["inventory"], device=DEVICE)
                context = _crafter_context(env, state)
                started = time.perf_counter()
                plan = planner.plan(
                    state, inventory, context, visited_positions, inventory_progress_peaks
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                plan_calls += 1
                imagined_total += int(plan.diagnostics["imagined_transitions"])
                actions = plan.actions[:planner.execute_steps] or [planner.fallback_action]
                for index, action in enumerate(actions):
                    previous_inv = np.asarray(observation["inventory"])
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    real_return += float(reward)
                    new = [str(name) for name in info.get("newly_unlocked", [])]
                    unlocked.update(new)
                    actual_position = tuple(map(int, env.get_agent_position()))
                    if (0 <= actual_position[0] < visited_positions.shape[0]
                            and 0 <= actual_position[1] < visited_positions.shape[1]):
                        visited_positions[actual_position] = True
                    pose_match = actual_position == plan.predicted_positions[index]
                    actual_inv = np.asarray(observation["inventory"])
                    actual_objects = np.transpose(observation["image"], (2, 0, 1))[0]
                    actual_stage = _crafter_inventory_progress_stage(actual_objects, actual_inv)
                    inventory_progress_peaks[actual_stage] = np.maximum(
                        inventory_progress_peaks[actual_stage], actual_inv
                    )
                    predicted_inv = np.asarray(plan.predicted_inventories[index])
                    inv_match = bool(np.array_equal(actual_inv, predicted_inv))
                    survival_inv_match = bool(np.array_equal(
                        actual_inv[:SURVIVAL_SLOTS], predicted_inv[:SURVIVAL_SLOTS]
                    ))
                    item_inv_match = bool(np.array_equal(
                        actual_inv[SURVIVAL_SLOTS:], predicted_inv[SURVIVAL_SLOTS:]
                    ))
                    if terminated:
                        reason = "terminated"
                    elif truncated:
                        reason = "truncated"
                    elif not (pose_match and inv_match):
                        reason = "prediction_mismatch"
                        early_replans += 1
                    elif index == len(actions) - 1:
                        reason = "block_complete"
                    else:
                        reason = "block_continue"
                    row = {
                        "episode": episode, "seed": seed, "environment_step": episode_steps,
                        "plan_id": plan_calls, "plan_step": index + 1, "replan_reason": reason,
                        "action": action, "action_name": CRAFTER_ACTION_NAMES[action],
                        "real_reward": float(reward), "cumulative_real_reward": real_return,
                        "plan_score": plan.score,
                        "planning_latency_ms": latency_ms if index == 0 else 0.0,
                        "imagined_transitions_this_plan": int(plan.diagnostics["imagined_transitions"]) if index == 0 else 0,
                        "imagined_transitions_total": imagined_total, "pose_match": pose_match,
                        "inventory_match": inv_match,
                        "survival_inventory_match": survival_inv_match,
                        "item_inventory_match": item_inv_match,
                        "actual_item_delta": json.dumps((
                            actual_inv[SURVIVAL_SLOTS:] - previous_inv[SURVIVAL_SLOTS:]
                        ).astype(int).tolist()),
                        "predicted_item_delta": json.dumps((
                            predicted_inv[SURVIVAL_SLOTS:] - previous_inv[SURVIVAL_SLOTS:]
                        ).astype(int).tolist()),
                        "newly_unlocked": json.dumps(new),
                        "unique_visited_cells": int(visited_positions.sum()),
                        "plan_exploration_score": (
                            float(plan.diagnostics["exploration_score"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_score": (
                            float(plan.diagnostics["tool_progress_score"]) if index == 0 else 0.0
                        ),
                        "plan_late_progress_score": (
                            float(plan.diagnostics["late_progress_score"]) if index == 0 else 0.0
                        ),
                        "plan_guided_attempt_score": (
                            float(plan.diagnostics["guided_attempt_score"]) if index == 0 else 0.0
                        ),
                        "plan_guided_candidate_count": (
                            int(plan.diagnostics["guided_candidate_count"]) if index == 0 else 0
                        ),
                        "plan_inventory_progress_stage": int(plan.diagnostics["inventory_progress_stage"]),
                        "plan_inventory_progress_score": (
                            float(plan.diagnostics["inventory_progress_score"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_start_total": (
                            float(plan.diagnostics["tool_progress_start_total"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_prefix_total": (
                            float(plan.diagnostics["tool_progress_prefix_total"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_end_total": (
                            float(plan.diagnostics["tool_progress_end_total"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_start_table": (
                            float(plan.diagnostics["tool_progress_start_table"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_start_wood_pickaxe": (
                            float(plan.diagnostics["tool_progress_start_wood_pickaxe"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_start_stone": (
                            float(plan.diagnostics["tool_progress_start_stone"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_start_stone_pickaxe": (
                            float(plan.diagnostics["tool_progress_start_stone_pickaxe"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_end_table": (
                            float(plan.diagnostics["tool_progress_end_table"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_end_wood_pickaxe": (
                            float(plan.diagnostics["tool_progress_end_wood_pickaxe"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_end_stone": (
                            float(plan.diagnostics["tool_progress_end_stone"]) if index == 0 else 0.0
                        ),
                        "plan_tool_progress_end_stone_pickaxe": (
                            float(plan.diagnostics["tool_progress_end_stone_pickaxe"]) if index == 0 else 0.0
                        ),
                        "terminated": bool(terminated), "truncated": bool(truncated),
                    }
                    episode_trace.append(row)
                    with step_path.open("a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=fields).writerow(row)
                    if episode_steps % print_every == 0 or new or terminated or truncated:
                        print(
                            f"[Crafter MPC][ep={episode} step={episode_steps}] "
                            f"action={row['action_name']} reward={reward:.3f} "
                            f"return={real_return:.3f} achievements={len(unlocked)} "
                            f"new={new} reason={reason}"
                        )
                    if reason in {"terminated", "truncated", "prediction_mismatch"}:
                        break
            rows.append({
                "episode": episode, "seed": seed, "environment_steps": episode_steps,
                "real_reward": real_return, "unique_achievements": len(unlocked),
                "achievements": json.dumps(sorted(unlocked)), "plan_calls": plan_calls,
                "early_replans": early_replans, "imagined_transitions": imagined_total,
                "unique_visited_cells": int(visited_positions.sum()),
                "wm_real_inventory_match_rate": sum(step["inventory_match"] for step in episode_trace) / episode_steps,
                "wm_real_survival_inventory_match_rate": sum(step["survival_inventory_match"] for step in episode_trace) / episode_steps,
                "wm_real_item_inventory_match_rate": sum(step["item_inventory_match"] for step in episode_trace) / episode_steps,
                "inventory_change_bias": (
                    model.crafter_inventory_event_residual_change_bias
                    if spec.inventory_event_residual_enabled else None
                ),
                "tool_progress_weight": planner.tool_progress_weight,
                "inventory_progress_weight": planner.inventory_progress_weight,
                "require_three_wood_for_first_table": planner.require_three_wood_for_first_table,
                "late_stage_guidance": planner.late_stage_guidance,
                "wm_checkpoint": str(checkpoint),
            })
            traces.append({"episode": episode, "actions": episode_trace})
        finally:
            env.close()
    summary = output_dir / "wm_mpc_results.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "wm_mpc_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"[Crafter MPC] results={summary}; step curve={step_path}")
    return rows


def _minigrid_block_reason(
    *, terminated: bool, truncated: bool, pose_match: bool, inventory_match: bool,
    block_step: int, block_length: int, stop_on_prediction_mismatch: bool,
) -> str:
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    if stop_on_prediction_mismatch and not (pose_match and inventory_match):
        return "prediction_mismatch"
    if block_step == block_length:
        return "block_complete"
    return "block_continue"


def _synchronize_for_timing(enabled: bool) -> None:
    if enabled and DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_online_mpc(
    cfg: DictConfig, loaded_model: tuple[torch.nn.Module, Path] | None = None
) -> list[dict]:
    """Evaluate online MPC with bounded multi-step real-environment execution.

    Sweep workers may pass an already loaded, matching checkpoint to amortize
    Python/model startup across target and horizon conditions.
    """
    if str(cfg.domain).lower() == "crafter":
        return run_online_crafter_mpc(cfg)
    if bool(getattr(cfg.PPO, "train_in_real_env", False)):
        raise ValueError("MPC requires PPO.train_in_real_env=false so it loads a WM")
    mode = str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower()
    if mode != "mpc":
        raise ValueError(f"run_online_mpc requires PPO.wm_control_mode=mpc, got {mode!r}")
    configured_reward_mode = str(
        getattr(cfg.PPO.mpc, "minigrid_reward_mode", "native_goal")
    ).lower()
    if (
        configured_reward_mode == "legacy_dense"
        and not bool(getattr(cfg.PPO, "use_main_dense_reward", False))
    ):
        raise ValueError(
            "legacy_dense MPC scoring requires PPO.use_main_dense_reward=true"
        )
    if DEVICE.type == "cpu":
        cpu_threads = int(cfg.PPO.mpc.cpu_threads)
        if cpu_threads < 1:
            raise ValueError("PPO.mpc.cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    _seed_everything(int(cfg.PPO.seed))
    mpc_cfg = cfg.PPO.mpc
    profile_timing = bool(getattr(mpc_cfg, "profile_timing", False))
    if loaded_model is None:
        _synchronize_for_timing(profile_timing)
        load_started = time.perf_counter()
        model, checkpoint = _load_world_model(cfg)
        _synchronize_for_timing(profile_timing)
        model_load_seconds = time.perf_counter() - load_started if profile_timing else float("nan")
    else:
        model, checkpoint = loaded_model
        configured_checkpoint = getattr(cfg.PPO, "checkpoint_path_wm", None)
        if configured_checkpoint is not None and str(configured_checkpoint).strip().lower() not in {"", "null", "none"}:
            configured_checkpoint = Path(str(configured_checkpoint)).expanduser().resolve()
            if configured_checkpoint != Path(checkpoint).resolve():
                raise ValueError(
                    "Cached MiniGrid WM checkpoint does not match this sweep case: "
                    f"{configured_checkpoint} != {checkpoint}"
                )
        model_load_seconds = 0.0 if profile_timing else float("nan")
        model.eval()
    if profile_timing and DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    planner = WorldModelMPC(model, cfg.attention_model.attention_mask_size, mpc_cfg, cfg.PPO)
    stop_on_prediction_mismatch = bool(
        getattr(mpc_cfg, "minigrid_stop_on_prediction_mismatch", False)
    )
    dense_diagnostic_enabled = planner.reward_mode == "legacy_dense"
    episodes = int(mpc_cfg.episodes)
    max_steps = int(cfg.PPO.max_ep_len)
    output_dir = Path(str(mpc_cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print_every_steps = int(getattr(mpc_cfg, "print_every_steps", 1))
    if print_every_steps < 1:
        raise ValueError("PPO.mpc.print_every_steps must be at least 1")
    rows = []
    all_traces = []
    total_environment_steps = 0
    step_path = output_dir / "wm_mpc_steps.csv"
    hash_path = output_dir / "wm_mpc_action_hashes.csv"
    timing_path = output_dir / "wm_mpc_timing.csv"
    hash_rows = []
    timing_rows = []
    block_path = output_dir / "wm_mpc_blocks.csv"
    block_fields = (
        "episode", "seed", "plan_id", "start_environment_step", "end_environment_step",
        "executed_steps", "native_reward", "realized_goal_guide_bonus",
        "cumulative_real_reward", "cumulative_realized_goal_guide_bonus",
        "cumulative_real_reward_plus_guide", "start_goal_distance", "end_goal_distance",
        "terminated", "truncated",
    )
    with block_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=block_fields).writeheader()
    step_fields = (
        "episode", "seed", "environment_step", "total_environment_steps",
        "plan_id", "plan_step", "plan_execute_steps", "replan_reason",
        "action", "action_name", "real_reward", "cumulative_real_reward",
        "dense_reward", "cumulative_dense_reward", "plan_reward_mode",
        "plan_native_return", "plan_goal_guide_score", "plan_failure_penalty",
        "plan_legacy_dense_return",
        "plan_score", "planning_latency_ms", "imagined_transitions_this_plan",
        "imagined_transitions_total", "wm_uncertainty", "pose_match",
        "inventory_match", "terminated", "truncated",
    )
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=step_fields).writeheader()

    for episode in range(episodes):
        env = FullyObsWrapper(
            CustomMiniGridEnv(
                txt_file_path=str(cfg.PPO.env_path),
                custom_mission="Reach the goal.",
                max_steps=max_steps,
                render_mode=None,
            )
        )
        observation, _ = env.reset(seed=int(cfg.PPO.seed) + episode)
        initial_state = torch.as_tensor(
            _state_from_observation(observation), device=DEVICE
        )
        reward_context = DenseRewardContext(
            initial_state, _find_goal(initial_state.cpu().numpy()), planner.reward_settings
        )
        terminated = truncated = False
        reward_sum = 0.0
        realized_goal_guide_return = 0.0
        dense_reward_sum = 0.0 if dense_diagnostic_enabled else float("nan")
        plan_native_returns = []
        plan_goal_guide_scores = []
        plan_failure_penalties = []
        plan_legacy_dense_returns = []
        plan_scores = []
        plan_calls = 0
        imagined_transitions = 0
        planning_seconds = 0.0
        pose_matches = []
        inventory_matches = []
        action_trace = []
        episode_step_rows = []
        episode_block_rows = []
        episode_env_step_seconds = 0.0
        episode_log_seconds = 0.0
        episode_forward_decode_seconds = 0.0
        plan_id = 0
        replan_reason_counts = {}

        while not (terminated or truncated):
            state_np = _state_from_observation(observation)
            inventory = carrying_token_from_env(env)
            goal_yx = _find_goal(state_np)
            _synchronize_for_timing(profile_timing)
            started = time.perf_counter()
            plan = planner.plan(
                torch.as_tensor(state_np, device=DEVICE), inventory, goal_yx,
                reward_context, episode_step=len(action_trace),
                max_episode_steps=max_steps,
            )
            _synchronize_for_timing(profile_timing)
            planning_latency_seconds = time.perf_counter() - started
            planning_seconds += planning_latency_seconds
            episode_forward_decode_seconds += float(
                plan.diagnostics.get("model_forward_decode_seconds", 0.0)
            )
            plan_calls += 1
            plan_start_y, plan_start_x = minigrid_utils.get_agent_position(state_np)
            plan_start_distance = float(
                reward_context.goal_distance_map[plan_start_y, plan_start_x].item()
            )
            block_start_environment_step = len(action_trace) + 1
            block_native_reward = 0.0
            imagined_transitions += int(plan.diagnostics["imagined_transitions"])
            plan_native_returns.append(float(plan.diagnostics["native_return"]))
            plan_goal_guide_scores.append(float(plan.diagnostics["goal_guide_score"]))
            plan_failure_penalties.append(float(plan.diagnostics["failure_penalty"]))
            plan_legacy_dense_returns.append(float(plan.diagnostics["legacy_dense_return"]))
            plan_scores.append(float(plan.score))
            plan_id += 1
            if planner.capture_action_hashes:
                for hash_kind, iteration, digest in planner.action_hashes:
                    hash_rows.append({
                        "episode": episode, "seed": int(cfg.PPO.seed) + episode,
                        "plan_id": plan_id, "hash_kind": hash_kind,
                        "cem_iteration": iteration, "sha256": digest,
                    })
            actions = plan.actions[: planner.execute_steps]
            if not actions:
                actions = [planner.fallback_action]
            block_length = len(actions)
            block_reason = "block_complete"
            for block_step, action in enumerate(actions, start=1):
                predicted_position = plan.predicted_positions[block_step - 1]
                predicted_inventory = plan.predicted_inventories[block_step - 1]
                previous_real_state = torch.as_tensor(
                    _state_from_observation(observation), device=DEVICE
                )
                previous_real_inventory = carrying_token_from_env(env)
                step_started = time.perf_counter() if profile_timing else 0.0
                observation, reward, terminated, truncated, _ = env.step(
                    compact_to_native(action)
                )
                if profile_timing:
                    episode_env_step_seconds += time.perf_counter() - step_started
                reward_sum += float(reward)
                block_native_reward += float(reward)
                total_environment_steps += 1
                real_next = _state_from_observation(observation)
                real_position = minigrid_utils.get_agent_position(real_next)
                real_inventory = carrying_token_from_env(env)
                real_success = bool(terminated and reward > 0.0)
                if dense_diagnostic_enabled:
                    dense_reward = reward_context.step(
                        previous_real_state.unsqueeze(0),
                        torch.as_tensor(real_next, device=DEVICE).unsqueeze(0),
                        torch.as_tensor([action], device=DEVICE),
                        torch.as_tensor([previous_real_inventory], device=DEVICE),
                        torch.as_tensor([real_inventory], device=DEVICE),
                        torch.as_tensor([real_success], device=DEVICE),
                        torch.as_tensor([terminated and not real_success], device=DEVICE),
                    )
                    dense_reward_value = float(dense_reward.item())
                    dense_reward_sum += dense_reward_value
                else:
                    dense_reward_value = float("nan")
                pose_match = tuple(map(int, real_position)) == tuple(predicted_position)
                inventory_match = predicted_inventory == real_inventory
                pose_matches.append(pose_match)
                inventory_matches.append(inventory_match)
                block_reason = _minigrid_block_reason(
                    terminated=terminated, truncated=truncated,
                    pose_match=pose_match, inventory_match=inventory_match,
                    block_step=block_step, block_length=block_length,
                    stop_on_prediction_mismatch=stop_on_prediction_mismatch,
                )
                log_started = time.perf_counter() if profile_timing else 0.0
                step_row = {
                    "episode": episode,
                    "seed": int(cfg.PPO.seed) + episode,
                    "environment_step": len(action_trace) + 1,
                    "total_environment_steps": total_environment_steps,
                    "plan_id": plan_id,
                    "plan_step": block_step,
                    "plan_execute_steps": planner.execute_steps,
                    "replan_reason": block_reason,
                    "action": int(action),
                    "action_name": COMPACT_ACTION_NAMES[action],
                    "real_reward": float(reward),
                    "cumulative_real_reward": reward_sum,
                    "dense_reward": dense_reward_value,
                    "cumulative_dense_reward": dense_reward_sum,
                    "plan_reward_mode": planner.reward_mode,
                    "plan_native_return": (
                        float(plan.diagnostics["native_return"]) if block_step == 1 else 0.0
                    ),
                    "plan_goal_guide_score": (
                        float(plan.diagnostics["goal_guide_score"]) if block_step == 1 else 0.0
                    ),
                    "plan_failure_penalty": (
                        float(plan.diagnostics["failure_penalty"]) if block_step == 1 else 0.0
                    ),
                    "plan_legacy_dense_return": (
                        float(plan.diagnostics["legacy_dense_return"]) if block_step == 1 else 0.0
                    ),
                    "plan_score": plan.score,
                    "planning_latency_ms": 1000.0 * planning_latency_seconds if block_step == 1 else 0.0,
                    "imagined_transitions_this_plan": int(plan.diagnostics["imagined_transitions"]) if block_step == 1 else 0,
                    "imagined_transitions_total": imagined_transitions,
                    "wm_uncertainty": plan.diagnostics["mean_rollout_uncertainty"],
                    "pose_match": pose_match,
                    "inventory_match": inventory_match,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                action_trace.append(step_row)
                episode_step_rows.append(step_row)
                if total_environment_steps % print_every_steps == 0 or terminated or truncated:
                    dense_log = (
                        f"dense_diag={dense_reward_value:.3f} "
                        f"dense_diag_return={dense_reward_sum:.3f} "
                        if dense_diagnostic_enabled else ""
                    )
                    print(
                        f"[WM MPC][ep={episode} step={step_row['environment_step']} "
                        f"plan={plan_id}:{block_step}/{block_length} total_env={total_environment_steps}] "
                        f"action={step_row['action_name']} native_reward={float(reward):.3f} "
                        f"{dense_log}"
                        f"imagined_native={plan.diagnostics['native_return']:.3f} "
                        f"goal_guide={plan.diagnostics['goal_guide_score']:.3f} "
                        f"failure_penalty={plan.diagnostics['failure_penalty']:.3f} "
                        f"score={plan.score:.3f} "
                        f"reason={block_reason} latency={step_row['planning_latency_ms']:.1f}ms"
                    )
                if profile_timing:
                    episode_log_seconds += time.perf_counter() - log_started
                if terminated or truncated or (
                    stop_on_prediction_mismatch and block_reason == "prediction_mismatch"
                ):
                    break
            end_state_np = _state_from_observation(observation)
            end_y, end_x = minigrid_utils.get_agent_position(end_state_np)
            end_distance = float(reward_context.goal_distance_map[end_y, end_x].item())
            block_executed_steps = len(action_trace) - block_start_environment_step + 1
            failed = bool(truncated or (terminated and not (reward > 0.0)))
            realized_guide_bonus = _realized_goal_guide_bonus(
                plan_start_distance,
                end_distance,
                planner.goal_guide_weight,
                planner.gamma,
                block_executed_steps,
                failed,
            )
            realized_goal_guide_return += realized_guide_bonus
            block_row = {
                "episode": episode,
                "seed": int(cfg.PPO.seed) + episode,
                "plan_id": plan_id,
                "start_environment_step": block_start_environment_step,
                "end_environment_step": len(action_trace),
                "executed_steps": block_executed_steps,
                "native_reward": block_native_reward,
                "realized_goal_guide_bonus": realized_guide_bonus,
                "cumulative_real_reward": reward_sum,
                "cumulative_realized_goal_guide_bonus": realized_goal_guide_return,
                "cumulative_real_reward_plus_guide": reward_sum + realized_goal_guide_return,
                "start_goal_distance": plan_start_distance,
                "end_goal_distance": end_distance,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            episode_block_rows.append(block_row)
            replan_reason_counts[block_reason] = replan_reason_counts.get(block_reason, 0) + 1

        log_started = time.perf_counter() if profile_timing else 0.0
        with step_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=step_fields).writerows(episode_step_rows)
        with block_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=block_fields).writerows(episode_block_rows)
        if profile_timing:
            episode_log_seconds += time.perf_counter() - log_started
            timing_rows.append({
                "episode": episode, "seed": int(cfg.PPO.seed) + episode,
                "model_load_seconds": model_load_seconds,
                "planning_seconds": planning_seconds,
                "model_forward_decode_seconds": episode_forward_decode_seconds,
                "other_planning_seconds": max(
                    0.0, planning_seconds - episode_forward_decode_seconds
                ),
                "env_step_seconds": episode_env_step_seconds,
                "logging_seconds": episode_log_seconds,
                "plan_calls": plan_calls, "environment_steps": len(action_trace),
                "cuda_peak_memory_bytes": (
                    torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else 0
                ),
            })
        rows.append(
            {
                "episode": episode,
                "seed": int(cfg.PPO.seed) + episode,
                "success": bool(terminated and reward_sum > 0.0),
                "environment_steps": len(action_trace),
                "real_reward": reward_sum,
                "realized_goal_guide_return": realized_goal_guide_return,
                "real_reward_plus_realized_goal_guide": reward_sum + realized_goal_guide_return,
                "dense_return": dense_reward_sum,
                "reward_mode": planner.reward_mode,
                "mean_plan_score": float(np.mean(plan_scores)) if plan_scores else 0.0,
                "mean_plan_native_return": (
                    float(np.mean(plan_native_returns)) if plan_native_returns else 0.0
                ),
                "mean_plan_goal_guide_score": (
                    float(np.mean(plan_goal_guide_scores)) if plan_goal_guide_scores else 0.0
                ),
                "mean_plan_failure_penalty": (
                    float(np.mean(plan_failure_penalties)) if plan_failure_penalties else 0.0
                ),
                "mean_plan_legacy_dense_return": (
                    float(np.mean(plan_legacy_dense_returns))
                    if plan_legacy_dense_returns else 0.0
                ),
                "plan_calls": plan_calls,
                "early_replans": replan_reason_counts.get("prediction_mismatch", 0),
                "mean_executed_steps_per_plan": len(action_trace) / max(plan_calls, 1),
                "imagined_transitions": imagined_transitions,
                "mean_planning_latency_ms": 1000.0 * planning_seconds / max(plan_calls, 1),
                "wm_real_pose_match_rate": float(np.mean(pose_matches)) if pose_matches else 0.0,
                "wm_real_inventory_match_rate": float(np.mean(inventory_matches)) if inventory_matches else 0.0,
                "wm_checkpoint": str(checkpoint),
            }
        )
        all_traces.append({"episode": episode, "actions": action_trace})
        env.close()

    if planner.capture_action_hashes:
        with hash_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=(
                    "episode", "seed", "plan_id", "hash_kind", "cem_iteration", "sha256"
                )
            )
            writer.writeheader()
            writer.writerows(hash_rows)
    if profile_timing:
        with timing_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
    summary_path = output_dir / "wm_mpc_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "wm_mpc_traces.json").open("w", encoding="utf-8") as handle:
        json.dump(all_traces, handle, indent=2)
    dense_summary = (
        f"; mean_legacy_dense_diagnostic={np.mean([row['dense_return'] for row in rows]):.3f}"
        if dense_diagnostic_enabled else ""
    )
    print(
        f"[WM MPC] success={np.mean([row['success'] for row in rows]):.1%}; "
        f"environment_steps={sum(row['environment_steps'] for row in rows)}"
        f"{dense_summary}; imagined_transitions={sum(row['imagined_transitions'] for row in rows)}"
    )
    print(f"[WM MPC] results={summary_path}")
    print(f"[WM MPC] step curve={step_path}")
    print(f"[WM MPC] realized reward blocks={block_path}")
    return rows


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    run_online_mpc(cfg)


if __name__ == "__main__":
    main()
