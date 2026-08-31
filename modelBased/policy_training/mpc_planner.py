"""Online MiniGrid MPC: search only in the learned world model.

Each control cycle evaluates action sequences in the WM, executes exactly the
chosen first action in the real environment, then replans from the new real
observation.  Imagined transitions are logged separately from ``env.step``.
"""

from __future__ import annotations

import csv
import json
import random
import time
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
from modelBased.policy_training.minigrid_wm_rollout import rollout_minigrid_wm
from modelBased.world_model import AttentionWM_support


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


def _goal_distance_map(state: torch.Tensor, goal_yx: tuple[int, int]) -> torch.Tensor:
    """Static optimistic distance map. Only walls and lava are impassable."""
    objects = state[0].detach().cpu().numpy()
    rows, columns = objects.shape
    distances = np.full((rows, columns), -1, dtype=np.int64)
    goal_y, goal_x = map(int, goal_yx)
    distances[goal_y, goal_x] = 0
    queue = [(goal_y, goal_x)]
    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < rows and 0 <= nx < columns):
                continue
            if distances[ny, nx] >= 0 or int(objects[ny, nx]) in (2, 9):
                continue
            distances[ny, nx] = distances[y, x] + 1
            queue.append((ny, nx))
    return torch.as_tensor(distances, device=state.device, dtype=torch.long)


@dataclass
class MPCPlan:
    actions: list[int]
    score: float
    predicted_next_state: torch.Tensor
    predicted_next_inventory: int
    diagnostics: dict[str, float | int]


class WorldModelMPC:
    """Random-shooting MPC with elite categorical resampling for MiniGrid."""

    def __init__(self, model, attention_mask_size: int, config):
        self.model = model
        self.attention_mask_size = int(attention_mask_size)
        self.horizon = int(config.horizon)
        self.population = int(config.population)
        self.elite_count = int(config.elite_count)
        self.iterations = int(config.iterations)
        self.gamma = float(config.gamma)
        self.goal_reward = float(config.goal_reward)
        self.progress_reward = float(config.progress_reward)
        self.step_penalty = float(config.step_penalty)
        self.invalid_action_penalty = float(config.invalid_action_penalty)
        self.uncertainty_penalty = float(config.uncertainty_penalty)
        self.fallback_action = int(config.fallback_action)
        if self.horizon < 1 or self.population < 1 or self.iterations < 1:
            raise ValueError("MPC horizon, population, and iterations must be positive")
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
        distance_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        states = start_state.unsqueeze(0).expand(self.population, -1, -1, -1).clone()
        inventories = torch.full(
            (self.population,), int(start_inventory), device=DEVICE, dtype=torch.long
        )
        scores = torch.zeros(self.population, device=DEVICE)
        reached = torch.zeros(self.population, device=DEVICE, dtype=torch.bool)
        first_states = None
        first_inventories = None
        uncertainty_sum = 0.0
        invalid_sum = 0
        for step in range(self.horizon):
            actions = sequences[:, step]
            before_positions = utils.get_agent_position_torch(states)
            before_distances = distance_map[
                before_positions[:, 0].long(), before_positions[:, 1].long()
            ]
            previous_states = states
            previous_inventories = inventories
            states, inventories, transition = rollout_minigrid_wm(
                self.model, states, actions, inventories, self.attention_mask_size
            )
            if step == 0:
                first_states = states.clone()
                first_inventories = inventories.clone()
            after_positions = transition["agent_positions_after"]
            after_distances = distance_map[
                after_positions[:, 0].long(), after_positions[:, 1].long()
            ]
            valid_distance = (before_distances >= 0) & (after_distances >= 0)
            progress = torch.where(
                valid_distance,
                (before_distances - after_distances).float(),
                torch.zeros_like(before_distances, dtype=torch.float32),
            )
            at_goal = (after_positions[:, 0] == goal_yx[0]) & (
                after_positions[:, 1] == goal_yx[1]
            )
            new_goal = at_goal & ~reached
            reached |= at_goal
            unchanged = states.eq(previous_states).flatten(1).all(dim=1) & inventories.eq(previous_inventories)
            invalid = unchanged & (actions >= 2)
            discount = self.gamma ** step
            scores += discount * (
                self.progress_reward * progress
                + self.goal_reward * new_goal.float()
                - self.step_penalty
                - self.invalid_action_penalty * invalid.float()
                - self.uncertainty_penalty * transition["uncertainty"]
            )
            uncertainty_sum += float(transition["uncertainty"].sum().item())
            invalid_sum += int(invalid.sum().item())
        diagnostics = {
            "mean_rollout_uncertainty": uncertainty_sum / (self.population * self.horizon),
            "invalid_imagined_actions": invalid_sum,
            "imagined_transitions": self.population * self.horizon,
        }
        return scores, first_states, first_inventories, diagnostics

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        inventory_token: int,
        goal_yx: tuple[int, int],
    ) -> MPCPlan:
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        if state.ndim != 3:
            raise ValueError(f"Expected state [3,H,W], got {tuple(state.shape)}")
        distance_map = _goal_distance_map(state, goal_yx)
        probabilities = None
        best = None
        total_imagined = 0
        total_invalid = 0
        uncertainty_values = []
        for _ in range(self.iterations):
            sequences = self._sample(probabilities)
            scores, first_states, first_inventories, diagnostics = self._evaluate(
                state, inventory_token, sequences, goal_yx, distance_map
            )
            total_imagined += int(diagnostics["imagined_transitions"])
            total_invalid += int(diagnostics["invalid_imagined_actions"])
            uncertainty_values.append(float(diagnostics["mean_rollout_uncertainty"]))
            best_index = int(scores.argmax().item())
            if best is None or scores[best_index] > best[0]:
                best = (
                    scores[best_index].detach().clone(),
                    sequences[best_index].detach().clone(),
                    first_states[best_index].detach().clone(),
                    int(first_inventories[best_index].item()),
                )
            elite_indices = scores.topk(self.elite_count).indices
            elite_actions = sequences[elite_indices]
            counts = torch.nn.functional.one_hot(
                elite_actions, num_classes=MODEL_ACTION_COUNT
            ).float().sum(dim=0)
            probabilities = (counts + 1.0) / (self.elite_count + MODEL_ACTION_COUNT)

        assert best is not None
        score, actions, first_state, first_inventory = best
        return MPCPlan(
            actions=[int(action) for action in actions.detach().cpu().tolist()],
            score=float(score.item()),
            predicted_next_state=first_state,
            predicted_next_inventory=first_inventory,
            diagnostics={
                "iterations": self.iterations,
                "population": self.population,
                "horizon": self.horizon,
                "imagined_transitions": total_imagined,
                "invalid_imagined_actions": total_invalid,
                "mean_rollout_uncertainty": float(np.mean(uncertainty_values)),
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


def run_online_mpc(cfg: DictConfig) -> list[dict]:
    """Evaluate online MPC; only the selected first action reaches ``env.step``."""
    if bool(getattr(cfg.PPO, "train_in_real_env", False)):
        raise ValueError("MPC requires PPO.train_in_real_env=false so it loads a WM")
    mode = str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower()
    if mode != "mpc":
        raise ValueError(f"run_online_mpc requires PPO.wm_control_mode=mpc, got {mode!r}")
    _seed_everything(int(cfg.PPO.seed))
    model, checkpoint = _load_world_model(cfg)
    mpc_cfg = cfg.PPO.mpc
    planner = WorldModelMPC(model, cfg.attention_model.attention_mask_size, mpc_cfg)
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
    step_fields = (
        "episode", "seed", "environment_step", "total_environment_steps",
        "action", "action_name", "real_reward", "cumulative_real_reward",
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
        terminated = truncated = False
        reward_sum = 0.0
        plan_calls = 0
        imagined_transitions = 0
        planning_seconds = 0.0
        pose_matches = []
        inventory_matches = []
        action_trace = []

        while not (terminated or truncated):
            state_np = _state_from_observation(observation)
            inventory = carrying_token_from_env(env)
            goal_yx = _find_goal(state_np)
            started = time.perf_counter()
            plan = planner.plan(
                torch.as_tensor(state_np, device=DEVICE), inventory, goal_yx
            )
            planning_latency_seconds = time.perf_counter() - started
            planning_seconds += planning_latency_seconds
            plan_calls += 1
            imagined_transitions += int(plan.diagnostics["imagined_transitions"])
            action = plan.actions[0] if plan.actions else planner.fallback_action

            # This is the sole real-environment transition in a control cycle.
            observation, reward, terminated, truncated, _ = env.step(compact_to_native(action))
            reward_sum += float(reward)
            total_environment_steps += 1
            real_next = _state_from_observation(observation)
            predicted_position = minigrid_utils.get_agent_position(
                plan.predicted_next_state.detach().cpu().numpy()
            )
            real_position = minigrid_utils.get_agent_position(real_next)
            real_inventory = carrying_token_from_env(env)
            pose_matches.append(predicted_position == real_position)
            inventory_matches.append(plan.predicted_next_inventory == real_inventory)
            step_row = {
                "episode": episode,
                "seed": int(cfg.PPO.seed) + episode,
                "environment_step": len(action_trace) + 1,
                "total_environment_steps": total_environment_steps,
                "action": int(action),
                "action_name": COMPACT_ACTION_NAMES[action],
                "real_reward": float(reward),
                "cumulative_real_reward": reward_sum,
                "plan_score": plan.score,
                "planning_latency_ms": 1000.0 * planning_latency_seconds,
                "imagined_transitions_this_plan": int(plan.diagnostics["imagined_transitions"]),
                "imagined_transitions_total": imagined_transitions,
                "wm_uncertainty": plan.diagnostics["mean_rollout_uncertainty"],
                "pose_match": pose_matches[-1],
                "inventory_match": inventory_matches[-1],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            action_trace.append(step_row)
            with step_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=step_fields)
                writer.writerow(step_row)
                handle.flush()
            if total_environment_steps % print_every_steps == 0 or terminated or truncated:
                print(
                    f"[WM MPC][ep={episode} step={step_row['environment_step']} "
                    f"total_env={total_environment_steps}] "
                    f"action={step_row['action_name']} reward={float(reward):.3f} "
                    f"return={reward_sum:.3f} score={plan.score:.3f} "
                    f"wm_steps={step_row['imagined_transitions_this_plan']} "
                    f"latency={step_row['planning_latency_ms']:.1f}ms"
                )

        rows.append(
            {
                "episode": episode,
                "seed": int(cfg.PPO.seed) + episode,
                "success": bool(terminated and reward_sum > 0.0),
                "environment_steps": len(action_trace),
                "real_reward": reward_sum,
                "plan_calls": plan_calls,
                "imagined_transitions": imagined_transitions,
                "mean_planning_latency_ms": 1000.0 * planning_seconds / max(plan_calls, 1),
                "wm_real_pose_match_rate": float(np.mean(pose_matches)) if pose_matches else 0.0,
                "wm_real_inventory_match_rate": float(np.mean(inventory_matches)) if inventory_matches else 0.0,
                "wm_checkpoint": str(checkpoint),
            }
        )
        all_traces.append({"episode": episode, "actions": action_trace})
        env.close()

    summary_path = output_dir / "wm_mpc_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "wm_mpc_traces.json").open("w", encoding="utf-8") as handle:
        json.dump(all_traces, handle, indent=2)
    print(
        f"[WM MPC] success={np.mean([row['success'] for row in rows]):.1%}; "
        f"environment_steps={sum(row['environment_steps'] for row in rows)}; "
        f"imagined_transitions={sum(row['imagined_transitions'] for row in rows)}"
    )
    print(f"[WM MPC] results={summary_path}")
    print(f"[WM MPC] step curve={step_path}")
    return rows


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_online_mpc(cfg)


if __name__ == "__main__":
    main()
