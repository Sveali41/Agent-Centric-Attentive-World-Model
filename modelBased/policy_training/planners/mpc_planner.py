"""Online MiniGrid MPC: search only in the learned world model.

Each control cycle evaluates action sequences in the WM, executes a bounded
prefix in the real environment, then replans from the new real observation.
The prefix is cut short when the real state disagrees with the WM prediction.
Imagined transitions are logged separately from ``env.step``.
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
from modelBased.policy_training.common.minigrid_dense_reward import (
    build_door_topology,
    build_goal_distance_map,
    build_goal_region_mask,
    main_dense_rewards,
    reward_settings,
)
from modelBased.policy_training.common.minigrid_wm_rollout import rollout_minigrid_wm
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
    predicted_inventories: list[int]
    diagnostics: dict[str, float | int]


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
        self.reward_settings = reward_settings(ppo_config)
        self.fallback_action = int(config.fallback_action)
        self.execute_steps = int(getattr(config, "execute_steps", 8))
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
    ]:
        states = start_state.unsqueeze(0).expand(self.population, -1, -1, -1).clone()
        inventories = torch.full(
            (self.population,), int(start_inventory), device=DEVICE, dtype=torch.long
        )
        scores = torch.zeros(self.population, device=DEVICE)
        finished = torch.zeros(self.population, device=DEVICE, dtype=torch.bool)
        imagined_reward = reward_context.for_population(self.population)
        predicted_positions = torch.empty(
            (self.population, self.horizon, 2), device=DEVICE, dtype=torch.long
        )
        predicted_inventories = torch.empty(
            (self.population, self.horizon), device=DEVICE, dtype=torch.long
        )
        uncertainty_sum = 0.0
        invalid_sum = 0
        for step in range(self.horizon):
            actions = sequences[:, step]
            previous_states = states
            previous_inventories = inventories
            before_positions = utils.get_agent_position_torch(previous_states)
            directions = previous_states[
                torch.arange(self.population, device=DEVICE),
                2,
                before_positions[:, 0],
                before_positions[:, 1],
            ].long()
            deltas = torch.as_tensor(
                ((0, 1), (1, 0), (0, -1), (-1, 0)), device=DEVICE
            )
            front = before_positions + deltas[directions.clamp(0, 3)]
            height, width = previous_states.shape[-2:]
            valid_front = (
                directions.ge(0) & directions.lt(4)
                & front[:, 0].ge(0) & front[:, 0].lt(height)
                & front[:, 1].ge(0) & front[:, 1].lt(width)
            )
            front_objects = previous_states[
                torch.arange(self.population, device=DEVICE),
                0,
                front[:, 0].clamp(0, height - 1),
                front[:, 1].clamp(0, width - 1),
            ]
            lava_terminated = (
                ~finished & actions.eq(2) & valid_front & front_objects.eq(9)
            )
            states, inventories, transition = rollout_minigrid_wm(
                self.model, states, actions, inventories, self.attention_mask_size
            )
            after_positions = transition["agent_positions_after"]
            predicted_positions[:, step] = after_positions
            predicted_inventories[:, step] = inventories
            at_goal = (after_positions[:, 0] == goal_yx[0]) & (
                after_positions[:, 1] == goal_yx[1]
            )
            new_goal = at_goal & ~finished & ~lava_terminated
            unchanged = states.eq(previous_states).flatten(1).all(dim=1) & inventories.eq(previous_inventories)
            invalid = unchanged & (actions >= 2)
            discount = self.gamma ** step
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
            uncertainty_sum += float(transition["uncertainty"].sum().item())
            invalid_sum += int(invalid.sum().item())
        diagnostics = {
            "mean_rollout_uncertainty": uncertainty_sum / (self.population * self.horizon),
            "invalid_imagined_actions": invalid_sum,
            "imagined_transitions": self.population * self.horizon,
        }
        return (
            scores,
            predicted_positions,
            predicted_inventories,
            diagnostics,
        )

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        inventory_token: int,
        goal_yx: tuple[int, int],
        reward_context: DenseRewardContext,
    ) -> MPCPlan:
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        if state.ndim != 3:
            raise ValueError(f"Expected state [3,H,W], got {tuple(state.shape)}")
        probabilities = None
        best = None
        total_imagined = 0
        total_invalid = 0
        uncertainty_values = []
        for _ in range(self.iterations):
            sequences = self._sample(probabilities)
            (
                scores,
                predicted_positions,
                predicted_inventories,
                diagnostics,
            ) = self._evaluate(state, inventory_token, sequences, goal_yx, reward_context)
            total_imagined += int(diagnostics["imagined_transitions"])
            total_invalid += int(diagnostics["invalid_imagined_actions"])
            uncertainty_values.append(float(diagnostics["mean_rollout_uncertainty"]))
            best_index = int(scores.argmax().item())
            if best is None or scores[best_index] > best[0]:
                best = (
                    scores[best_index].detach().clone(),
                    sequences[best_index].detach().clone(),
                    predicted_positions[best_index].detach().clone(),
                    predicted_inventories[best_index].detach().clone(),
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
        ) = best
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
    """Evaluate online MPC with bounded multi-step real-environment execution."""
    if bool(getattr(cfg.PPO, "train_in_real_env", False)):
        raise ValueError("MPC requires PPO.train_in_real_env=false so it loads a WM")
    mode = str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower()
    if mode != "mpc":
        raise ValueError(f"run_online_mpc requires PPO.wm_control_mode=mpc, got {mode!r}")
    if not bool(getattr(cfg.PPO, "use_main_dense_reward", False)):
        raise ValueError("MPC dense scoring requires PPO.use_main_dense_reward=true")
    if DEVICE.type == "cpu":
        cpu_threads = int(cfg.PPO.mpc.cpu_threads)
        if cpu_threads < 1:
            raise ValueError("PPO.mpc.cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    _seed_everything(int(cfg.PPO.seed))
    model, checkpoint = _load_world_model(cfg)
    mpc_cfg = cfg.PPO.mpc
    planner = WorldModelMPC(model, cfg.attention_model.attention_mask_size, mpc_cfg, cfg.PPO)
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
        "plan_id", "plan_step", "plan_execute_steps", "replan_reason",
        "action", "action_name", "real_reward", "cumulative_real_reward",
        "dense_reward", "cumulative_dense_reward",
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
        dense_reward_sum = 0.0
        plan_calls = 0
        imagined_transitions = 0
        planning_seconds = 0.0
        pose_matches = []
        inventory_matches = []
        action_trace = []
        plan_id = 0
        replan_reason_counts = {}

        while not (terminated or truncated):
            state_np = _state_from_observation(observation)
            inventory = carrying_token_from_env(env)
            goal_yx = _find_goal(state_np)
            started = time.perf_counter()
            plan = planner.plan(
                torch.as_tensor(state_np, device=DEVICE), inventory, goal_yx,
                reward_context,
            )
            planning_latency_seconds = time.perf_counter() - started
            planning_seconds += planning_latency_seconds
            plan_calls += 1
            imagined_transitions += int(plan.diagnostics["imagined_transitions"])
            plan_id += 1
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
                observation, reward, terminated, truncated, _ = env.step(
                    compact_to_native(action)
                )
                reward_sum += float(reward)
                total_environment_steps += 1
                real_next = _state_from_observation(observation)
                real_position = minigrid_utils.get_agent_position(real_next)
                real_inventory = carrying_token_from_env(env)
                real_success = bool(terminated and reward > 0.0)
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
                pose_match = tuple(map(int, real_position)) == tuple(predicted_position)
                inventory_match = predicted_inventory == real_inventory
                pose_matches.append(pose_match)
                inventory_matches.append(inventory_match)
                if terminated:
                    block_reason = "terminated"
                elif truncated:
                    block_reason = "truncated"
                elif not (pose_match and inventory_match):
                    block_reason = "prediction_mismatch"
                elif block_step == block_length:
                    block_reason = "block_complete"
                else:
                    block_reason = "block_continue"
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
                with step_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=step_fields)
                    writer.writerow(step_row)
                    handle.flush()
                if total_environment_steps % print_every_steps == 0 or terminated or truncated:
                    print(
                        f"[WM MPC][ep={episode} step={step_row['environment_step']} "
                        f"plan={plan_id}:{block_step}/{block_length} total_env={total_environment_steps}] "
                        f"action={step_row['action_name']} native_reward={float(reward):.3f} "
                        f"dense_reward={dense_reward_value:.3f} dense_return={dense_reward_sum:.3f} "
                        f"score={plan.score:.3f} "
                        f"reason={block_reason} latency={step_row['planning_latency_ms']:.1f}ms"
                    )
                if terminated or truncated or block_reason == "prediction_mismatch":
                    break
            replan_reason_counts[block_reason] = replan_reason_counts.get(block_reason, 0) + 1

        rows.append(
            {
                "episode": episode,
                "seed": int(cfg.PPO.seed) + episode,
                "success": bool(terminated and reward_sum > 0.0),
                "environment_steps": len(action_trace),
                "real_reward": reward_sum,
                "dense_return": dense_reward_sum,
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
        f"mean_dense_return={np.mean([row['dense_return'] for row in rows]):.3f}; "
        f"imagined_transitions={sum(row['imagined_transitions'] for row in rows)}"
    )
    print(f"[WM MPC] results={summary_path}")
    print(f"[WM MPC] step curve={step_path}")
    return rows


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    run_online_mpc(cfg)


if __name__ == "__main__":
    main()
