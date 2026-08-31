"""Online UCT-MCTS for the learned symbolic MiniGrid world model.

Tree selection, expansion, and rollouts use only ``rollout_minigrid_wm``.
The outer control loop executes precisely one selected root action in the
real environment, then discards the old tree and plans from the new observation.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper
from omegaconf import DictConfig

from domain.minigrid import minigrid_support as minigrid_utils
from domain.minigrid.action_codec import (
    COMPACT_ACTION_NAMES,
    MODEL_ACTION_COUNT,
    carrying_token_from_env,
    compact_to_native,
)
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from modelBased.common import utils
from modelBased.policy_training.minigrid_wm_rollout import rollout_minigrid_wm
from modelBased.policy_training.mpc_planner import (
    DEVICE,
    _find_goal,
    _load_world_model,
    _seed_everything,
    _state_from_observation,
)


@dataclass
class MCTSPlan:
    actions: list[int]
    score: float
    predicted_next_state: torch.Tensor
    predicted_next_inventory: int
    diagnostics: dict[str, float | int]


@dataclass
class _Node:
    state: torch.Tensor
    inventory: int
    parent: "_Node | None" = None
    action: int | None = None
    immediate_reward: float = 0.0
    depth: int = 0
    terminal: bool = False
    children: dict[int, "_Node"] = field(default_factory=dict)
    unexpanded_actions: set[int] = field(
        default_factory=lambda: set(range(MODEL_ACTION_COUNT))
    )
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class WorldModelMCTS:
    """UCT search using MiniGrid's native sparse reward over WM states."""

    def __init__(self, model, attention_mask_size: int, config):
        self.model = model
        self.attention_mask_size = int(attention_mask_size)
        self.simulations = int(config.simulations)
        self.max_depth = int(config.max_depth)
        self.rollout_depth = int(config.rollout_depth)
        self.exploration_constant = float(config.exploration_constant)
        self.gamma = float(config.gamma)
        self.fallback_action = int(config.fallback_action)
        if self.simulations < 1 or self.max_depth < 1 or self.rollout_depth < 0:
            raise ValueError("MCTS simulations/max_depth must be positive and rollout_depth non-negative")
        if self.exploration_constant < 0.0:
            raise ValueError("MCTS exploration_constant must be non-negative")
        if not 0 <= self.fallback_action < MODEL_ACTION_COUNT:
            raise ValueError("MCTS fallback_action must be a compact MiniGrid action")
        self._imagined_transitions = 0
        self._uncertainty_sum = 0.0
        self._invalid_actions = 0
        self._tree_nodes = 0

    def _position(self, state: torch.Tensor) -> tuple[int, int]:
        position = utils.get_agent_position_torch(state)
        return tuple(map(int, position.detach().cpu().tolist()))

    @staticmethod
    def _forward_enters_lava(state: torch.Tensor, action: int) -> bool:
        """Mirror MiniGrid's terminal lava transition from the decoded state."""
        if action != 2:  # compact MiniGrid ``forward``
            return False
        position = utils.get_agent_position_torch(state)
        row, column = map(int, position.detach().cpu().tolist())
        direction = int(state[2, row, column].item())
        offsets = ((0, 1), (1, 0), (0, -1), (-1, 0))
        if direction not in range(len(offsets)):
            return False
        delta_row, delta_column = offsets[direction]
        next_row, next_column = row + delta_row, column + delta_column
        if not (0 <= next_row < state.shape[1] and 0 <= next_column < state.shape[2]):
            return False
        return int(state[0, next_row, next_column].item()) == OBJECT_TO_IDX["lava"]

    @staticmethod
    def _native_goal_reward(step_count: int, max_episode_steps: int) -> float:
        """MiniGridEnv._reward() for a goal reached on ``step_count``."""
        return 1.0 - 0.9 * float(step_count) / float(max(max_episode_steps, 1))

    def _transition(
        self,
        state: torch.Tensor,
        inventory: int,
        action: int,
        goal_yx: tuple[int, int],
        episode_step: int,
        max_episode_steps: int,
    ) -> tuple[torch.Tensor, int, float, bool]:
        enters_lava = self._forward_enters_lava(state, action)
        next_states, next_inventory, diagnostics = rollout_minigrid_wm(
            self.model,
            state.unsqueeze(0),
            torch.tensor([action], device=DEVICE),
            torch.tensor([inventory], device=DEVICE),
            self.attention_mask_size,
        )
        next_state = next_states[0]
        next_inventory_value = int(next_inventory[0].item())
        after_position = diagnostics["agent_positions_after"][0]
        reached_goal = tuple(map(int, after_position.detach().cpu().tolist())) == goal_yx
        uncertainty = float(diagnostics["uncertainty"][0].item())
        reward = self._native_goal_reward(episode_step, max_episode_steps) if reached_goal else 0.0
        self._imagined_transitions += 1
        self._uncertainty_sum += uncertainty
        return next_state, next_inventory_value, reward, reached_goal or enters_lava

    def _select_child(self, node: _Node) -> _Node:
        parent_visits = max(node.visits, 1)
        def score(child: _Node) -> float:
            exploit = child.immediate_reward + self.gamma * child.value
            explore = self.exploration_constant * math.sqrt(
                math.log(parent_visits + 1.0) / max(child.visits, 1)
            )
            return exploit + explore
        return max(node.children.values(), key=score)

    def _expand(
        self,
        node: _Node,
        goal_yx: tuple[int, int],
        environment_step: int,
        max_episode_steps: int,
    ) -> _Node:
        action = random.choice(tuple(node.unexpanded_actions))
        node.unexpanded_actions.remove(action)
        state, inventory, reward, terminal = self._transition(
            node.state,
            node.inventory,
            action,
            goal_yx,
            environment_step + node.depth + 1,
            max_episode_steps,
        )
        child = _Node(
            state=state,
            inventory=inventory,
            parent=node,
            action=action,
            immediate_reward=reward,
            depth=node.depth + 1,
            terminal=terminal,
        )
        if terminal or child.depth >= self.max_depth:
            child.unexpanded_actions.clear()
        node.children[action] = child
        self._tree_nodes += 1
        return child

    def _rollout_value(
        self,
        node: _Node,
        goal_yx: tuple[int, int],
        environment_step: int,
        max_episode_steps: int,
    ) -> float:
        if node.terminal:
            return 0.0
        state, inventory = node.state, node.inventory
        value = 0.0
        discount = 1.0
        remaining_depth = max(0, self.max_depth - node.depth)
        for rollout_step in range(min(self.rollout_depth, remaining_depth)):
            action = random.randrange(MODEL_ACTION_COUNT)
            state, inventory, reward, terminal = self._transition(
                state,
                inventory,
                action,
                goal_yx,
                environment_step + node.depth + rollout_step + 1,
                max_episode_steps,
            )
            value += discount * reward
            if terminal:
                return value
            discount *= self.gamma
        return value

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        inventory_token: int,
        goal_yx: tuple[int, int],
        environment_step: int = 0,
        max_episode_steps: int = 1000,
    ) -> MCTSPlan:
        state = torch.as_tensor(state, device=DEVICE, dtype=torch.float32)
        if state.ndim != 3:
            raise ValueError(f"Expected MiniGrid state [3,H,W], got {tuple(state.shape)}")
        self._imagined_transitions = 0
        self._uncertainty_sum = 0.0
        self._invalid_actions = 0
        self._tree_nodes = 1
        root = _Node(state=state, inventory=int(inventory_token))

        for _ in range(self.simulations):
            node = root
            path = [root]
            while (
                not node.terminal
                and node.depth < self.max_depth
                and not node.unexpanded_actions
                and node.children
            ):
                node = self._select_child(node)
                path.append(node)
            if not node.terminal and node.depth < self.max_depth and node.unexpanded_actions:
                node = self._expand(
                    node, goal_yx, environment_step, max_episode_steps
                )
                path.append(node)
            value = self._rollout_value(
                node, goal_yx, environment_step, max_episode_steps
            )
            for current in reversed(path):
                current.visits += 1
                current.value_sum += value
                if current.parent is not None:
                    value = current.immediate_reward + self.gamma * value

        if not root.children:
            return MCTSPlan(
                actions=[self.fallback_action],
                score=float("-inf"),
                predicted_next_state=state,
                predicted_next_inventory=int(inventory_token),
                diagnostics={"simulations": self.simulations, "imagined_transitions": 0, "mean_rollout_uncertainty": 0.0},
            )
        child = max(
            root.children.values(),
            key=lambda candidate: (candidate.visits, candidate.immediate_reward + self.gamma * candidate.value),
        )
        actions = [int(child.action)]
        cursor = child
        while cursor.children and len(actions) < self.max_depth:
            cursor = max(cursor.children.values(), key=lambda candidate: candidate.visits)
            actions.append(int(cursor.action))
        return MCTSPlan(
            actions=actions,
            score=child.immediate_reward + self.gamma * child.value,
            predicted_next_state=child.state,
            predicted_next_inventory=child.inventory,
            diagnostics={
                "simulations": self.simulations,
                "tree_nodes": self._tree_nodes,
                "root_visits": root.visits,
                "imagined_transitions": self._imagined_transitions,
                "invalid_imagined_actions": self._invalid_actions,
                "mean_rollout_uncertainty": self._uncertainty_sum / max(self._imagined_transitions, 1),
            },
        )


def run_online_mcts(cfg: DictConfig) -> list[dict]:
    if bool(getattr(cfg.PPO, "train_in_real_env", False)):
        raise ValueError("MCTS requires PPO.train_in_real_env=false so it can load a WM")
    if str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower() != "mcts":
        raise ValueError("run_online_mcts requires PPO.wm_control_mode=mcts")
    _seed_everything(int(cfg.PPO.seed))
    model, checkpoint = _load_world_model(cfg)
    tree_cfg = cfg.PPO.mcts
    planner = WorldModelMCTS(model, cfg.attention_model.attention_mask_size, tree_cfg)
    output_dir = Path(str(tree_cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print_every_steps = int(tree_cfg.print_every_steps)
    if print_every_steps < 1:
        raise ValueError("PPO.mcts.print_every_steps must be at least 1")
    step_fields = ("episode", "seed", "environment_step", "total_environment_steps", "action", "action_name", "real_reward", "cumulative_real_reward", "plan_score", "planning_latency_ms", "imagined_transitions_this_plan", "imagined_transitions_total", "wm_uncertainty", "pose_match", "inventory_match", "terminated", "truncated")
    step_path = output_dir / "wm_mcts_steps.csv"
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=step_fields).writeheader()
    rows, traces, total_environment_steps = [], [], 0
    for episode in range(int(tree_cfg.episodes)):
        env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=str(cfg.PPO.env_path), custom_mission="Reach the goal.", max_steps=int(cfg.PPO.max_ep_len), render_mode=None))
        observation, _ = env.reset(seed=int(cfg.PPO.seed) + episode)
        terminated = truncated = False
        reward_sum = planning_seconds = 0.0
        imagined_transitions = plan_calls = 0
        pose_matches, inventory_matches, action_trace = [], [], []
        while not (terminated or truncated):
            state_np = _state_from_observation(observation)
            inventory = carrying_token_from_env(env)
            started = time.perf_counter()
            plan = planner.plan(
                torch.as_tensor(state_np, device=DEVICE),
                inventory,
                _find_goal(state_np),
                environment_step=len(action_trace),
                max_episode_steps=int(cfg.PPO.max_ep_len),
            )
            latency = time.perf_counter() - started
            planning_seconds += latency
            plan_calls += 1
            imagined_transitions += int(plan.diagnostics["imagined_transitions"])
            action = plan.actions[0] if plan.actions else planner.fallback_action
            observation, reward, terminated, truncated, _ = env.step(compact_to_native(action))
            reward_sum += float(reward)
            total_environment_steps += 1
            real_next = _state_from_observation(observation)
            predicted_position = minigrid_utils.get_agent_position(plan.predicted_next_state.detach().cpu().numpy())
            real_position = minigrid_utils.get_agent_position(real_next)
            pose_matches.append(predicted_position == real_position)
            inventory_matches.append(plan.predicted_next_inventory == carrying_token_from_env(env))
            step_row = {"episode": episode, "seed": int(cfg.PPO.seed) + episode, "environment_step": len(action_trace) + 1, "total_environment_steps": total_environment_steps, "action": int(action), "action_name": COMPACT_ACTION_NAMES[action], "real_reward": float(reward), "cumulative_real_reward": reward_sum, "plan_score": plan.score, "planning_latency_ms": 1000.0 * latency, "imagined_transitions_this_plan": int(plan.diagnostics["imagined_transitions"]), "imagined_transitions_total": imagined_transitions, "wm_uncertainty": plan.diagnostics["mean_rollout_uncertainty"], "pose_match": pose_matches[-1], "inventory_match": inventory_matches[-1], "terminated": bool(terminated), "truncated": bool(truncated)}
            action_trace.append(step_row)
            with step_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=step_fields)
                writer.writerow(step_row)
                handle.flush()
            if total_environment_steps % print_every_steps == 0 or terminated or truncated:
                print(f"[WM MCTS][ep={episode} step={step_row['environment_step']} total_env={total_environment_steps}] action={step_row['action_name']} reward={float(reward):.3f} return={reward_sum:.3f} score={plan.score:.3f} wm_steps={step_row['imagined_transitions_this_plan']} latency={step_row['planning_latency_ms']:.1f}ms")
        rows.append({"episode": episode, "seed": int(cfg.PPO.seed) + episode, "success": bool(terminated and reward_sum > 0.0), "environment_steps": len(action_trace), "real_reward": reward_sum, "plan_calls": plan_calls, "imagined_transitions": imagined_transitions, "mean_planning_latency_ms": 1000.0 * planning_seconds / max(plan_calls, 1), "wm_real_pose_match_rate": float(np.mean(pose_matches)) if pose_matches else 0.0, "wm_real_inventory_match_rate": float(np.mean(inventory_matches)) if inventory_matches else 0.0, "wm_checkpoint": str(checkpoint)})
        traces.append({"episode": episode, "actions": action_trace})
        env.close()
    results_path = output_dir / "wm_mcts_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "wm_mcts_traces.json").open("w", encoding="utf-8") as handle:
        json.dump(traces, handle, indent=2)
    print(f"[WM MCTS] success={np.mean([row['success'] for row in rows]):.1%}; environment_steps={sum(row['environment_steps'] for row in rows)}; imagined_transitions={sum(row['imagined_transitions'] for row in rows)}")
    print(f"[WM MCTS] results={results_path}")
    print(f"[WM MCTS] step curve={step_path}")
    return rows


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_online_mcts(cfg)


if __name__ == "__main__":
    main()
