"""Deterministic tests for MiniGrid MPC native-goal reward scoring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelBased.policy_training.planners import mpc_planner
from modelBased.policy_training.planners.mpc_planner import DenseRewardContext, WorldModelMPC


class MiniGridMPCRewardTest(unittest.TestCase):
    def setUp(self):
        self.height = 5
        self.width = 5
        self.base_state = torch.zeros((3, self.height, self.width), dtype=torch.float32)
        self.base_state[0].fill_(2)
        self.base_state[0, 1:4, 1:4] = 1
        self.base_state[0, 2, 3] = 8

    def _start(self, position=(2, 1), direction=0):
        state = self.base_state.clone()
        state[0, position[0], position[1]] = 10
        state[2, position[0], position[1]] = direction
        return state

    def _planner(self, horizon, population, mode="native_goal"):
        config = SimpleNamespace(
            horizon=horizon,
            population=population,
            elite_count=1,
            iterations=1,
            gamma=0.99,
            fallback_action=0,
            minigrid_execute_steps=horizon,
            minigrid_reward_mode=mode,
            minigrid_goal_guide_weight=0.05,
        )
        ppo = SimpleNamespace(main_dense_reward=SimpleNamespace(lava_reward=-1.0))
        return WorldModelMPC(None, 1, config, ppo)

    def _distance_context(self, start, goal=(2, 3)):
        distance = mpc_planner.build_goal_distance_map(
            start.to(mpc_planner.DEVICE), goal
        )
        return SimpleNamespace(
            goal_distance_map=distance,
            settings={"lava_reward": -1.0},
        )

    def _dense_context(self, start, goal=(2, 3)):
        settings = SimpleNamespace(
            step_penalty=-0.001,
            progress_weight=0.0,
            best_progress_weight=0.0,
            progress_milestone_reward=0.0,
            goal_region_action_progress_enabled=False,
            goal_region_reward=0.0,
            goal_region_key_pickup_reward=0.0,
            goal_region_door_open_reward=0.0,
            critical_door_crossing_reward=0.0,
            lava_reward=-1.0,
        )
        return DenseRewardContext(
            start.to(mpc_planner.DEVICE), goal,
            mpc_planner.reward_settings(SimpleNamespace(main_dense_reward=settings)),
        )

    def _install_fixed_rollout(self):
        base = self.base_state.to(mpc_planner.DEVICE).clone()
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

        def rollout(_model, states, actions, inventories, _mask_size):
            next_states = []
            next_positions = []
            next_inventories = inventories.clone()
            for row, action in zip(states, actions.tolist()):
                position = mpc_planner.utils.get_agent_position_torch(row)
                y, x = map(int, position.tolist())
                direction = int(row[2, y, x].item())
                if action == 0:
                    direction = (direction - 1) % 4
                elif action == 1:
                    direction = (direction + 1) % 4
                elif action == 2:
                    dy, dx = directions[direction]
                    ny, nx = y + dy, x + dx
                    if int(base[0, ny, nx].item()) != 2:
                        y, x = ny, nx
                next_state = base.clone()
                next_state[0, y, x] = 10
                next_state[2, y, x] = direction
                next_states.append(next_state)
                next_positions.append((y, x))
            return (
                torch.stack(next_states),
                next_inventories,
                {
                    "agent_positions_after": torch.tensor(
                        next_positions, device=states.device, dtype=torch.long
                    ),
                    "uncertainty": torch.zeros(
                        len(states), device=states.device, dtype=torch.float32
                    ),
                },
            )

        return patch.object(mpc_planner, "rollout_minigrid_wm", side_effect=rollout)

    def _evaluate(self, planner, start, context, sequences, episode_step=0, max_steps=100):
        with self._install_fixed_rollout():
            return planner._evaluate(
                start.to(mpc_planner.DEVICE), 0,
                torch.tensor(sequences, dtype=torch.long, device=mpc_planner.DEVICE), (2, 3),
                context, episode_step, max_steps,
            )

    def test_realized_guide_bonus_matches_endpoint_progress_and_failure_rules(self):
        bonus = mpc_planner._realized_goal_guide_bonus(
            start_distance=4.0,
            end_distance=2.0,
            weight=0.05,
            gamma=0.9,
            executed_steps=3,
            failed=False,
        )
        self.assertAlmostEqual(bonus, 0.05 * 0.5 * (0.9 ** 2), places=7)
        self.assertAlmostEqual(
            mpc_planner._realized_goal_guide_bonus(2.0, 4.0, 0.05, 0.9, 1, False),
            -0.05,
            places=7,
        )
        self.assertEqual(
            mpc_planner._realized_goal_guide_bonus(4.0, 0.0, 0.05, 0.9, 3, True),
            0.0,
        )
        self.assertEqual(
            mpc_planner._realized_goal_guide_bonus(float("inf"), 0.0, 0.05, 0.9, 3, False),
            0.0,
        )

    def test_closer_gets_positive_net_distance_guidance(self):
        planner = self._planner(horizon=1, population=2)
        result = self._evaluate(
            planner, self._start(), self._distance_context(self._start()),
            [[2], [1]],
        )
        scores, _, _, native, guide, _, _, _ = result
        self.assertEqual(native.tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(guide[0]), 0.025, places=6)
        self.assertEqual(float(guide[1]), 0.0)
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_moving_away_is_negative_and_turning_loop_has_zero_net_progress(self):
        planner = self._planner(horizon=2, population=2)
        start = self._start(position=(2, 2), direction=2)
        context = self._distance_context(start)
        result = self._evaluate(planner, start, context, [[2, 0], [1, 0]])
        _, _, _, _, guide, _, _, _ = result
        self.assertLess(float(guide[0]), 0.0)
        self.assertEqual(float(guide[1]), 0.0)

    def test_native_success_reward_reflects_real_and_imagined_step_count(self):
        planner = self._planner(horizon=3, population=2)
        start = self._start(position=(2, 2), direction=0)
        context = self._distance_context(start)
        result = self._evaluate(
            planner, start, context,
            [[2, 0, 0], [1, 0, 2]],
            episode_step=10, max_steps=100,
        )
        scores, _, _, native, guide, _, _, _ = result
        self.assertAlmostEqual(float(native[0]), 1.0 - 0.9 * 11 / 100, places=6)
        self.assertAlmostEqual(
            float(native[1]), (1.0 - 0.9 * 13 / 100) * planner.gamma**2, places=6
        )
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertGreater(float(guide[0]), 0.0)

    def test_lava_termination_keeps_failure_cost_and_suppresses_guide(self):
        self.base_state[0, 2, 2] = 9
        start = self._start()
        planner = self._planner(horizon=1, population=1)
        context = self._distance_context(start)
        result = self._evaluate(planner, start, context, [[2]])
        scores, _, _, native, guide, failure, _, _ = result
        self.assertEqual(float(native[0]), 0.0)
        self.assertEqual(float(guide[0]), 0.0)
        self.assertEqual(float(failure[0]), -1.0)
        self.assertEqual(float(scores[0]), -1.0)

    def test_legacy_mode_returns_only_legacy_dense_score(self):
        planner = self._planner(horizon=2, population=1, mode="legacy_dense")
        start = self._start()
        result = self._evaluate(
            planner, start, self._dense_context(start), [[1, 0]]
        )
        scores, _, _, native, guide, failure, legacy, _ = result
        self.assertEqual(float(native[0]), 0.0)
        self.assertEqual(float(guide[0]), 0.0)
        self.assertEqual(float(failure[0]), 0.0)
        expected = -0.001 * (1.0 + planner.gamma)
        self.assertAlmostEqual(float(scores[0]), expected, places=7)
        self.assertAlmostEqual(float(scores[0]), float(legacy[0]), places=7)

    def test_time_limit_ends_native_rollout_without_distance_bonus(self):
        planner = self._planner(horizon=2, population=1)
        start = self._start()
        result = self._evaluate(
            planner, start, self._distance_context(start), [[2, 0]],
            episode_step=99, max_steps=100,
        )
        scores, _, _, native, guide, failure, _, _ = result
        self.assertEqual(float(native[0]), 0.0)
        self.assertEqual(float(guide[0]), 0.0)
        self.assertEqual(float(failure[0]), 0.0)
        self.assertEqual(float(scores[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
