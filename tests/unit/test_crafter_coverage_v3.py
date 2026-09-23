"""Focused invariants for target-reachable Crafter coverage_v3."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from modelBased.common.artifacts import metadata_array
from modelBased.data.data_collect import (
    _CRAFTER_COVERAGE_V2_QUOTAS,
    _crafter_coverage_v3_queue,
    _crafter_coverage_v3_setup,
)


_LAYOUT = (
    Path(__file__).resolve().parents[3]
    / "trainer/level/crafter/target_tasks/crafter_target_task_1.txt"
)


class CrafterCoverageV3Test(unittest.TestCase):
    def setUp(self):
        self.env = CustomCrafterEnv(
            txt_file_path=str(_LAYOUT), max_steps=20, seed=11, ai_enabled=False
        )

    def tearDown(self):
        self.env.close()

    def test_action_queue_preserves_12500_published_quota(self):
        queue = _crafter_coverage_v3_queue(3, 12500)
        actions = np.asarray([action for action, _ in queue])
        self.assertEqual(len(actions), 12500)
        for action, count in _CRAFTER_COVERAGE_V2_QUOTAS.items():
            self.assertEqual(int((actions == action).sum()), count)

    def test_path_tail_is_a_native_success_without_lava_or_sand(self):
        # index >= 540 is the 60-row movement tail retained for each action.
        self.env.reset()
        event, _ = _crafter_coverage_v3_setup(
            self.env, 1, 540, np.random.default_rng(7)
        )
        self.assertEqual(event, "move_path_success")
        player = self.env.env._player
        # Native action 1 is left; movement setup defines this direction even
        # though the player's stored facing remains a seeded independent value.
        front = tuple((np.asarray(player.pos) + np.asarray((-1, 0))).tolist())
        self.assertEqual(self.env.env._world[front][0], "path")
        materials = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                point = tuple((np.asarray(player.pos) + (dx, dy)).tolist())
                materials.append(self.env.env._world[point][0])
        self.assertNotIn("lava", materials)
        self.assertNotIn("sand", materials)
        before = tuple(player.pos)
        _, _, terminated, truncated, _ = self.env.step(1)
        self.assertFalse(terminated or truncated)
        self.assertNotEqual(tuple(self.env.env._player.pos), before)

    def test_native_mining_turns_non_tree_resources_into_path(self):
        # action 5 uses five 80-row positive resource blocks. Tree becomes
        # grass; every mineral resource must use Crafter's native path result.
        for index, resource in ((80, "stone"), (160, "coal"), (240, "iron"), (320, "diamond")):
            self.env.reset()
            event, _ = _crafter_coverage_v3_setup(
                self.env, 5, index, np.random.default_rng(100 + index)
            )
            self.assertEqual(event, f"do_{resource}_success")
            player = self.env.env._player
            front = tuple((np.asarray(player.pos) + np.asarray(player.facing)).tolist())
            self.assertEqual(self.env.env._world[front][0], resource)
            _, _, terminated, truncated, _ = self.env.step(5)
            self.assertFalse(terminated or truncated)
            self.assertEqual(self.env.env._world[front][0], "path")

    def test_all_scenario_family_boundaries_never_inject_lava_or_sand(self):
        # Test every family at representative/boundary positions, not merely
        # movement, so a future setup change cannot silently reintroduce an
        # out-of-target terrain element into v3.
        boundaries = {
            **{action: (0, 239, 240, 539, 540, 599) for action in range(1, 5)},
            5: (0, 79, 80, 159, 160, 239, 240, 319, 320, 399, 400, 449,
                450, 499, 500, 579, 580, 659, 660, 739, 740, 819, 820, 899),
            6: (0, 1),
            **{action: (0, 399, 400, 599, 600, 699, 700, 799)
               for action in range(7, 17)},
        }
        rng = np.random.default_rng(509)
        for action, indices in boundaries.items():
            for index in indices:
                self.env.reset()
                _crafter_coverage_v3_setup(self.env, action, index, rng)
                player = self.env.env._player
                materials = [
                    self.env.env._world[
                        tuple((np.asarray(player.pos) + (dx, dy)).tolist())
                    ][0]
                    for dx in range(-2, 3)
                    for dy in range(-2, 3)
                ]
                self.assertNotIn("lava", materials, msg=f"action={action}, index={index}")
                self.assertNotIn("sand", materials, msg=f"action={action}, index={index}")

    def test_seeded_v3_setup_replays_exactly(self):
        def states(seed):
            rng = np.random.default_rng(seed)
            result = []
            for _ in range(12):
                self.env.reset()
                event, stage = _crafter_coverage_v3_setup(self.env, 4, 540, rng)
                player = self.env.env._player
                result.append((event, stage, tuple(player.pos), tuple(player.facing)))
            return result

        self.assertEqual(states(31), states(31))

    def test_reset_reuses_textures_but_rebuilds_world_views(self):
        textures = self.env.env._textures
        first_world = self.env.env._world
        first_local_view = self.env.env._local_view
        first_item_view = self.env.env._item_view

        self.env.reset()

        self.assertIs(self.env.env._textures, textures)
        self.assertIsNot(self.env.env._world, first_world)
        self.assertIsNot(self.env.env._local_view, first_local_view)
        self.assertIsNot(self.env.env._item_view, first_item_view)
        self.assertIs(self.env.env._local_view._textures, textures)
        self.assertIs(self.env.env._item_view._textures, textures)
        self.assertIs(self.env.env._local_view._world, self.env.env._world)

    def test_metadata_declares_target_reachable_protocol(self):
        cfg = OmegaConf.create({
            "domain": "crafter",
            "env": {"collect": {"data_type": "coverage_v3"}},
            "domains": {"crafter": {
                "task_name": "crafter_target_task_1",
                "layout_path": str(_LAYOUT),
                "data_collection": {"max_steps": 20},
            }},
        })
        metadata = json.loads(metadata_array(cfg).item())
        self.assertEqual(
            metadata["collection_policy"], "crafter_coverage_v3_target_reachable_v1"
        )
        self.assertEqual(
            metadata["coverage_context_protocol"], "causal_tech_stage_inventory_v3"
        )


if __name__ == "__main__":
    unittest.main()
