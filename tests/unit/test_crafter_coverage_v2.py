"""Focused invariants for the Crafter coverage_v2 validation collector."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

import numpy as np
from omegaconf import OmegaConf

from modelBased.common.artifacts import metadata_array
from modelBased.data.data_collect import (
    _CRAFTER_COVERAGE_V2_QUOTAS,
    _CRAFTER_COLLECT_MIN_STAGE,
    _CRAFTER_MAKE_MIN_STAGE,
    _CRAFTER_PLACE_MIN_STAGE,
    _crafter_coverage_v2_observed_event,
    _crafter_coverage_v2_queue,
    _crafter_coverage_v2_setup,
)
from domain.crafter.crafter_custom_env import CustomCrafterEnv


_LAYOUT = (
    Path(__file__).resolve().parents[3]
    / "trainer/level/crafter/target_tasks/crafter_target_task_1.txt"
)


class CrafterCoverageV2Test(unittest.TestCase):
    def setUp(self):
        self.env = CustomCrafterEnv(
            txt_file_path=str(_LAYOUT), max_steps=20, seed=11, ai_enabled=False
        )

    def tearDown(self):
        self.env.close()

    def _setup_and_step(self, action, index, seed):
        self.env.reset()
        event, stage = _crafter_coverage_v2_setup(
            self.env, action, index, np.random.default_rng(seed)
        )
        obs = self.env._extract_obs()
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        return event, stage, obs, next_obs, reward, terminated, truncated

    def test_regular_and_lava_transitions_have_native_liveness(self):
        _, _, obs, _, reward, terminated, truncated = self._setup_and_step(0, 0, 1)
        self.assertEqual(obs["inventory"][0], 9)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _, _, obs, _, _, terminated, _ = self._setup_and_step(1, 540, 2)
        self.assertEqual(obs["inventory"][0], 9)
        self.assertTrue(terminated)

    def test_seeded_setup_replays_and_uses_multiple_teleports(self):
        def setup_states(seed):
            rng = np.random.default_rng(seed)
            result = []
            for _ in range(12):
                self.env.reset()
                _, stage = _crafter_coverage_v2_setup(self.env, 0, 0, rng)
                player = self.env.env._player
                result.append((
                    tuple(player.pos), tuple(player.facing),
                    tuple(self.env._extract_obs()["inventory"]), stage,
                ))
            return result

        first = setup_states(31)
        self.assertEqual(first, setup_states(31))
        self.assertGreater(len({state[0] for state in first}), 1)

    def test_progression_backgrounds_are_diverse_and_causally_valid(self):
        # action 0 adds no scenario-specific inventory overrides, exposing
        # the sampled background protocol directly.
        rng = np.random.default_rng(71)
        contexts = set()
        stages = set()
        raw_minimum_stage = {4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 1}
        tool_slots = (10, 11, 12)  # wood/stone/iron pickaxe
        for _ in range(100):
            self.env.reset()
            _, stage = _crafter_coverage_v2_setup(self.env, 0, 0, rng)
            inv = self.env._extract_obs()["inventory"]
            contexts.add(tuple(inv[4:]))
            stages.add(stage)
            # Future raw resources and pickaxes cannot appear early.
            for slot, minimum in raw_minimum_stage.items():
                if stage < minimum:
                    self.assertEqual(float(inv[slot]), 0.0)
            for required, slot in enumerate(tool_slots, start=2):
                self.assertEqual(float(inv[slot]), 1.0 if stage >= required else 0.0)
        self.assertEqual(stages, {0, 1, 2, 3, 4})
        self.assertGreater(len(contexts), 50)

    def test_progression_scenarios_preserve_success_and_missing_controls(self):
        # Diamond success requires the stage-4 iron-pickaxe context and keeps
        # output capacity; missing diamond collection is sampled pre-stage-4.
        _, stage, obs, next_obs, _, terminated, _ = self._setup_and_step(5, 320, 9)
        self.assertEqual(stage, 4)
        self.assertEqual(obs["inventory"][12], 1.0)
        self.assertEqual(obs["inventory"][8], 0.0)
        self.assertGreater(next_obs["inventory"][8], 0.0)
        self.assertFalse(terminated)

        _, stage, obs, next_obs, _, _, _ = self._setup_and_step(5, 740, 9)
        self.assertLess(stage, 4)
        self.assertEqual(obs["inventory"][8], 0.0)
        self.assertEqual(next_obs["inventory"][8], 0.0)

    def test_place_and_make_material_controls_are_real_native_failures(self):
        rng = np.random.default_rng(107)
        # Every recipe's material-negative control must remain a no-change
        # transition even when its sampled stage would otherwise be rich.
        for action in range(7, 11):
            self.env.reset()
            event, _ = _crafter_coverage_v2_setup(self.env, action, 400, rng)
            before = self.env._extract_obs()
            after, reward, terminated, truncated, info = self.env.step(action)
            self.assertIn("missing_material", event)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated or truncated)
            self.assertFalse(info["newly_unlocked"])
            np.testing.assert_array_equal(before["image"], after["image"])
            np.testing.assert_array_equal(before["inventory"], after["inventory"])

    def test_missing_material_controls_use_recipe_valid_progression_stages(self):
        """Missing-material negatives must not contain future progression state."""
        raw_minimum = {
            4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 1,
        }
        tool_minimum = {
            10: 2, 11: 3, 12: 4, 13: 1, 14: 2, 15: 3,
        }
        rng = np.random.default_rng(407)
        for action in range(7, 17):
            self.env.reset()
            event, stage = _crafter_coverage_v2_setup(self.env, action, 400, rng)
            before = self.env._extract_obs()
            after, reward, terminated, truncated, _ = self.env.step(action)
            self.assertIn("missing_material", event)
            minimum = (
                _CRAFTER_PLACE_MIN_STAGE[("stone", "table", "furnace", "plant")[action - 7]]
                if action < 11 else
                _CRAFTER_MAKE_MIN_STAGE[(
                    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                    "wood_sword", "stone_sword", "iron_sword",
                )[action - 11]]
            )
            self.assertGreaterEqual(stage, minimum)
            for slot, slot_minimum in {**raw_minimum, **tool_minimum}.items():
                if stage < slot_minimum:
                    self.assertEqual(float(before["inventory"][slot]), 0.0)
                    self.assertEqual(float(after["inventory"][slot]), 0.0)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated or truncated)
            np.testing.assert_array_equal(before["image"], after["image"])
            np.testing.assert_array_equal(before["inventory"], after["inventory"])

    def test_all_scenario_boundaries_have_no_future_inventory_slots(self):
        """Audit representative boundaries for every event family."""
        raw_minimum = {4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 1}
        tool_minimum = {10: 2, 11: 3, 12: 4, 13: 1, 14: 2, 15: 3}
        # Include every event boundary without iterating through a full target.
        boundaries = {
            0: (0, 239, 240, 539, 540), 1: (0, 239, 240, 539, 540),
            2: (0, 239, 240, 539, 540), 3: (0, 239, 240, 539, 540),
            4: (0, 239, 240, 539, 540),
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
                _, stage = _crafter_coverage_v2_setup(self.env, action, index, rng)
                inv = self.env._extract_obs()["inventory"]
                for slot, slot_minimum in {**raw_minimum, **tool_minimum}.items():
                    if stage < slot_minimum:
                        self.assertEqual(
                            float(inv[slot]), 0.0,
                            msg=f"action={action}, index={index}, stage={stage}, slot={slot}",
                        )
        for action in range(11, 17):
            self.env.reset()
            event, _ = _crafter_coverage_v2_setup(self.env, action, 400, rng)
            before = self.env._extract_obs()
            after, reward, terminated, truncated, info = self.env.step(action)
            self.assertIn("missing_material", event)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated or truncated)
            self.assertFalse(info["newly_unlocked"])
            np.testing.assert_array_equal(before["image"], after["image"])
            np.testing.assert_array_equal(before["inventory"], after["inventory"])

        for action in range(7, 11):
            self.env.reset()
            event, _ = _crafter_coverage_v2_setup(self.env, action, 600, rng)
            before = self.env._extract_obs()
            after, reward, terminated, truncated, info = self.env.step(action)
            self.assertIn("invalid_terrain", event)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated or truncated)
            self.assertFalse(info["newly_unlocked"])
            np.testing.assert_array_equal(before["image"], after["image"])
            np.testing.assert_array_equal(before["inventory"], after["inventory"])

    def test_action_quota_is_exact(self):
        queue = _crafter_coverage_v2_queue(3, 12500)
        actions = np.asarray([action for action, _ in queue])
        self.assertEqual(len(actions), 12500)
        for action, count in _CRAFTER_COVERAGE_V2_QUOTAS.items():
            self.assertEqual(int((actions == action).sum()), count)

    def test_grass_label_uses_observed_sapling_delta(self):
        obs = {"inventory": np.zeros(16, dtype=np.float32)}
        no_effect = {"inventory": np.zeros(16, dtype=np.float32)}
        gained = {"inventory": np.zeros(16, dtype=np.float32)}
        gained["inventory"][9] = 1
        self.assertEqual(
            _crafter_coverage_v2_observed_event("do_grass_stochastic", obs, no_effect),
            "do_grass_no_effect",
        )
        self.assertEqual(
            _crafter_coverage_v2_observed_event("do_grass_stochastic", obs, gained),
            "do_grass_sapling_gain",
        )

    def test_metadata_declares_coverage_protocol(self):
        cfg = OmegaConf.create({
            "domain": "crafter",
            "env": {"collect": {"data_type": "coverage_v2"}},
            "domains": {"crafter": {
                "task_name": "crafter_target_task_1",
                "layout_path": str(_LAYOUT),
                "data_collection": {"max_steps": 20},
            }},
        })
        metadata = json.loads(metadata_array(cfg).item())
        self.assertEqual(
            metadata["collection_policy"], "crafter_coverage_v2_progression_v3"
        )
        self.assertEqual(
            metadata["coverage_context_protocol"], "causal_tech_stage_inventory_v2"
        )


if __name__ == "__main__":
    unittest.main()
