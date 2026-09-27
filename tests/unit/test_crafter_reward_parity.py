"""Crafter reward parity for craft events whose output can saturate."""

import unittest

import numpy as np
import torch
from crafter import constants

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import (
    ACHIEVEMENT_NAMES, CrafterAchievementTracker, native_reward_batch,
)


class CrafterRewardParityTests(unittest.TestCase):
    def test_craft_reward_matches_native_crafter(self):
        for action in range(11, 17):
            for tool_count in (0, 9):
                with self.subTest(action=action, tool_count=tool_count):
                    self._check_craft(action, tool_count)

    def _check_craft(self, action, tool_count):
        item = constants.actions[action].removeprefix("make_")
        initial_inventory = {name: 1 for name in constants.make[item]["uses"]}
        initial_inventory[item] = tool_count
        env = CustomCrafterEnv(
            layout_str="GGGGG\nGGUGG\nGGAXG\nGGGGG\nGGGGG",
            initial_inventory=initial_inventory, max_steps=1, seed=0,
        )
        try:
            observation, _ = env.reset()
            player = env.env._player
            life = {
                name: torch.tensor([getattr(player, attribute)], dtype=dtype)
                for name, attribute, dtype in (
                    ("hunger", "_hunger", torch.float32),
                    ("thirst", "_thirst", torch.float32),
                    ("fatigue", "_fatigue", torch.float32),
                    ("recover", "_recover", torch.float32),
                    ("sleeping", "sleeping", torch.bool),
                )
            }
            state = torch.tensor(np.transpose(observation["image"], (2, 0, 1)), dtype=torch.float32)
            inventory = torch.tensor(observation["inventory"], dtype=torch.float32)
            following, real_reward, _, _, info = env.step(action)
            next_state = torch.tensor(np.transpose(following["image"], (2, 0, 1)), dtype=torch.float32)
            next_inventory = torch.tensor(following["inventory"], dtype=torch.float32)
            imagined_reward, new, _, _, _, _ = native_reward_batch(
                state[None], inventory[None], torch.tensor([action]),
                next_state[None], next_inventory[None],
                torch.zeros((1, len(ACHIEVEMENT_NAMES)), dtype=torch.bool),
                life["sleeping"], torch.full((1, 5, 5), -1.0), life,
            )
            scalar = CrafterAchievementTracker().update(
                state.numpy(), inventory.numpy(), action,
                next_state.numpy(), next_inventory.numpy(),
            )
            expected = [f"make_{item}"]
            self.assertEqual(info["newly_unlocked"], expected)
            self.assertEqual(
                [name for name, present in zip(ACHIEVEMENT_NAMES, new[0].tolist()) if present],
                expected,
            )
            self.assertEqual(scalar["newly_unlocked"], expected)
            self.assertAlmostEqual(float(imagined_reward[0]), real_reward)
            self.assertAlmostEqual(scalar["reward"], real_reward)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
