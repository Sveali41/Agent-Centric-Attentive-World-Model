import unittest

import torch

from generator.reward_system import DiversityModule


class CrafterDiversityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = DiversityModule(
            input_h=6, input_w=6, k=1, max_archive_size=10, device="cpu",
            env_type="crafter", crafter_map_weight=1.0,
            crafter_start_weight=0.1, crafter_inventory_weight=0.2,
        )
        self.inventory = torch.zeros(16)

    @staticmethod
    def _map(agent=(2, 2), tree=None):
        grid = torch.zeros((2, 6, 6), dtype=torch.long)
        grid[0].fill_(2)  # grass
        grid[0, agent[0], agent[1]] = 13
        if tree is not None:
            grid[0, tree[0], tree[1]] = 6
        return grid

    def _score_after_base(self, candidate, inventory=None):
        self.module.archive.clear()
        self.assertEqual(self.module.get_reward(self._map().unsqueeze(0), self.inventory), 0.0)
        score = self.module.get_reward(candidate.unsqueeze(0), self.inventory if inventory is None else inventory)
        return score, self.module.last_components.copy()

    def test_components_measure_only_their_intended_change(self):
        self.module.archive.clear()
        self.module.get_reward(self._map().unsqueeze(0), self.inventory)
        self.assertEqual(
            self.module.get_reward(self._map().unsqueeze(0), self.inventory), 0.0
        )

        _, moved = self._score_after_base(self._map(agent=(3, 3)))
        self.assertEqual(moved["map_edit_novelty"], 0.0)
        self.assertEqual(moved["inventory_novelty"], 0.0)
        self.assertGreater(moved["start_position_novelty"], 0.0)
        self.assertAlmostEqual(moved["total_novelty"], 0.1 * moved["start_position_novelty"])

        _, structure = self._score_after_base(self._map(tree=(1, 1)))
        self.assertGreater(structure["map_edit_novelty"], 0.0)
        self.assertEqual(structure["start_position_novelty"], 0.0)
        self.assertEqual(structure["inventory_novelty"], 0.0)

        inventory = self.inventory.clone()
        inventory[0] = 9
        _, changed_inventory = self._score_after_base(self._map(), inventory)
        self.assertEqual(changed_inventory["map_edit_novelty"], 0.0)
        self.assertEqual(changed_inventory["start_position_novelty"], 0.0)
        self.assertGreater(changed_inventory["inventory_novelty"], 0.0)

    def test_weighted_sum_and_iron_is_not_masked(self):
        self.module.archive.clear()
        base = self._map()
        self.module.get_reward(base.unsqueeze(0), self.inventory)
        candidate = self._map(agent=(3, 3), tree=(1, 1))
        inventory = self.inventory.clone()
        inventory[0] = 9
        total = self.module.get_reward(candidate.unsqueeze(0), inventory)
        c = self.module.last_components
        self.assertAlmostEqual(
            total,
            c["map_edit_novelty"] + 0.1 * c["start_position_novelty"] + 0.2 * c["inventory_novelty"],
        )

        # Crafter iron is ID 9.  It must contribute to map novelty, not be
        # mistaken for the agent as in the old implementation.
        self.module.archive.clear()
        self.module.get_reward(base.unsqueeze(0), self.inventory)
        iron = self._map()
        iron[0, 1, 1] = 9
        self.module.get_reward(iron.unsqueeze(0), self.inventory)
        self.assertGreater(self.module.last_components["map_edit_novelty"], 0.0)

    def test_new_cross_component_combination_remains_novel(self):
        self.module.archive.clear()
        rich_inventory = self.inventory.clone()
        rich_inventory[0] = 9
        tree_map = self._map(tree=(1, 1))

        # Each candidate component has appeared before, but never together.
        self.module.get_reward(self._map().unsqueeze(0), rich_inventory)
        self.module.get_reward(tree_map.unsqueeze(0), self.inventory)
        score = self.module.get_reward(tree_map.unsqueeze(0), rich_inventory)

        self.assertGreater(score, 0.0)
        self.assertAlmostEqual(score, self.module.last_components["total_novelty"])


if __name__ == "__main__":
    unittest.main()
