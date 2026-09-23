"""Focused tests for Crafter local-state boundary padding."""

from __future__ import annotations

import unittest

import numpy as np

from domain.minigrid import minigrid_support


class CrafterPaddingTest(unittest.TestCase):
    def test_corner_crop_uses_zero_for_both_channels(self):
        state = np.zeros((2, 4, 4), dtype=np.int64)
        state[0] = np.arange(16).reshape(4, 4)
        state[1] = 100 + np.arange(16).reshape(4, 4)

        cropped = minigrid_support.extract_masked_state(
            state, mask_size=3, agent_position_yx=(0, 0)
        )

        expected = np.zeros((2, 3, 3), dtype=np.int64)
        expected[:, 1:, 1:] = state[:, :2, :2]
        np.testing.assert_array_equal(cropped, expected)


if __name__ == "__main__":
    unittest.main()
