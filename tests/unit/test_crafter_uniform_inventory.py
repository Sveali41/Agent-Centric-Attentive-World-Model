"""Focused tests for Crafter uniform inventory collection."""

from types import SimpleNamespace
from unittest.mock import patch

from modelBased.data.data_collect import (
    _CRAFTER_UNIFORM_INVENTORY_ITEMS,
    _randomize_crafter_uniform_inventory,
)


def test_uniform_inventory_samples_all_editable_slots_from_zero_to_nine():
    survival = {"health": 4.0, "food": 5.0, "drink": 6.0, "energy": 7.0}
    player = SimpleNamespace(inventory={**survival})
    draws = list(range(10)) + [2, 9]

    with patch("modelBased.data.data_collect.np.random.randint", side_effect=draws) as randint:
        _randomize_crafter_uniform_inventory(player)

    assert set(player.inventory) == set(survival) | set(_CRAFTER_UNIFORM_INVENTORY_ITEMS)
    assert [player.inventory[item] for item in _CRAFTER_UNIFORM_INVENTORY_ITEMS] == [float(v) for v in draws]
    assert all(0 <= player.inventory[item] <= 9 for item in _CRAFTER_UNIFORM_INVENTORY_ITEMS)
    assert player.inventory["wood_pickaxe"] == 6.0
    assert player.inventory["iron_sword"] == 9.0
    assert {name: player.inventory[name] for name in survival} == survival
    assert randint.call_count == len(_CRAFTER_UNIFORM_INVENTORY_ITEMS)
    assert all(call.args == (0, 10) for call in randint.call_args_list)
