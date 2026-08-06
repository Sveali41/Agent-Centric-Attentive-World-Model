"""Action mapping between the compact WM/PPO space and MiniGrid's native space.

The data collector excludes only ``done``. The first five compact IDs preserve
the historical mapping so existing WM datasets/checkpoints keep their meaning;
``drop`` is appended as compact action 5:

    compact: 0, 1, 2, 3, 4, 5
    native:  0, 1, 2, 3, 5, 4
"""

from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX


COMPACT_ACTION_NAMES = ("left", "right", "forward", "pickup", "toggle", "drop")
COMPACT_TO_NATIVE = np.asarray([0, 1, 2, 3, 5, 4], dtype=np.int64)
NATIVE_TO_COMPACT = {int(native): compact for compact, native in enumerate(COMPACT_TO_NATIVE)}
MODEL_ACTION_COUNT = len(COMPACT_ACTION_NAMES)
INVENTORY_TOKEN_COUNT = len(COLOR_TO_IDX) + 1


def carrying_token_from_env(env) -> int:
    """Encode the current carried key as empty=0 or colour-id+1."""
    carrying = getattr(env.unwrapped, "carrying", None)
    if carrying is None or getattr(carrying, "type", None) != "key":
        return 0
    return int(COLOR_TO_IDX[carrying.color]) + 1



def compact_to_native(action: int) -> int:
    """Convert a policy/WM action to a MiniGrid ``env.step`` action."""
    action = int(action)
    if action < 0 or action >= MODEL_ACTION_COUNT:
        raise ValueError(
            f"Invalid compact MiniGrid action {action}; "
            f"expected 0..{MODEL_ACTION_COUNT - 1}"
        )
    return int(COMPACT_TO_NATIVE[action])


def native_to_compact(actions):
    """Convert collected native actions to the compact dataset action space."""
    values = np.asarray(actions)
    flat = values.reshape(-1)
    try:
        compact = np.asarray([NATIVE_TO_COMPACT[int(x)] for x in flat], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"Dataset contains excluded/unknown MiniGrid action {exc.args[0]}; "
            f"allowed native actions are {sorted(NATIVE_TO_COMPACT)}"
        ) from exc
    return compact.reshape(values.shape)
