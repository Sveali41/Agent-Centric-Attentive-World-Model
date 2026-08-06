"""Action mapping between the compact WM/PPO space and MiniGrid's native space.

The data collector intentionally excludes ``drop`` and ``done``. Therefore the
model uses five compact actions, while MiniGrid uses native IDs with a gap:

    compact: 0, 1, 2, 3, 4
    native:  0, 1, 2, 3, 5
"""

from __future__ import annotations

import numpy as np


COMPACT_ACTION_NAMES = ("left", "right", "forward", "pickup", "toggle")
COMPACT_TO_NATIVE = np.asarray([0, 1, 2, 3, 5], dtype=np.int64)
NATIVE_TO_COMPACT = {int(native): compact for compact, native in enumerate(COMPACT_TO_NATIVE)}
MODEL_ACTION_COUNT = len(COMPACT_ACTION_NAMES)



def compact_to_native(action: int) -> int:
    """Convert a policy/WM action to a MiniGrid ``env.step`` action."""
    action = int(action)
    if action < 0 or action >= MODEL_ACTION_COUNT:
        raise ValueError(f"Invalid compact MiniGrid action {action}; expected 0..4")
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
