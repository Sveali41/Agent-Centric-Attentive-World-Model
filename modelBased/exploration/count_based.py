"""World-model-independent count novelty for Crafter acquisition."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch


CRAFTER_PLAYER_ID = 13
COUNT_KEY_VERSION = "local5_inv12_sa_v1"


def _as_hwc_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one 3-D Crafter image, got {image.shape}")
    if image.shape[-1] == 2:
        return image
    if image.shape[0] == 2:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"Expected two Crafter image channels, got {image.shape}")


def agent_centric_patch(image, mask_size: int = 5) -> np.ndarray:
    """Return a square HWC patch centred on the symbolic Crafter player."""
    if mask_size < 1 or mask_size % 2 == 0:
        raise ValueError("mask_size must be a positive odd integer")
    image = _as_hwc_image(image)
    hits = np.argwhere(image[..., 0] == CRAFTER_PLAYER_ID)
    if len(hits) != 1:
        raise ValueError(
            f"Expected exactly one Crafter player (id={CRAFTER_PLAYER_ID}), "
            f"found {len(hits)}"
        )
    y, x = (int(value) for value in hits[0])
    half = mask_size // 2
    padded = np.pad(
        image.astype(np.int16, copy=False),
        ((half, half), (half, half), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    y += half
    x += half
    return padded[y - half : y + half + 1, x - half : x + half + 1]


class CrafterStateActionCounter:
    """Global ``N(s, a)`` table using local symbols and progression inventory."""

    def __init__(self, mask_size: int = 5, reward_scale: float = 1.0) -> None:
        self.mask_size = int(mask_size)
        if self.mask_size < 1 or self.mask_size % 2 == 0:
            raise ValueError("mask_size must be a positive odd integer")
        self.reward_scale = float(reward_scale)
        if self.reward_scale < 0.0:
            raise ValueError("reward_scale must be non-negative")
        self._counts: defaultdict[str, int] = defaultdict(int)

    def state_action_key(self, image, inventory, action: int) -> str:
        patch = agent_centric_patch(image, self.mask_size)
        inventory = np.asarray(inventory, dtype=np.float32).reshape(-1)
        if inventory.size < 16:
            raise ValueError(
                f"Crafter count novelty requires 16 inventory values, got {inventory.size}"
            )
        # Slots 0:4 are survival variables. Their routine drift must not create
        # new exploration states, so only the 12 progression slots are keyed.
        progression = np.rint(inventory[4:16]).astype(np.int16)
        payload = (
            patch.tobytes()
            + progression.tobytes()
            + np.asarray(int(action), dtype=np.int16).tobytes()
        )
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def count(self, image, inventory, action: int) -> int:
        return self._counts.get(self.state_action_key(image, inventory, action), 0)

    def rewards(
        self,
        images,
        inventories,
        actions,
        *,
        terminated=None,
        update: bool = True,
    ) -> np.ndarray:
        """Compute ``scale / sqrt(N(s,a)+1)`` in deterministic batch order."""
        images = np.asarray(images)
        inventories = np.asarray(inventories, dtype=np.float32)
        actions = np.asarray(actions).reshape(-1)
        if images.ndim == 3:
            images = images[None]
        if inventories.ndim == 1:
            inventories = inventories[None]
        if not (len(images) == len(inventories) == len(actions)):
            raise ValueError("Count novelty batch lengths must match")
        if terminated is None:
            terminated = np.zeros(len(actions), dtype=np.bool_)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        if len(terminated) != len(actions):
            raise ValueError("terminated length must match count novelty batch")

        rewards = np.empty(len(actions), dtype=np.float32)
        for index, (image, inventory, action, is_terminal) in enumerate(
            zip(images, inventories, actions, terminated)
        ):
            key = self.state_action_key(image, inventory, int(action))
            visits = self._counts.get(key, 0)
            reward = self.reward_scale / np.sqrt(visits + 1.0)
            rewards[index] = 0.0 if bool(is_terminal) else reward
            if update:
                self._counts[key] = visits + 1
        return rewards

    @property
    def unique_state_actions(self) -> int:
        return len(self._counts)

    @property
    def total_visits(self) -> int:
        return int(sum(self._counts.values()))


def build_crafter_policy_state(images, inventories, obs_norm_values) -> torch.Tensor:
    """Build the existing full-map, 16-slot PPO observation without a WM."""
    images = np.asarray(images)
    inventories = np.asarray(inventories, dtype=np.float32)
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4:
        raise ValueError(f"Expected batched Crafter images, got {images.shape}")
    if images.shape[-1] == 2:
        images = np.moveaxis(images, -1, 1)
    if images.shape[1] != 2:
        raise ValueError(f"Expected two Crafter channels, got {images.shape}")
    if inventories.ndim == 1:
        inventories = inventories[None]
    if inventories.shape != (len(images), 16):
        raise ValueError(
            f"Expected Crafter inventory batch {(len(images), 16)}, got {inventories.shape}"
        )

    states = torch.as_tensor(images, dtype=torch.float32)
    norm_values = list(obs_norm_values)[:2]
    if len(norm_values) != 2:
        raise ValueError("Crafter policy normalization requires two channel values")
    for channel, value in enumerate(norm_values):
        if float(value) != 0.0:
            states[:, channel] /= float(value)
    inventory = torch.as_tensor(inventories, dtype=torch.float32).clamp_min(0.0) / 9.0
    return torch.cat([states.reshape(len(images), -1), inventory], dim=-1)
