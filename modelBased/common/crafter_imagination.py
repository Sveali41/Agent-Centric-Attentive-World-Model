"""Pure Crafter world-model stepping helpers shared by planning and P2E."""

from __future__ import annotations

import torch

from modelBased.common import utils
from domain.crafter.crafter_support import crafter_reconstruct_from_logits

CRAFTER_PLAYER_ID = 13


def get_crafter_agent_position(states: torch.Tensor) -> torch.Tensor:
    """Return the first player position in each ``(B, 2, H, W)`` state."""
    obj = states[:, 0]
    batch, height, width = obj.shape
    flat = (obj == CRAFTER_PLAYER_ID).reshape(batch, -1).float()
    index = torch.argmax(flat, dim=1)
    return torch.stack((index // width, index % width), dim=1)


@torch.no_grad()
def imagined_crafter_step_batch(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    inventory: torch.Tensor,
    attention_mask_size: int,
    inventory_target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance symbolic Crafter states through a frozen world model."""
    positions = get_crafter_agent_position(states)
    masked = utils.extract_masked_state_torch(
        states, attention_mask_size, positions
    )
    wm_out, _, inv_pred = model(
        masked.float(), actions, None, inv=inventory.float()
    )
    predicted = crafter_reconstruct_from_logits(wm_out, current=masked)
    next_states = utils.put_back_masked_state_torch(
        predicted, states, attention_mask_size, positions
    )
    if inv_pred is None:
        next_inventory = inventory.clone()
    elif inventory_target_mode == "delta":
        next_inventory = (inventory.float() + inv_pred.float()).clamp(0.0, 9.0)
    elif inventory_target_mode == "absolute":
        next_inventory = inv_pred.float()
    else:
        raise ValueError(
            f"Unsupported Crafter inventory target mode: {inventory_target_mode}"
        )
    return next_states, next_inventory
