"""Shared learned MiniGrid transition used by PPO, MPC, and diagnostics."""

from __future__ import annotations

import math

import torch

from domain.minigrid.transition_codec import (
    MINIGRID_FIELDS,
    decode_minigrid_transition,
    minigrid_contract,
)
from modelBased.common import utils


def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return one normalized categorical entropy value for each batch item."""
    classes = logits.shape[1]
    if classes < 2:
        return torch.zeros(logits.shape[0], device=logits.device)
    probabilities = logits.softmax(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy.reshape(entropy.shape[0], -1).mean(dim=1) / math.log(float(classes))


def rollout_minigrid_wm(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    inventory_tokens: torch.Tensor,
    attention_mask_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Advance a batch of full MiniGrid states using only the learned WM.

    ``states`` is the decoded full symbolic map while the WM itself receives
    the agent-centred patch.  The function deliberately owns that boundary so
    every imagined consumer uses the same decoder and inventory contract.
    """
    if states.ndim != 4 or states.shape[1] != 3:
        raise ValueError(f"Expected states [B,3,H,W], got {tuple(states.shape)}")
    batch_size = states.shape[0]
    actions = torch.as_tensor(actions, device=states.device, dtype=torch.long).reshape(-1)
    inventory_tokens = torch.as_tensor(
        inventory_tokens, device=states.device, dtype=torch.long
    ).reshape(-1)
    if len(actions) != batch_size or len(inventory_tokens) != batch_size:
        raise ValueError("actions and inventory_tokens must have one value per state")

    agent_positions = utils.get_agent_position_torch(states)
    masked = utils.extract_masked_state_torch(
        states, int(attention_mask_size), agent_positions
    )
    prediction, _, inventory_logits = model(masked, actions, None, inv=inventory_tokens)
    next_masked, next_inventory, decoder_diagnostics = decode_minigrid_transition(
        prediction,
        masked,
        inventory_logits,
        inventory_tokens,
        mode=getattr(model, "minigrid_transition_mode", "effect"),
        constrain_agent=True,
    )
    if next_inventory is None:
        raise RuntimeError("MiniGrid WM planning requires inventory logits")
    next_states = utils.put_back_masked_state_torch(
        next_masked, states, int(attention_mask_size), agent_positions
    )

    contract = minigrid_contract(getattr(model, "minigrid_transition_mode", "effect"))
    spatial_entropy = []
    for name, _, _ in MINIGRID_FIELDS:
        start, stop = contract.spatial_slices[name]
        spatial_entropy.append(_normalized_entropy(prediction[:, start:stop]))
    uncertainty = torch.stack(spatial_entropy, dim=0).mean(dim=0)
    if inventory_logits is not None:
        uncertainty = (uncertainty + _normalized_entropy(inventory_logits)) / 2.0

    diagnostics = dict(decoder_diagnostics)
    diagnostics["uncertainty"] = uncertainty
    diagnostics["agent_positions_before"] = agent_positions
    diagnostics["agent_positions_after"] = utils.get_agent_position_torch(next_states)
    return next_states, next_inventory.long(), diagnostics
