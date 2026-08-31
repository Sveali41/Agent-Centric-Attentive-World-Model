"""Crafter transition targets, losses, decoding, and imagined dynamics.

Effect class 0 means KEEP.  Effect class ``k + 1`` means SET_TO category
``k``.  This keeps categorical state transitions semantic: category IDs are
never subtracted as though they were continuous values.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from modelBased.common import utils


SURVIVAL_SLOTS = 4
ITEM_SLOTS = 12
INVENTORY_SLOTS = SURVIVAL_SLOTS + ITEM_SLOTS
INVENTORY_VALUES = 10
CRAFTER_PLAYER_ID = 13


def categorical_effect_target(
    current: torch.Tensor,
    following: torch.Tensor,
) -> torch.Tensor:
    """Encode ``current -> following`` as KEEP/SET_TO categorical effects."""
    current_ids = current.long()
    following_ids = following.long()
    if current_ids.shape != following_ids.shape:
        raise ValueError(
            "Categorical effect states must have matching shapes, got "
            f"{tuple(current_ids.shape)} and {tuple(following_ids.shape)}"
        )
    changed = following_ids.ne(current_ids)
    return torch.where(
        changed,
        following_ids + 1,
        torch.zeros_like(following_ids),
    )


def balanced_categorical_effect_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "balanced_mean",
    label_smoothing: float = 0.0,
    normalize: bool = True,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Categorical effect loss with optional per-cell focal modulation.

    ``balanced_mean`` first averages each non-empty group (KEEP and SET_TO),
    then averages the group losses.  It therefore avoids a hand-tuned class
    weight while preventing unchanged cells from dominating sparse effects.

    When ``focal_gamma`` is positive, the focal factor is applied to each
    categorical cell before any reduction.  ``focal_gamma=0`` is exactly the
    normalized cross-entropy path used by the legacy implementation.
    """
    if logits.ndim != target.ndim + 1:
        raise ValueError(
            "Effect logits must add one class dimension to the target, got "
            f"{tuple(logits.shape)} and {tuple(target.shape)}"
        )
    target = target.long()
    if target.numel():
        target_min = int(target.min().item())
        target_max = int(target.max().item())
        if target_min < 0 or target_max >= logits.shape[1]:
            raise ValueError(
                f"Effect target range [{target_min}, {target_max}] exceeds "
                f"{logits.shape[1]} output classes"
            )

    per_cell = F.cross_entropy(
        logits,
        target,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    focal_gamma = float(focal_gamma)
    if focal_gamma < 0.0 or not math.isfinite(focal_gamma):
        raise ValueError(f"focal_gamma must be finite and non-negative, got {focal_gamma!r}")
    if focal_gamma > 0.0:
        probabilities = torch.softmax(logits, dim=1)
        p_true = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        p_true = p_true.clamp(min=torch.finfo(probabilities.dtype).eps, max=1.0)
        focal_factor = (1.0 - p_true).pow(focal_gamma)
        per_cell = per_cell * focal_factor
    scale = math.log(logits.shape[1]) if normalize else 1.0
    per_cell = per_cell / scale

    if reduction == "none":
        return per_cell
    if reduction == "mean":
        return per_cell.mean()
    if reduction != "balanced_mean":
        raise ValueError(f"Unsupported categorical-effect reduction: {reduction}")

    changed = target.ne(0)
    group_losses = []
    if (~changed).any():
        group_losses.append(per_cell[~changed].mean())
    if changed.any():
        group_losses.append(per_cell[changed].mean())
    if not group_losses:
        return per_cell.sum()
    return torch.stack(group_losses).mean()


def apply_categorical_effect(
    current: torch.Tensor,
    effect: torch.Tensor,
) -> torch.Tensor:
    """Apply decoded KEEP/SET_TO effect IDs to a categorical state tensor."""
    effect_ids = effect.long()
    if current.shape != effect_ids.shape:
        raise ValueError(
            "Categorical effect and state must have matching shapes, got "
            f"{tuple(current.shape)} and {tuple(effect_ids.shape)}"
        )
    changed = effect_ids.ne(0)
    set_values = (effect_ids - 1).to(dtype=current.dtype)
    return torch.where(changed, set_values, current)


def validate_discrete_inventory(values: torch.Tensor, name: str) -> None:
    """Validate one batch of Crafter's 0..9 integer inventory slots."""
    values_float = values.float()
    if not torch.equal(values_float, values_float.round()):
        raise ValueError(f"Crafter {name} inventory contains fractional values")
    if values_float.numel() and (
        values_float.min() < 0 or values_float.max() >= INVENTORY_VALUES
    ):
        raise ValueError(
            f"Crafter {name} inventory must be in [0, {INVENTORY_VALUES - 1}]"
        )


def decode_crafter_inventory(
    current: torch.Tensor,
    prediction: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Decode structured logits into an integer next inventory."""
    required = {"survival_effect_logits", "item_gate_logits", "item_value_logits"}
    if not isinstance(prediction, dict) or not required.issubset(prediction):
        raise ValueError(
            "Crafter inventory prediction must contain survival effect, item "
            "gate, and item value logits"
        )
    current_ids = current.long()
    if current_ids.ndim != 2 or current_ids.shape[1] != INVENTORY_SLOTS:
        raise ValueError(
            f"Crafter inventory must have shape (batch, {INVENTORY_SLOTS}), "
            f"got {tuple(current_ids.shape)}"
        )

    survival_effect = prediction["survival_effect_logits"].argmax(dim=1)
    survival = apply_categorical_effect(
        current_ids[:, :SURVIVAL_SLOTS], survival_effect
    )
    item_change = prediction["item_gate_logits"].argmax(dim=1).bool()
    item_value = prediction["item_value_logits"].argmax(dim=1)
    items = torch.where(
        item_change,
        item_value.to(current_ids.dtype),
        current_ids[:, SURVIVAL_SLOTS:],
    )
    return torch.cat((survival, items), dim=1).clamp(
        0, INVENTORY_VALUES - 1
    )


def crafter_inventory_gate_loss(
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    following: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute normalized survival, item-gate, and item-value losses."""
    validate_discrete_inventory(current, "current")
    validate_discrete_inventory(following, "next")
    current_ids, following_ids = current.long(), following.long()

    survival_target = categorical_effect_target(
        current_ids[:, :SURVIVAL_SLOTS],
        following_ids[:, :SURVIVAL_SLOTS],
    )
    survival_loss = balanced_categorical_effect_loss(
        prediction["survival_effect_logits"],
        survival_target,
        reduction="mean",
    )

    item_current = current_ids[:, SURVIVAL_SLOTS:]
    item_following = following_ids[:, SURVIVAL_SLOTS:]
    changed = item_following.ne(item_current)
    gate_per_slot = F.cross_entropy(
        prediction["item_gate_logits"], changed.long(), reduction="none"
    ) / math.log(2.0)
    gate_loss = gate_per_slot.mean()

    value_per_slot = F.cross_entropy(
        prediction["item_value_logits"], item_following, reduction="none"
    ) / math.log(INVENTORY_VALUES)
    if changed.any():
        value_loss = value_per_slot[changed].mean()
        components = [survival_loss, gate_loss, value_loss]
    else:
        value_loss = prediction["item_value_logits"].sum() * 0.0
        components = [survival_loss, gate_loss]

    return torch.stack(components).mean(), {
        "survival": survival_loss,
        "item_gate": gate_loss,
        "item_value": value_loss,
    }


def get_crafter_agent_position(states: torch.Tensor) -> torch.Tensor:
    """Return the first player position in each ``(B, 2, H, W)`` state."""
    obj = states[:, 0]
    batch, _, width = obj.shape
    flat = (obj == CRAFTER_PLAYER_ID).reshape(batch, -1).float()
    index = torch.argmax(flat, dim=1)
    return torch.stack((index // width, index % width), dim=1)


def restore_missing_crafter_players(
    next_states: torch.Tensor,
    current_states: torch.Tensor,
    fallback_positions: torch.Tensor,
) -> torch.Tensor:
    """Treat a missing imagined player token as a blocked movement."""
    has_player = (next_states[:, 0] == CRAFTER_PLAYER_ID).flatten(1).any(dim=1)
    missing = torch.nonzero(~has_player, as_tuple=False).reshape(-1)
    if missing.numel() == 0:
        return next_states
    repaired = next_states.clone()
    positions = fallback_positions[missing].long()
    y, x = positions[:, 0], positions[:, 1]
    repaired[missing, 0, y, x] = CRAFTER_PLAYER_ID
    repaired[missing, 1, y, x] = current_states[missing, 1, y, x]
    return repaired


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
    # Local import keeps domain visualization/support independent while this
    # module supplies the categorical primitives used by that support code.
    from domain.crafter.crafter_support import crafter_reconstruct_from_logits

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
    next_states = restore_missing_crafter_players(
        next_states, states, positions
    )
    if inv_pred is None:
        next_inventory = inventory.clone()
    elif inventory_target_mode == "categorical_gate":
        next_inventory = decode_crafter_inventory(inventory, inv_pred)
    else:
        raise ValueError(
            f"Unsupported Crafter inventory target mode: {inventory_target_mode}"
        )
    return next_states, next_inventory
