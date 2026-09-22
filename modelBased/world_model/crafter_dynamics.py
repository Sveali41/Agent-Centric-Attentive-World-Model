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
CRAFTER_POSE_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CRAFTER_POSE_DIRECTIONS = (1, 2, 3, 4)
# Item transitions in the current Crafter ruleset either acquire/craft one
# item, consume one item, place a table, or place a furnace.
ITEM_DELTA_VALUES = (-4, -2, -1, 1)
ITEM_DELTA_CLASSES = len(ITEM_DELTA_VALUES)
ITEM_EFFECT_CLASSES = ITEM_DELTA_CLASSES + 1  # KEEP plus the four signed deltas.

# The observed non-target DR inventory dynamics use these complete item-row
# effects.  Class zero is KEEP; remaining rows are stored in stable tuple
# order.  Keeping the table here gives training, validation and planning one
# exact semantic contract rather than reconstructing it from a dataset at
# load time.
CRAFTER_CANONICAL_EVENT_CODEBOOK = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (-1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (-1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1),
    (-1, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0),
    (-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


def crafter_event_codebook(*, device=None, dtype=torch.long) -> torch.Tensor:
    """Return the fixed 17-event inventory transition codebook."""
    return torch.tensor(CRAFTER_CANONICAL_EVENT_CODEBOOK, device=device, dtype=dtype)


def categorical_crafter_inventory_event_target(
    current_items: torch.Tensor, following_items: torch.Tensor,
    *, codebook: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode a full 12-slot item delta as one canonical event ID."""
    if current_items.shape != following_items.shape or current_items.shape[-1] != ITEM_SLOTS:
        raise ValueError("Crafter event target requires matching [B,12] item inventories")
    delta = following_items.long() - current_items.long()
    allowed = torch.as_tensor((0,) + ITEM_DELTA_VALUES, device=delta.device, dtype=delta.dtype)
    if not bool((delta.unsqueeze(-1).eq(allowed).any(dim=-1)).all()):
        raise ValueError("Crafter event target contains unsupported item delta")
    table = crafter_event_codebook(device=delta.device, dtype=delta.dtype) if codebook is None else codebook.to(delta)
    matches = delta.unsqueeze(1).eq(table.unsqueeze(0)).all(dim=2)
    if not bool(matches.any(dim=1).all()):
        unknown = sorted({tuple(row) for row in delta[~matches.any(dim=1)].detach().cpu().tolist()})
        raise ValueError(f"Crafter inventory event is not in canonical codebook: {unknown}")
    return matches.long().argmax(dim=1)


def crafter_event_scores_from_slot_logits(
    item_effect_logits: torch.Tensor, current_items: torch.Tensor,
    *, codebook: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score canonical events with factorised slot logits and mask invalid rows."""
    if item_effect_logits.ndim != 3 or item_effect_logits.shape[1:] != (ITEM_EFFECT_CLASSES, ITEM_SLOTS):
        raise ValueError("Crafter item-effect logits must have shape [B,5,12]")
    table = crafter_event_codebook(device=item_effect_logits.device) if codebook is None else codebook.to(item_effect_logits.device)
    if tuple(table.shape) != (len(CRAFTER_CANONICAL_EVENT_CODEBOOK), ITEM_SLOTS):
        raise ValueError("Crafter event codebook must have shape [17,12]")
    value_to_class = {0: 0, -4: 1, -2: 2, -1: 3, 1: 4}
    event_classes = torch.tensor(
        [[value_to_class[int(v)] for v in row] for row in table.detach().cpu().tolist()],
        device=item_effect_logits.device,
    )
    log_prob = item_effect_logits.log_softmax(dim=1)
    slots = torch.arange(ITEM_SLOTS, device=item_effect_logits.device)
    scores = torch.stack([log_prob[:, event_classes[event], slots].sum(dim=1)
                          for event in range(table.shape[0])], dim=1)
    candidates = current_items.long().unsqueeze(1) + table.long().unsqueeze(0)
    invalid = ((candidates < 0) | (candidates >= INVENTORY_VALUES)).any(dim=2)
    return scores.masked_fill(invalid, -torch.inf), invalid


def decode_crafter_event_scores(
    scores: torch.Tensor, current_items: torch.Tensor, *,
    codebook: torch.Tensor | None = None, change_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode valid canonical event scores into next item values and IDs."""
    table = crafter_event_codebook(device=scores.device) if codebook is None else codebook.to(scores.device)
    if scores.ndim != 2 or scores.shape[1] != table.shape[0]:
        raise ValueError("Crafter event scores must have one column per canonical event")
    calibrated = scores.clone()
    calibrated[:, 1:] += float(change_bias)
    event_ids = calibrated.argmax(dim=1)
    return (current_items.long() + table[event_ids]).clamp(0, INVENTORY_VALUES - 1), event_ids


def _crafter_unique_player_position(states: torch.Tensor, name: str) -> torch.Tensor:
    """Return [B,2] player coordinates, rejecting malformed pose labels."""
    if states.ndim != 4 or states.shape[1] < 2:
        raise ValueError(f"Crafter {name} state must have shape [B,>=2,H,W]")
    B, _, H, W = states.shape
    player = states[:, 0].long().eq(CRAFTER_PLAYER_ID).reshape(B, -1)
    if not bool(player.sum(dim=1).eq(1).all()):
        raise ValueError(f"Crafter pose target requires exactly one player in {name} state")
    index = player.float().argmax(dim=1)
    return torch.stack((torch.div(index, W, rounding_mode="floor"), index.remainder(W)), dim=1)


def crafter_pose_target(current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    """Encode observed relative position and next direction as a 20-way class."""
    current_pos = _crafter_unique_player_position(current, "current")
    following_pos = _crafter_unique_player_position(following, "following")
    delta = following_pos - current_pos
    allowed = torch.as_tensor(CRAFTER_POSE_DELTAS, device=delta.device, dtype=delta.dtype)
    matches = delta.unsqueeze(1).eq(allowed.unsqueeze(0)).all(dim=2)
    if not bool(matches.any(dim=1).all()):
        values = sorted({tuple(row) for row in delta[~matches.any(dim=1)].detach().cpu().tolist()})
        raise ValueError(f"Crafter pose transition has unsupported relative delta(s): {values}")
    delta_class = matches.long().argmax(dim=1)
    next_direction_map = following[:, 1].long().reshape(following.shape[0], -1)
    next_index = following_pos[:, 0] * following.shape[-1] + following_pos[:, 1]
    direction = next_direction_map.gather(1, next_index.unsqueeze(1)).squeeze(1)
    if bool(((direction < 1) | (direction > 4)).any()):
        values = sorted(set(direction[((direction < 1) | (direction > 4))].detach().cpu().tolist()))
        raise ValueError(f"Crafter pose target expects direction 1..4 at player, got {values}")
    return delta_class * len(CRAFTER_POSE_DIRECTIONS) + (direction - 1)


def decode_crafter_pose_logits(
    logits: torch.Tensor, *, sample_mode: str = "mode", generator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode 20 joint pose logits into [B,2] delta and [B] direction."""
    if logits.ndim != 2 or logits.shape[1] != 20:
        raise ValueError(f"Crafter pose logits must have shape [B,20], got {tuple(logits.shape)}")
    if sample_mode not in {"mode", "sample"}:
        raise ValueError("Crafter pose sample_mode must be 'mode' or 'sample'")
    if sample_mode == "mode":
        index = logits.argmax(dim=1)
    else:
        index = torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).squeeze(1)
    delta_index = torch.div(index, len(CRAFTER_POSE_DIRECTIONS), rounding_mode="floor")
    direction = index.remainder(len(CRAFTER_POSE_DIRECTIONS)) + 1
    deltas = torch.as_tensor(CRAFTER_POSE_DELTAS, device=logits.device, dtype=torch.long)
    return deltas[delta_index], direction


def apply_crafter_pose_projection(
    decoded: torch.Tensor,
    current: torch.Tensor,
    pose_logits: torch.Tensor,
    *, sample_mode: str = "mode", generator=None,
    global_position: torch.Tensor | None = None,
    global_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Tie the predicted agent and direction to one learned joint pose.

    When a local WM patch is centered at a map boundary, a predicted move can
    be valid in patch coordinates while its global destination is outside the
    map.  Such moves are projected to a blocked (stay-in-place) transition.
    """
    current_pos = _crafter_unique_player_position(current, "current")
    delta, direction = decode_crafter_pose_logits(
        pose_logits, sample_mode=sample_mode, generator=generator,
    )
    next_pos = current_pos + delta
    B, _, H, W = decoded.shape
    valid_local = (
        (next_pos[:, 0] >= 0) & (next_pos[:, 0] < H)
        & (next_pos[:, 1] >= 0) & (next_pos[:, 1] < W)
    )
    if global_position is not None or global_shape is not None:
        if global_position is None or global_shape is None:
            raise ValueError("Crafter pose projection requires both global_position and global_shape")
        global_position = global_position.to(next_pos.device).long().reshape(B, 2)
        global_rows, global_cols = (int(global_shape[0]), int(global_shape[1]))
        global_next = global_position + delta
        valid_global = (
            (global_next[:, 0] >= 0) & (global_next[:, 0] < global_rows)
            & (global_next[:, 1] >= 0) & (global_next[:, 1] < global_cols)
        )
        valid = valid_local & valid_global
    else:
        valid = valid_local
    # Keep the current local position for a move beyond the global map edge.
    # This is a blocked movement, rather than an invalid imagined state.
    next_pos = torch.where(valid[:, None], next_pos, current_pos)
    projected = decoded.clone()
    # Preserve directions for every non-player entity.  Raw spatial decoding
    # can contain no/multiple players, so clear only the current player, any
    # raw predicted player candidates, and the final learned-pose destination.
    raw_predicted_players = projected[:, 0].eq(CRAFTER_PLAYER_ID)
    current_players = current[:, 0].long().eq(CRAFTER_PLAYER_ID)
    pose_related = raw_predicted_players | current_players
    projected[:, 0][raw_predicted_players] = 0
    projected[:, 1][pose_related] = 0
    rows = torch.arange(B, device=decoded.device)
    projected[rows, 1, next_pos[:, 0], next_pos[:, 1]] = 0
    projected[rows, 0, next_pos[:, 0], next_pos[:, 1]] = CRAFTER_PLAYER_ID
    projected[rows, 1, next_pos[:, 0], next_pos[:, 1]] = direction.to(projected.dtype)
    return projected


def _validate_inventory_value_mode(value_mode: str) -> str:
    value_mode = str(value_mode).strip().lower()
    if value_mode not in {"categorical_absolute", "categorical_delta"}:
        raise ValueError(
            "Crafter inventory value_mode must be 'categorical_absolute' or "
            f"'categorical_delta', got {value_mode!r}"
        )
    return value_mode


def categorical_inventory_delta_target(
    current_items: torch.Tensor,
    following_items: torch.Tensor,
) -> torch.Tensor:
    """Map changed Crafter item values to the four supported delta classes.

    The caller applies these labels only where the inventory gate's true
    target is CHANGE.  Rejecting an unknown non-zero delta prevents a future
    environment-rule change from silently producing a wrong training label.
    """
    delta = following_items.long() - current_items.long()
    changed = delta.ne(0)
    allowed = torch.as_tensor(ITEM_DELTA_VALUES, device=delta.device, dtype=delta.dtype)
    supported = (delta.unsqueeze(-1) == allowed).any(dim=-1)
    invalid = changed & ~supported
    if invalid.any():
        values = sorted(set(delta[invalid].detach().cpu().tolist()))
        raise ValueError(
            "Crafter item transition contains unsupported non-zero inventory "
            f"delta(s) {values}; expected one of {list(ITEM_DELTA_VALUES)}"
        )
    target = torch.zeros_like(delta)
    for class_id, delta_value in enumerate(ITEM_DELTA_VALUES):
        target = torch.where(delta.eq(delta_value), torch.full_like(target, class_id), target)
    return target


def categorical_inventory_effect_target(
    current_items: torch.Tensor,
    following_items: torch.Tensor,
) -> torch.Tensor:
    """Encode item transitions as ``KEEP, -4, -2, -1, +1`` classes.

    Class zero is deliberately KEEP.  The remaining class IDs are the
    corresponding ``ITEM_DELTA_VALUES`` in their declared order.  This is a
    unified replacement for the *new* effect-head mode only; the legacy
    gate/value targets above remain unchanged for checkpoint compatibility.
    """
    delta_class = categorical_inventory_delta_target(current_items, following_items)
    changed = following_items.long().ne(current_items.long())
    return torch.where(changed, delta_class + 1, torch.zeros_like(delta_class))


def decode_crafter_item_effects(
    current_items: torch.Tensor,
    item_effect_logits: torch.Tensor,
) -> torch.Tensor:
    """Decode five-way ``KEEP, -4, -2, -1, +1`` item effects."""
    if item_effect_logits.ndim != 3 or item_effect_logits.shape[1] != ITEM_EFFECT_CLASSES:
        raise ValueError(
            "Crafter item-effect logits must have shape (batch, "
            f"{ITEM_EFFECT_CLASSES}, 12), got {tuple(item_effect_logits.shape)}"
        )
    effect = item_effect_logits.argmax(dim=1)
    delta_values = torch.as_tensor(
        (0,) + ITEM_DELTA_VALUES,
        device=item_effect_logits.device,
        dtype=current_items.dtype,
    )
    return (current_items + delta_values[effect]).clamp(0, INVENTORY_VALUES - 1)


def decode_crafter_item_values(
    current_items: torch.Tensor,
    item_value_logits: torch.Tensor,
    *,
    value_mode: str = "categorical_absolute",
) -> torch.Tensor:
    """Decode an item-value head into absolute next inventory values."""
    value_mode = _validate_inventory_value_mode(value_mode)
    if value_mode == "categorical_absolute":
        return item_value_logits.argmax(dim=1).to(dtype=current_items.dtype)
    expected_shape = (ITEM_DELTA_CLASSES,)
    if item_value_logits.shape[1] != ITEM_DELTA_CLASSES:
        raise ValueError(
            "Crafter delta value logits must have "
            f"{ITEM_DELTA_CLASSES} classes {expected_shape}, got shape "
            f"{tuple(item_value_logits.shape)}"
        )
    predicted_class = item_value_logits.argmax(dim=1)
    delta_values = torch.as_tensor(
        ITEM_DELTA_VALUES, device=item_value_logits.device, dtype=current_items.dtype
    )
    return (current_items + delta_values[predicted_class]).clamp(
        0, INVENTORY_VALUES - 1
    )


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

    ``sqrt_balanced`` also averages each non-empty group, but weights the
    resulting group means by the square root of that group's sample count.
    This provides a soft transition between equal group weighting and the
    natural per-cell mean.  For example, groups with counts 1 and 4 receive
    weights 1:2.

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
    if reduction not in {"balanced_mean", "sqrt_balanced"}:
        raise ValueError(f"Unsupported categorical-effect reduction: {reduction}")

    changed = target.ne(0)
    group_losses = []
    group_weights = []
    if (~changed).any():
        group_losses.append(per_cell[~changed].mean())
        group_weights.append(math.sqrt(float((~changed).sum().item())))
    if changed.any():
        group_losses.append(per_cell[changed].mean())
        group_weights.append(math.sqrt(float(changed.sum().item())))
    if not group_losses:
        return per_cell.sum()
    losses = torch.stack(group_losses)
    if reduction == "balanced_mean":
        return losses.mean()
    weights = torch.as_tensor(
        group_weights, device=losses.device, dtype=losses.dtype
    )
    return (losses * weights).sum() / weights.sum()


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
    *,
    predict_survival: bool = True,
    output_mode: str = "categorical_gate",
    value_mode: str = "categorical_absolute",
) -> torch.Tensor:
    """Decode structured logits into an integer next inventory.

    When ``predict_survival`` is false, the four tracker-owned survival slots
    are carried forward unchanged.  The survival head remains in ``prediction``
    so checkpoints keep the same tensor layout.
    """
    output_mode = str(output_mode).strip().lower()
    if output_mode not in {"categorical_gate", "categorical_effect"}:
        raise ValueError(f"Unsupported Crafter inventory output mode: {output_mode!r}")
    required = {"survival_effect_logits"}
    if output_mode == "categorical_gate":
        required |= {"item_gate_logits", "item_value_logits"}
    else:
        required.add("item_effect_logits")
    if not isinstance(prediction, dict) or not required.issubset(prediction):
        raise ValueError(
            f"Crafter {output_mode} inventory prediction is missing required "
            f"logits: {sorted(required - set(prediction or {}))}"
        )
    current_ids = current.long()
    if current_ids.ndim != 2 or current_ids.shape[1] != INVENTORY_SLOTS:
        raise ValueError(
            f"Crafter inventory must have shape (batch, {INVENTORY_SLOTS}), "
            f"got {tuple(current_ids.shape)}"
        )

    if predict_survival:
        survival_effect = prediction["survival_effect_logits"].argmax(dim=1)
        survival = apply_categorical_effect(
            current_ids[:, :SURVIVAL_SLOTS], survival_effect
        )
    else:
        survival = current_ids[:, :SURVIVAL_SLOTS]
    if output_mode == "categorical_gate":
        item_change = prediction["item_gate_logits"].argmax(dim=1).bool()
        item_value = decode_crafter_item_values(
            current_ids[:, SURVIVAL_SLOTS:],
            prediction["item_value_logits"],
            value_mode=value_mode,
        )
        items = torch.where(
            item_change,
            item_value.to(current_ids.dtype),
            current_ids[:, SURVIVAL_SLOTS:],
        )
    else:
        if "item_event_scores" in prediction:
            items, _ = decode_crafter_event_scores(
                prediction["item_event_scores"], current_ids[:, SURVIVAL_SLOTS:],
                codebook=prediction.get("item_event_codebook"),
                change_bias=float(prediction.get("item_event_change_bias", 0.0)),
            )
        else:
            items = decode_crafter_item_effects(
                current_ids[:, SURVIVAL_SLOTS:], prediction["item_effect_logits"]
            )
    return torch.cat((survival, items), dim=1).clamp(
        0, INVENTORY_VALUES - 1
    )


def crafter_inventory_gate_loss(
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    following: torch.Tensor,
    *,
    predict_survival: bool = True,
    value_mode: str = "categorical_absolute",
    gate_reduction: str = "global_balanced",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute normalized Crafter inventory losses.

    ``predict_survival=False`` keeps the legacy head present but removes its
    supervision.  Crafter's rule tracker then owns slots 0--3 during imagined
    rollout.
    """
    validate_discrete_inventory(current, "current")
    validate_discrete_inventory(following, "next")
    current_ids, following_ids = current.long(), following.long()

    if predict_survival:
        survival_target = categorical_effect_target(
            current_ids[:, :SURVIVAL_SLOTS],
            following_ids[:, :SURVIVAL_SLOTS],
        )
        survival_loss = balanced_categorical_effect_loss(
            prediction["survival_effect_logits"],
            survival_target,
            reduction="balanced_mean",
        )
    else:
        # Retain a stable zero-valued diagnostic without supervising the head.
        survival_loss = prediction["survival_effect_logits"].sum() * 0.0

    item_current = current_ids[:, SURVIVAL_SLOTS:]
    item_following = following_ids[:, SURVIVAL_SLOTS:]
    changed = item_following.ne(item_current)
    gate_per_slot = F.cross_entropy(
        prediction["item_gate_logits"], changed.long(), reduction="none"
    ) / math.log(2.0)
    # Item changes are exceptionally sparse in naturally collected Crafter
    # trajectories.  Averaging every slot makes an all-KEEP gate a trivial
    # optimum, even when the value head receives useful changed-item signal.
    # Average the two observed groups equally instead; no fixed class weight
    # or data resampling is introduced here.
    if gate_reduction not in {"global_balanced", "slot_macro_changed"}:
        raise ValueError(
            "crafter_inventory_gate_reduction must be 'global_balanced' or "
            f"'slot_macro_changed', got {gate_reduction!r}"
        )
    keep_loss = gate_per_slot[~changed].mean() if (~changed).any() else None
    if changed.any():
        if gate_reduction == "slot_macro_changed":
            # Every observed item slot receives one equal CHANGE contribution.
            # KEEP remains globally averaged so ordinary non-change calibration
            # remains natural-distribution based.
            changed_loss = torch.stack([
                gate_per_slot[:, slot][changed[:, slot]].mean()
                for slot in range(changed.shape[1]) if changed[:, slot].any()
            ]).mean()
        else:
            changed_loss = gate_per_slot[changed].mean()
    else:
        changed_loss = None
    if keep_loss is not None and changed_loss is not None:
        gate_loss = 0.5 * (keep_loss + changed_loss)
    elif keep_loss is not None:
        gate_loss = keep_loss
    elif changed_loss is not None:
        gate_loss = changed_loss
    else:
        gate_loss = gate_per_slot.sum()

    value_mode = _validate_inventory_value_mode(value_mode)
    if value_mode == "categorical_absolute":
        value_target = item_following
        value_normalizer = math.log(INVENTORY_VALUES)
    else:
        value_target = categorical_inventory_delta_target(item_current, item_following)
        value_normalizer = math.log(ITEM_DELTA_CLASSES)
    value_per_slot = F.cross_entropy(
        prediction["item_value_logits"], value_target, reduction="none"
    ) / value_normalizer
    if changed.any():
        # Both retained representations use the historical changed-entry
        # mean.  This keeps absolute checkpoints semantically compatible.
        value_loss = value_per_slot[changed].mean()
        components = ([survival_loss] if predict_survival else []) + [gate_loss, value_loss]
    else:
        value_loss = prediction["item_value_logits"].sum() * 0.0
        components = ([survival_loss] if predict_survival else []) + [gate_loss]

    return torch.stack(components).mean(), {
        "survival": survival_loss,
        "item_gate": gate_loss,
        "item_value": value_loss,
    }


def crafter_inventory_effect_loss(
    prediction: dict[str, torch.Tensor],
    current: torch.Tensor,
    following: torch.Tensor,
    *,
    predict_survival: bool = True,
    effect_reduction: str = "balanced_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss for the unified five-way Crafter item-effect head.

    ``effect_reduction`` is explicit because it is a research choice.  The
    default retains the prior gate objective's KEEP-vs-any-change balancing;
    no prior correction or per-item weighting is implicit in this mode.
    """
    validate_discrete_inventory(current, "current")
    validate_discrete_inventory(following, "next")
    required = {"survival_effect_logits", "item_effect_logits"}
    if not required.issubset(prediction):
        raise ValueError("Crafter item-effect prediction is missing required logits")
    current_ids, following_ids = current.long(), following.long()
    if predict_survival:
        survival_target = categorical_effect_target(
            current_ids[:, :SURVIVAL_SLOTS], following_ids[:, :SURVIVAL_SLOTS]
        )
        survival_loss = balanced_categorical_effect_loss(
            prediction["survival_effect_logits"], survival_target,
            reduction="balanced_mean",
        )
    else:
        survival_loss = prediction["survival_effect_logits"].sum() * 0.0
    item_target = categorical_inventory_effect_target(
        current_ids[:, SURVIVAL_SLOTS:], following_ids[:, SURVIVAL_SLOTS:]
    )
    item_loss = balanced_categorical_effect_loss(
        prediction["item_effect_logits"], item_target, reduction=effect_reduction,
    )
    components = ([survival_loss] if predict_survival else []) + [item_loss]
    return torch.stack(components).mean(), {
        "survival": survival_loss,
        "item_effect": item_loss,
    }


def crafter_inventory_event_residual_loss(
    prediction: dict[str, torch.Tensor], current: torch.Tensor, following: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Residual event CE on true changes plus base-model false-positive KEEPs."""
    required = {"item_event_base_scores", "item_event_residual_logits", "item_event_codebook"}
    if not required.issubset(prediction):
        raise ValueError("Crafter event residual prediction is missing required tensors")
    current_items = current.long()[:, SURVIVAL_SLOTS:]
    following_items = following.long()[:, SURVIVAL_SLOTS:]
    target = categorical_crafter_inventory_event_target(
        current_items, following_items, codebook=prediction["item_event_codebook"]
    )
    base_scores = prediction["item_event_base_scores"]
    base_event = base_scores.argmax(dim=1)
    hard_rows = target.ne(0) | (target.eq(0) & base_event.ne(0))
    residual_logits = prediction["item_event_residual_logits"]
    logits = base_scores + residual_logits
    if bool(hard_rows.any()):
        loss = F.cross_entropy(logits[hard_rows], target[hard_rows]) / math.log(float(logits.shape[1]))
    else:
        loss = residual_logits.sum() * 0.0
    return loss, {
        "event_residual": loss,
        "event_residual_hard_rows": hard_rows.sum().to(dtype=loss.dtype),
        "event_residual_true_change_rows": target.ne(0).sum().to(dtype=loss.dtype),
        "event_residual_false_positive_keep_rows": (target.eq(0) & base_event.ne(0)).sum().to(dtype=loss.dtype),
    }


def get_crafter_agent_position(states: torch.Tensor) -> torch.Tensor:
    """Return the first player position in each ``(B, 2, H, W)`` state."""
    obj = states[:, 0]
    batch, _, width = obj.shape
    flat = (obj == CRAFTER_PLAYER_ID).reshape(batch, -1).float()
    index = torch.argmax(flat, dim=1)
    return torch.stack((index // width, index % width), dim=1)


def crafter_player_counts(states: torch.Tensor) -> torch.Tensor:
    """Return the number of player tokens in each full Crafter state."""
    if states.ndim != 4 or states.shape[1] < 1:
        raise ValueError("Crafter states must have shape [B,>=1,H,W]")
    return states[:, 0].eq(CRAFTER_PLAYER_ID).flatten(1).sum(dim=1)


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
    *,
    predict_survival: bool = True,
    inventory_value_mode: str = "categorical_absolute",
    pose_sample_mode: str = "mode",
    pose_generator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance symbolic Crafter states through a frozen world model."""
    # Local import keeps domain visualization/support independent while this
    # module supplies the categorical primitives used by that support code.
    from domain.crafter.crafter_support import crafter_reconstruct_from_logits

    positions = get_crafter_agent_position(states)
    # Unlike MiniGrid, a Crafter object at (0, 0) may be the player.  Border
    # padding must therefore be neutral rather than a copy of that cell.
    masked = utils.extract_masked_state_torch(
        states, attention_mask_size, positions, pad_value=0
    )
    pose_enabled = bool(getattr(model, "crafter_pose_enabled", False))
    if pose_enabled:
        wm_out, _, inv_pred, pose_logits = model.forward_pose(
            masked.float(), actions, None, inv=inventory.float()
        )
    else:
        wm_out, _, inv_pred = model(
            masked.float(), actions, None, inv=inventory.float()
        )
    # Planning requires a valid symbolic state.  Project the decoded object
    # map onto Crafter's exactly-one-player invariant; validation and training
    # continue to use the raw decoder by default.
    predicted = crafter_reconstruct_from_logits(
        wm_out,
        current=masked,
        # The pose head owns the only player token.  Decode non-player
        # classes first so a departing agent cell retains WM terrain instead
        # of being hard-coded to empty.
        suppress_agent=pose_enabled,
        constrain_agent=not pose_enabled,
    )
    if pose_enabled:
        predicted = apply_crafter_pose_projection(
            predicted, masked, pose_logits,
            sample_mode=pose_sample_mode, generator=pose_generator,
            global_position=positions,
            global_shape=states.shape[-2:],
        )
    next_states = utils.put_back_masked_state_torch(
        predicted, states, attention_mask_size, positions
    )
    next_states = restore_missing_crafter_players(
        next_states, states, positions
    )
    if inv_pred is None:
        next_inventory = inventory.clone()
    elif inventory_target_mode in {"categorical_gate", "categorical_effect"}:
        next_inventory = decode_crafter_inventory(
            inventory, inv_pred, predict_survival=predict_survival,
            output_mode=inventory_target_mode,
            value_mode=inventory_value_mode,
        )
    else:
        raise ValueError(
            f"Unsupported Crafter inventory target mode: {inventory_target_mode}"
        )
    return next_states, next_inventory
