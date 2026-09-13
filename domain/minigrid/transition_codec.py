"""Canonical MiniGrid transition representation.

The environment and datasets store absolute symbolic frames.  This module is
the single boundary where those frames are represented as either ordinary
next-state classes or KEEP/SET_TO effects.  Training and all imagined-state
consumers must use these helpers so that the class counts and slices cannot
drift between entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from collections.abc import Mapping

import torch
import torch.nn.functional as F


MINIGRID_FIELDS = (
    ("object", 0, 11),
    ("color", 1, 6),
    ("state", 2, 4),
)
MINIGRID_INVENTORY_CLASSES = 7  # empty + six key colours
MINIGRID_ACTION_COUNT = 6


def _categorical_effect_target(
    current: torch.Tensor,
    following: torch.Tensor,
) -> torch.Tensor:
    """Encode an absolute MiniGrid pair as KEEP/SET_TO labels.

    This small implementation intentionally lives in the MiniGrid codec rather
    than importing the Crafter dynamics module.  The latter imports the full
    project environment configuration at module import time, which would make
    a standalone dataset/codec test depend on unrelated environment variables.
    """
    current_ids = current.long()
    following_ids = following.long()
    if current_ids.shape != following_ids.shape:
        raise ValueError(
            "MiniGrid effect states must have matching shapes, got "
            f"{tuple(current_ids.shape)} and {tuple(following_ids.shape)}"
        )
    changed = following_ids.ne(current_ids)
    return torch.where(
        changed,
        following_ids + 1,
        torch.zeros_like(following_ids),
    )


@dataclass(frozen=True)
class MiniGridTransitionContract:
    mode: str
    spatial_slices: dict[str, tuple[int, int]]
    spatial_classes: dict[str, int]
    inventory_classes: int
    inventory_output_classes: int
    output_channels: int
    action_count: int = MINIGRID_ACTION_COUNT


def _validate_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "effect"}:
        raise ValueError(
            "minigrid_transition_mode must be 'absolute' or 'effect', "
            f"got {mode!r}"
        )
    return mode


def minigrid_contract(mode: str = "effect") -> MiniGridTransitionContract:
    mode = _validate_mode(mode)
    start = 0
    slices: dict[str, tuple[int, int]] = {}
    for name, _, classes in MINIGRID_FIELDS:
        width = classes + 1 if mode == "effect" else classes
        slices[name] = (start, start + width)
        start += width
    return MiniGridTransitionContract(
        mode=mode,
        spatial_slices=slices,
        spatial_classes={name: classes for name, _, classes in MINIGRID_FIELDS},
        inventory_classes=MINIGRID_INVENTORY_CLASSES,
        inventory_output_classes=(MINIGRID_INVENTORY_CLASSES + 1 if mode == "effect" else MINIGRID_INVENTORY_CLASSES),
        output_channels=start,
    )


def canonical_minigrid_schema(
    mode: str = "effect",
    effect_reduction: str = "balanced_mean",
) -> list[dict]:
    """Return the schema used by every MiniGrid training entry point."""
    contract = minigrid_contract(mode)
    effect_reduction = str(effect_reduction).strip().lower()
    if effect_reduction not in {"mean", "balanced_mean"}:
        raise ValueError(
            "MiniGrid effect_reduction must be 'mean' or 'balanced_mean', "
            f"got {effect_reduction!r}"
        )
    schema = []
    distribution = "categorical_effect" if contract.mode == "effect" else "categorical"
    target_mode = "categorical_effect" if contract.mode == "effect" else "absolute"
    for name, target_index, classes in MINIGRID_FIELDS:
        start, stop = contract.spatial_slices[name]
        field = {
            "name": name,
            "distribution": distribution,
            "prediction_slice": [start, stop],
            "target_index": target_index,
            "target_mode": target_mode,
            "classes": classes,
            "label_smoothing": 0.0,
        }
        if contract.mode == "effect":
            field["effect_reduction"] = effect_reduction
        schema.append(field)
    inventory = {
        "name": "inventory",
        "distribution": distribution,
        "prediction_source": "auxiliary",
        "target_source": "inventory",
        "target_mode": target_mode,
        "classes": MINIGRID_INVENTORY_CLASSES,
        "label_smoothing": 0.0,
    }
    if contract.mode == "effect":
        inventory["effect_reduction"] = effect_reduction
    schema.append(inventory)
    return schema


def validate_schema(
    schema: Iterable,
    mode: str,
    effect_reduction: str = "balanced_mean",
) -> None:
    """Reject hand-written MiniGrid schemas that do not match the contract."""
    expected = canonical_minigrid_schema(mode, effect_reduction)
    def plain(value):
        if isinstance(value, Mapping) or hasattr(value, "items"):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)) or (
            not isinstance(value, (str, bytes))
            and hasattr(value, "__iter__")
            and not torch.is_tensor(value)
        ):
            return [plain(v) for v in value]
        return value

    actual = []
    for field in schema or []:
        actual.append(plain(field))
    expected = plain(expected)
    if actual != expected:
        raise ValueError(
            "MiniGrid observation_schema is not aligned with "
            f"minigrid_transition_mode={mode!r}. Use the canonical schema; "
            f"expected={expected}, got={actual}"
        )


def effect_target(current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    """Encode absolute categorical states as KEEP/SET_TO labels."""
    current = current.long()
    following = following.long()
    if current.shape != following.shape:
        raise ValueError(
            f"MiniGrid effect shapes differ: {tuple(current.shape)} vs "
            f"{tuple(following.shape)}"
        )
    return _categorical_effect_target(current, following)


def _next_class_probabilities(effect_logits: torch.Tensor, current: torch.Tensor, classes: int) -> torch.Tensor:
    """Map effect probabilities to probabilities over absolute next classes."""
    current = current.long().clamp(0, classes - 1)
    probs = effect_logits.softmax(dim=1)
    next_probs = probs[:, 1 : classes + 1].clone()
    keep = probs[:, 0:1]
    next_probs.scatter_add_(1, current.unsqueeze(1), keep)
    return next_probs


def _decode_object_with_single_agent(
    logits: torch.Tensor,
    current: torch.Tensor,
    constrain_agent: bool,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probs = _next_class_probabilities(logits, current, 11)
    if not constrain_agent:
        decoded = probs.argmax(dim=1)
        if not collect_diagnostics:
            return decoded, {}
        raw_counts = (decoded == 10).flatten(1).sum(dim=1)
        return decoded, {
            "raw_agent_count": raw_counts,
            "agent_correction": torch.zeros_like(raw_counts, dtype=torch.bool),
        }

    decoded = probs.argmax(dim=1) if collect_diagnostics else None
    flat = probs.flatten(2)
    agent_score = flat[:, 10]
    nonagent = flat.clone()
    nonagent[:, 10] = 0.0
    best_nonagent_prob, best_nonagent = nonagent.max(dim=1)
    # The optimal exactly-one-agent assignment maximizes the selected agent
    # log-probability plus all other cells' best non-agent log-probability.
    choice_score = torch.log(agent_score.clamp_min(1e-12)) - torch.log(
        best_nonagent_prob.clamp_min(1e-12)
    )
    chosen = choice_score.argmax(dim=1)
    decoded_flat = nonagent.argmax(dim=1)
    decoded_flat.scatter_(1, chosen.unsqueeze(1), torch.full_like(chosen.unsqueeze(1), 10))
    constrained = decoded_flat.reshape_as(current)
    if not collect_diagnostics:
        return constrained, {}
    raw_counts = (decoded == 10).flatten(1).sum(dim=1)
    return constrained, {
        "raw_agent_count": raw_counts,
        "agent_correction": constrained.ne(decoded).flatten(1).any(dim=1),
        "chosen_agent_index": chosen,
    }


def _constrain_absolute_object(
    logits: torch.Tensor,
    decoded: torch.Tensor,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project absolute object probabilities onto the one-agent invariant."""
    probs = logits.softmax(dim=1)
    flat = probs.flatten(2)
    agent_prob = flat[:, 10]
    nonagent = flat.clone()
    nonagent[:, 10] = 0.0
    best_nonagent_prob, _ = nonagent.max(dim=1)
    choice_score = torch.log(agent_prob.clamp_min(1e-12)) - torch.log(
        best_nonagent_prob.clamp_min(1e-12)
    )
    chosen = choice_score.argmax(dim=1)
    decoded_flat = nonagent.argmax(dim=1)
    decoded_flat.scatter_(1, chosen.unsqueeze(1), torch.full_like(chosen.unsqueeze(1), 10))
    constrained = decoded_flat.reshape_as(decoded)
    if not collect_diagnostics:
        return constrained, {}
    raw_counts = (decoded == 10).flatten(1).sum(dim=1)
    return constrained, {
        "raw_agent_count": raw_counts,
        "agent_correction": constrained.ne(decoded).flatten(1).any(dim=1),
        "chosen_agent_index": chosen,
    }


def decode_minigrid_transition(
    spatial_logits: torch.Tensor,
    current_state: torch.Tensor,
    inventory_logits: torch.Tensor | None,
    current_inventory: torch.Tensor | None,
    mode: str = "effect",
    constrain_agent: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Decode a MiniGrid model output into absolute next state values."""
    contract = minigrid_contract(mode)
    if spatial_logits.ndim != 4 or spatial_logits.shape[1] != contract.output_channels:
        raise ValueError(
            f"Expected MiniGrid spatial logits [B,{contract.output_channels},H,W], "
            f"got {tuple(spatial_logits.shape)}"
        )
    if current_state.ndim != 4 or current_state.shape[1] != 3:
        raise ValueError(f"Expected current MiniGrid state [B,3,H,W], got {tuple(current_state.shape)}")

    decoded_fields = []
    diagnostics: dict[str, torch.Tensor] = {}
    for name, target_index, classes in MINIGRID_FIELDS:
        start, stop = contract.spatial_slices[name]
        logits = spatial_logits[:, start:stop]
        current = current_state[:, target_index].long()
        if contract.mode == "effect":
            if name == "object":
                decoded, object_diag = _decode_object_with_single_agent(
                    logits, current, constrain_agent, collect_diagnostics
                )
                if collect_diagnostics:
                    diagnostics.update(object_diag)
            else:
                decoded = _next_class_probabilities(logits, current, classes).argmax(dim=1)
        else:
            decoded = logits.argmax(dim=1)
            if name == "object":
                if collect_diagnostics:
                    diagnostics["raw_agent_count"] = (
                        (decoded == 10).flatten(1).sum(dim=1)
                    )
                if constrain_agent:
                    decoded, object_diag = _constrain_absolute_object(
                        logits, decoded, collect_diagnostics
                    )
                    if collect_diagnostics:
                        diagnostics.update(object_diag)
        decoded_fields.append(decoded)

    next_state = torch.stack(decoded_fields, dim=1).to(dtype=current_state.dtype)
    next_inventory = None
    if inventory_logits is not None:
        if current_inventory is None:
            raise ValueError("current_inventory is required when inventory logits are provided")
        current_inventory = current_inventory.long().reshape(-1)
        if contract.mode == "effect":
            inv_probs = _next_class_probabilities(
                inventory_logits, current_inventory, MINIGRID_INVENTORY_CLASSES
            )
            next_inventory = inv_probs.argmax(dim=1)
        else:
            next_inventory = inventory_logits.argmax(dim=1)
    return next_state, next_inventory, diagnostics


def minigrid_effect_cell_nll(
    spatial_logits: torch.Tensor,
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    mode: str = "effect",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return normalized per-cell NLL and one map per spatial field."""
    contract = minigrid_contract(mode)
    if current_state.shape != next_state.shape or current_state.shape[1] != 3:
        raise ValueError("MiniGrid current and next states must both have shape [B,3,H,W]")
    maps: dict[str, torch.Tensor] = {}
    for name, target_index, classes in MINIGRID_FIELDS:
        start, stop = contract.spatial_slices[name]
        logits = spatial_logits[:, start:stop]
        target = next_state[:, target_index].long()
        if contract.mode == "effect":
            target = effect_target(current_state[:, target_index], target)
            denominator = classes + 1
        else:
            denominator = classes
        maps[name] = F.cross_entropy(logits, target, reduction="none") / torch.log(
            torch.tensor(float(denominator), device=logits.device)
        )
    return torch.stack(list(maps.values()), dim=0).mean(dim=0), maps
