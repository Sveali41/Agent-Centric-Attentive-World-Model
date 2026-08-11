"""Reusable categorical state-effect targets and reconstruction helpers.

Effect class 0 means KEEP.  Effect class ``k + 1`` means SET_TO category
``k``.  This keeps categorical state transitions semantic: category IDs are
never subtracted as though they were continuous values.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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
) -> torch.Tensor:
    """Cross entropy with automatic equal averaging of KEEP and SET_TO cells.

    ``balanced_mean`` first averages each non-empty group (KEEP and SET_TO),
    then averages the group losses.  It therefore avoids a hand-tuned class
    weight while preventing unchanged cells from dominating sparse effects.
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
