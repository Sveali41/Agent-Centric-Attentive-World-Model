"""Pure inventory-event counters for Crafter training diagnostics."""

from __future__ import annotations

import json

import numpy as np

from modelBased.world_model.crafter_dynamics import CRAFTER_CANONICAL_EVENT_CODEBOOK


_EVENT_IDS = {tuple(delta): event_id for event_id, delta in enumerate(CRAFTER_CANONICAL_EVENT_CODEBOOK)}


def inventory_event_rows(current, following):
    """Count complete 12-item-slot deltas; unknown vectors remain explicit."""
    if hasattr(current, "detach"):
        current = current.detach().cpu().numpy()
    if hasattr(following, "detach"):
        following = following.detach().cpu().numpy()
    current = np.asarray(current)
    following = np.asarray(following)
    if current.ndim != 2 or following.shape != current.shape or current.shape[1] < 16:
        raise ValueError(
            "Crafter inventory event audit expects aligned [N, >=16] current/next inventories"
        )
    current_items = current[:, 4:16]
    following_items = following[:, 4:16]
    if not np.isfinite(current_items).all() or not np.isfinite(following_items).all():
        raise ValueError("Crafter inventory event audit received non-finite item values")
    if not np.equal(current_items, np.rint(current_items)).all() or not np.equal(
        following_items, np.rint(following_items)
    ).all():
        raise ValueError("Crafter inventory event audit received non-integer item values")
    delta = following_items.astype(np.int64) - current_items.astype(np.int64)
    vectors, counts = np.unique(delta, axis=0, return_counts=True)
    rows = []
    for vector, count in zip(vectors, counts):
        key = tuple(int(value) for value in vector)
        changed_slots = np.flatnonzero(vector).astype(int).tolist()
        rows.append({
            "event_id": _EVENT_IDS.get(key, -1),
            "delta_vector": json.dumps(key, separators=(",", ":")),
            "count": int(count),
            "changed_slot_count": int(len(changed_slots)),
            "changed_slots": json.dumps(changed_slots, separators=(",", ":")),
        })
    return rows


def inventory_event_episode_age_rows(current, following, done, *, cutoff=250):
    """Count item events by within-episode age for one rollout/map."""
    if hasattr(current, "detach"):
        current = current.detach().cpu().numpy()
    if hasattr(following, "detach"):
        following = following.detach().cpu().numpy()
    current = np.asarray(current)
    following = np.asarray(following)
    if hasattr(done, "detach"):
        done = done.detach().cpu().numpy()
    done = np.asarray(done).reshape(-1).astype(bool)
    if current.ndim != 2 or current.shape[1] < 16 or following.shape != current.shape:
        raise ValueError("Episode-age audit expects aligned [N, >=16] inventory arrays")
    if len(done) != len(current):
        raise ValueError("Episode-age audit requires one done flag per inventory transition")
    if cutoff < 1:
        raise ValueError("Episode-age cutoff must be positive")

    # Reuse canonical validation before computing raw complete-item deltas.
    inventory_event_rows(current, following)
    deltas = following[:, 4:16].astype(np.int64) - current[:, 4:16].astype(np.int64)
    age = 0
    episode_id = 0
    grouped = {}
    bin_stats = {
        "steps_1_250": {"transition_count": 0, "episode_ids": set()},
        "steps_251_plus": {"transition_count": 0, "episode_ids": set()},
    }
    for index, delta in enumerate(deltas):
        age += 1
        age_bin = "steps_1_250" if age <= cutoff else "steps_251_plus"
        event = tuple(int(value) for value in delta)
        entry = grouped.setdefault((age_bin, event), {"count": 0})
        entry["count"] += 1
        bin_stats[age_bin]["transition_count"] += 1
        bin_stats[age_bin]["episode_ids"].add(episode_id)
        if done[index]:
            age = 0
            episode_id += 1

    rows = []
    for (age_bin, event), value in sorted(grouped.items()):
        denominator = bin_stats[age_bin]["transition_count"]
        changed_slots = np.flatnonzero(event).astype(int).tolist()
        rows.append({
            "episode_age_bin": age_bin,
            "bin_transition_count": denominator,
            "episodes_with_transitions_in_bin": len(bin_stats[age_bin]["episode_ids"]),
            "event_id": _EVENT_IDS.get(event, -1),
            "delta_vector": json.dumps(event, separators=(",", ":")),
            "event_count": value["count"],
            "rate_per_1000": 1000.0 * value["count"] / denominator,
            "changed_slot_count": len(changed_slots),
            "changed_slots": json.dumps(changed_slots, separators=(",", ":")),
        })
    return rows
