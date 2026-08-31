"""Experiment artifact paths, variant identity, and dataset compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTINUAL_SUFFIX = "continue"
EFFECT_SUFFIX = "effect"
INVENTORY_GATE_SUFFIX = "inventory_gate_natural_v2"
# Canonical DreamerV2 Plan2Explore marker. Older artifacts remain on disk but
# are intentionally not addressable by the new acquisition path.
P2E_SUFFIX = "p2e_official_dv2_attnwm_v1"
RMAX_LIKE_SUFFIX = "rmax_count_local5_inv12_v1"


def detail_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove aggregate rows before merging a CSV with a new evaluation run."""
    detailed = frame.copy()
    for column in ("seed", "episode"):
        if column in detailed.columns:
            detailed = detailed.loc[
                detailed[column].astype(str).str.lower() != "mean"
            ]
    if "seed" in detailed.columns:
        detailed = detailed.loc[
            detailed["seed"].astype(str).str.lower() != "all"
        ]
    return detailed


def append_mean_row(
    frame: pd.DataFrame,
    *,
    mean_columns: list[str],
    labels: dict[str, object],
) -> pd.DataFrame:
    """Append one explicitly labelled aggregate row to an evaluation table."""
    mean_row: dict[str, object] = {column: np.nan for column in frame.columns}
    mean_row.update(labels)
    for column in mean_columns:
        mean_row[column] = pd.to_numeric(frame[column], errors="coerce").mean()
    return pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)


def continual_learning_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether the selected domain uses continual WM training."""
    selected_domain = str(domain or cfg.domain)
    domain_cfg = cfg.domains[selected_domain]
    continual_cfg = getattr(domain_cfg, "continual_learning", None)
    return bool(
        continual_cfg is not None
        and getattr(continual_cfg, "enabled", False)
    )


def categorical_effect_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether the domain map schema uses categorical effects."""
    selected_domain = str(domain or cfg.domain)
    if selected_domain == "minigrid":
        return str(
            getattr(getattr(cfg, "attention_model", None), "minigrid_transition_mode", "")
            or getattr(cfg.domains[selected_domain], "minigrid_transition_mode", "effect")
        ).lower() == "effect"
    schema = getattr(cfg.domains[selected_domain], "observation_schema", [])
    for field in schema:
        distribution = (
            field.get("distribution", "")
            if hasattr(field, "get")
            else getattr(field, "distribution", "")
        )
        if str(distribution) == "categorical_effect":
            return True
    return False


def inventory_gate_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether inventory uses structured KEEP/CHANGE and value heads."""
    selected_domain = str(domain or cfg.domain)
    schema = getattr(cfg.domains[selected_domain], "observation_schema", [])
    for field in schema:
        name = field.get("name", "") if hasattr(field, "get") else getattr(field, "name", "")
        distribution = (
            field.get("distribution", "")
            if hasattr(field, "get")
            else getattr(field, "distribution", "")
        )
        target_mode = (
            field.get("target_mode", "")
            if hasattr(field, "get")
            else getattr(field, "target_mode", "")
        )
        if (
            str(name) == "inventory"
            and str(distribution) == "categorical_inventory_gate"
            and str(target_mode) == "categorical_gate"
        ):
            return True
    return False


def p2e_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether Crafter WM acquisition uses Plan2Explore."""
    selected_domain = str(domain or cfg.domain)
    p2e_cfg = getattr(cfg, "p2e", None)
    return bool(
        selected_domain == "crafter"
        and p2e_cfg is not None
        and getattr(p2e_cfg, "enabled", False)
    )


def rmax_like_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether Crafter WM acquisition uses count-based real PPO."""
    selected_domain = str(domain or cfg.domain)
    rmax_cfg = getattr(cfg, "rmax_like", None)
    return bool(
        selected_domain == "crafter"
        and rmax_cfg is not None
        and getattr(rmax_cfg, "enabled", False)
    )


def validate_acquisition_flags(cfg, domain: str | None = None) -> None:
    """Reject ambiguous acquisition configurations."""
    selected_domain = str(domain or cfg.domain)
    if p2e_enabled(cfg, selected_domain) and rmax_like_enabled(cfg, selected_domain):
        raise ValueError("p2e.enabled and rmax_like.enabled are mutually exclusive")
    if p2e_enabled(cfg, selected_domain) and continual_learning_enabled(cfg, selected_domain):
        raise ValueError("official single-environment P2E cannot run with continual_learning.enabled=true")


def add_effect_suffix(path: str | Path, *, before_seed: bool = False) -> Path:
    """Add the categorical-effect representation marker once."""
    artifact = Path(str(path)).expanduser()
    marker = f"_{EFFECT_SUFFIX}"
    if marker in artifact.stem:
        return artifact.resolve()
    stem = artifact.stem
    if before_seed and "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}{marker}_seed{seed}"
    else:
        stem = f"{stem}{marker}"
    return artifact.with_name(f"{stem}{artifact.suffix}").resolve()


def add_inventory_gate_suffix(path: str | Path, *, before_seed: bool = False) -> Path:
    """Add the structured inventory-gate representation marker once."""
    artifact = Path(str(path)).expanduser()
    marker = f"_{INVENTORY_GATE_SUFFIX}"
    if marker in artifact.stem:
        return artifact.resolve()
    stem = artifact.stem
    if before_seed and "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}{marker}_seed{seed}"
    else:
        stem = f"{stem}{marker}"
    return artifact.with_name(f"{stem}{artifact.suffix}").resolve()


def add_continual_suffix(path: str | Path, *, before_seed: bool = False) -> Path:
    """Add the canonical continual marker once, preserving the extension."""
    artifact = Path(str(path)).expanduser()
    stem = artifact.stem
    marker = f"_{CONTINUAL_SUFFIX}"
    if marker in stem:
        return artifact.resolve()

    if before_seed and "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}{marker}_seed{seed}"
    else:
        stem = f"{stem}{marker}"
    return artifact.with_name(f"{stem}{artifact.suffix}").resolve()


def add_p2e_suffix(path: str | Path, *, before_seed: bool = False) -> Path:
    """Add the Plan2Explore acquisition marker once."""
    artifact = Path(str(path)).expanduser()
    marker = f"_{P2E_SUFFIX}"
    if marker in artifact.stem:
        return artifact.resolve()
    stem = artifact.stem
    if before_seed and "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}{marker}_seed{seed}"
    else:
        stem = f"{stem}{marker}"
    return artifact.with_name(f"{stem}{artifact.suffix}").resolve()


def add_rmax_like_suffix(path: str | Path, *, before_seed: bool = False) -> Path:
    """Add the count-based acquisition marker once."""
    artifact = Path(str(path)).expanduser()
    marker = f"_{RMAX_LIKE_SUFFIX}"
    if marker in artifact.stem:
        return artifact.resolve()
    stem = artifact.stem
    if before_seed and "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}{marker}_seed{seed}"
    else:
        stem = f"{stem}{marker}"
    return artifact.with_name(f"{stem}{artifact.suffix}").resolve()


def world_model_checkpoint_path(cfg, domain: str | None = None) -> Path:
    """Resolve the WM checkpoint matching standard or continual training."""
    selected_domain = str(domain or cfg.domain)
    validate_acquisition_flags(cfg, selected_domain)
    domain_cfg = cfg.domains[selected_domain]
    if continual_learning_enabled(cfg, selected_domain):
        configured = continual_artifact_path(
            cfg, "final_checkpoint", selected_domain
        )
        if configured is not None:
            if categorical_effect_enabled(cfg, selected_domain):
                configured = add_effect_suffix(configured)
            return configured
    base = Path(str(domain_cfg.world_model_checkpoint)).expanduser().resolve()
    if categorical_effect_enabled(cfg, selected_domain):
        base = add_effect_suffix(base)
    if p2e_enabled(cfg, selected_domain):
        base = add_p2e_suffix(base)
    if rmax_like_enabled(cfg, selected_domain):
        base = add_rmax_like_suffix(base)
    if continual_learning_enabled(cfg, selected_domain):
        base = add_continual_suffix(base)
    return base


def align_world_model_artifact_path(cfg) -> Path | None:
    """Align the configured WM path with the active MiniGrid representation."""
    if str(getattr(cfg, "domain", "")) != "minigrid":
        return None
    configured = getattr(getattr(cfg, "attention_model", None), "model_save_path", None)
    if configured is None:
        return None
    path = Path(str(configured)).expanduser().resolve()
    if categorical_effect_enabled(cfg, "minigrid"):
        path = add_effect_suffix(path, before_seed=True)
    cfg.attention_model.model_save_path = str(path)
    return path


def single_environment_dataset_path(cfg, domain: str | None = None) -> Path:
    """Resolve the dataset used by a non-continual WM run.

    P2E and random acquisition must not silently overwrite or reuse each
    other's data. Continual runs do not use this helper because every phase
    already owns an explicit ``data_dir``.
    """
    selected_domain = str(domain or cfg.domain)
    validate_acquisition_flags(cfg, selected_domain)
    domain_cfg = cfg.domains[selected_domain]
    if p2e_enabled(cfg, selected_domain):
        configured = getattr(domain_cfg, "p2e_data_save_path", None)
        if configured is not None and str(configured).strip():
            return Path(str(configured)).expanduser().resolve()
        return add_p2e_suffix(domain_cfg.data_save_path)
    if rmax_like_enabled(cfg, selected_domain):
        configured = getattr(domain_cfg, "rmax_like_data_save_path", None)
        if configured is not None and str(configured).strip():
            return Path(str(configured)).expanduser().resolve()
        return add_rmax_like_suffix(domain_cfg.data_save_path)
    return Path(str(domain_cfg.data_save_path)).expanduser().resolve()


def continual_artifact_path(
    cfg,
    field: str,
    domain: str | None = None,
) -> Path | None:
    """Select the random or P2E state/replay/checkpoint/metrics artifact."""
    selected_domain = str(domain or cfg.domain)
    continual_cfg = cfg.domains[selected_domain].continual_learning
    configured = None
    if p2e_enabled(cfg, selected_domain):
        configured = getattr(continual_cfg, f"p2e_{field}", None)
    if configured is None or not str(configured).strip():
        configured = getattr(continual_cfg, field, None)
        if (
            configured is not None
            and str(configured).strip()
            and p2e_enabled(cfg, selected_domain)
        ):
            return add_p2e_suffix(configured)
    if configured is None or not str(configured).strip():
        return None
    return Path(str(configured)).expanduser().resolve()


def continual_phase_data_path(cfg, phase, domain: str | None = None) -> Path:
    """Select one continual phase's random or P2E dataset."""
    selected_domain = str(domain or cfg.domain)
    configured = None
    if p2e_enabled(cfg, selected_domain):
        configured = phase.get("p2e_data_dir", None)
    if configured is None or not str(configured).strip():
        configured = phase.get("data_dir", None)
    if configured is None or not str(configured).strip():
        raise ValueError("Continual phase has no selected data_dir")
    path = Path(str(configured)).expanduser()
    if rmax_like_enabled(cfg, selected_domain):
        return add_rmax_like_suffix(path)
    return path.resolve()


def layout_hash(path: str | Path | None) -> str | None:
    """Return a stable content hash for one environment layout."""
    if not path:
        return None
    file_path = Path(str(path)).expanduser().resolve()
    if not file_path.exists():
        return None
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def identity_from_config(cfg: Any, domain: str | None = None) -> dict[str, Any]:
    """Build the metadata contract for a collected transition dataset."""
    domain = domain or str(cfg.domain)
    domain_cfg = cfg.domains[domain]
    layout_path = str(
        getattr(domain_cfg, "layout_path", getattr(domain_cfg, "env_path", ""))
    )
    identity = {
        "domain": domain,
        "task_name": str(getattr(domain_cfg, "task_name", "")),
        "env_path": str(Path(layout_path).expanduser().resolve()),
        "layout_path": str(Path(layout_path).expanduser().resolve()),
        "layout_hash": layout_hash(layout_path),
    }
    if domain == "minigrid":
        identity["observation_pair_encoding"] = "absolute_state_pair_v1"
        identity["action_encoding"] = "compact6_v1"
        collect_cfg = getattr(getattr(cfg, "env", None), "collect", None)
        identity["collection_replace_start_with_empty"] = bool(
            getattr(collect_cfg, "replace_start_with_empty", False)
        )
        identity["inventory_encoding"] = "key_color_token_v1"
    elif domain == "crafter":
        identity["reward_schema"] = "crafter_native_v1"
        identity["inventory_encoding"] = "crafter_inventory_v1"
        data_collection = getattr(domain_cfg, "data_collection", None)
        if data_collection is not None:
            identity["episode_max_steps"] = int(
                getattr(data_collection, "max_steps", 0)
            )
        continual_cfg = getattr(domain_cfg, "continual_learning", None)
        continual_enabled = bool(
            continual_cfg is not None
            and getattr(continual_cfg, "enabled", False)
        )
        p2e_cfg = getattr(cfg, "p2e", None)
        uses_p2e = bool(
            p2e_cfg is not None and getattr(p2e_cfg, "enabled", False)
        )
        rmax_cfg = getattr(cfg, "rmax_like", None)
        uses_rmax = bool(
            rmax_cfg is not None and getattr(rmax_cfg, "enabled", False)
        )
        if uses_p2e and uses_rmax:
            raise ValueError("p2e.enabled and rmax_like.enabled are mutually exclusive")
        if uses_rmax:
            identity["collection_policy"] = "crafter_rmax_count_local5_inv12_v1"
            identity["rmax_like"] = {
                "count_key_version": "local5_inv12_sa_v1",
                "mask_size": int(getattr(rmax_cfg, "mask_size", 5)),
                "death_penalty": float(getattr(rmax_cfg, "death_penalty", 0.0)),
                "train_steps": int(getattr(rmax_cfg, "train_steps", 0)),
                "frozen_steps": int(getattr(rmax_cfg, "frozen_steps", 0)),
                "seed": int(getattr(rmax_cfg, "seed", 0)),
            }
        elif uses_p2e or continual_enabled:
            identity["collection_policy"] = (
                "crafter_p2e_official_dv2_attnwm_v1"
                if uses_p2e
                else "crafter_uniform_random_v1"
            )
        initial_inventory = getattr(domain_cfg, "initial_inventory", None)
        if initial_inventory:
            identity["initial_inventory"] = {
                str(key): float(value)
                for key, value in sorted(dict(initial_inventory).items())
            }
    return identity


def dataset_metadata(path: str | Path) -> dict[str, Any] | None:
    """Load identity metadata from an NPZ dataset."""
    try:
        with np.load(path, allow_pickle=True) as data:
            if "metadata" not in data.files:
                return None
            raw = (
                data["metadata"].item()
                if data["metadata"].shape == ()
                else data["metadata"].tolist()
            )
            return json.loads(str(raw))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def dataset_matches(path: str | Path, cfg: Any, domain: str | None = None) -> bool:
    """Return whether a saved dataset matches the active experiment config."""
    return dataset_metadata(path) == identity_from_config(cfg, domain)


def metadata_array(cfg: Any, domain: str | None = None) -> np.ndarray:
    """Serialize the active dataset identity for NPZ storage."""
    return np.asarray(json.dumps(identity_from_config(cfg, domain)), dtype=object)
