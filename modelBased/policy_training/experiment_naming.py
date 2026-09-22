"""Consistent policy artifact and WandB naming across training sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch

from domain.minigrid.action_codec import INVENTORY_TOKEN_COUNT, MODEL_ACTION_COUNT
from modelBased.common.artifacts import (
    INVENTORY_GATE_SUFFIX,
    P2E_SUFFIX,
    RMAX_LIKE_SUFFIX,
    add_continual_suffix,
    add_effect_suffix,
    add_inventory_gate_suffix,
    add_p2e_suffix,
    add_rmax_like_suffix,
    categorical_effect_enabled,
    continual_learning_enabled,
    inventory_gate_enabled,
    p2e_enabled,
    rmax_like_enabled,
    validate_acquisition_flags,
)


_WANDB_NAME_LIMIT = 128
CRAFTER_ACHIEVEMENT_MASK_SUFFIX = "achmask22"
_CHECKPOINT_SEED_PATTERN = re.compile(r"(?:^|_)seed(?P<seed>\d+)(?:_|$)")


def _bounded_wandb_name(value: str, limit: int = _WANDB_NAME_LIMIT) -> str:
    """Keep WandB identifiers within its limit without creating collisions."""
    if len(value) <= limit:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[: limit - len(digest) - 1]}_{digest}"


def policy_training_source(cfg) -> str:
    """Return the automatically selected transition source label."""
    return "real_env" if bool(cfg.PPO.train_in_real_env) else "planning"


def policy_experiment_label(cfg) -> str:
    """Return a filesystem/WandB-safe optional policy experiment label."""
    configured = getattr(cfg.PPO, "experiment_label", None)
    label = (
        ""
        if configured is None
        or str(configured).strip().lower() in {"", "null", "none"}
        else str(configured).strip()
    )
    if bool(getattr(cfg.PPO, "use_main_dense_reward", False)):
        label = f"{label}_main_dense" if label else "main_dense"
    if not label:
        return ""
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    label = label.strip("._-")
    if not label:
        raise ValueError("PPO.experiment_label must contain a letter or number")
    return label


def _add_policy_experiment_label(path: Path, cfg) -> Path:
    label = policy_experiment_label(cfg)
    if not label:
        return path.resolve()
    stem = path.stem
    if "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}_{label}_seed{seed}"
    else:
        stem = f"{stem}_{label}"
    return path.with_name(f"{stem}{path.suffix}").resolve()


def _add_crafter_achievement_mask_suffix(path: Path, domain: str) -> Path:
    """Keep 22-bit achievement-aware policies separate from legacy inputs."""
    if str(domain).lower() != "crafter":
        return path
    stem = path.stem
    if "_seed" in stem:
        prefix, seed = stem.rsplit("_seed", 1)
        stem = f"{prefix}_{CRAFTER_ACHIEVEMENT_MASK_SUFFIX}_seed{seed}"
    else:
        stem = f"{stem}_{CRAFTER_ACHIEVEMENT_MASK_SUFFIX}"
    return path.with_name(f"{stem}{path.suffix}")


def _crafter_planning_policy_path(cfg) -> Path:
    """Name a Crafter imagined-policy by task and its actual WM seed."""
    configured_wm = getattr(cfg.PPO, "checkpoint_path_wm", None)
    if configured_wm is None or str(configured_wm).strip().lower() in {
        "",
        "null",
        "none",
    }:
        configured_wm = getattr(
            cfg.domains.crafter, "planning_world_model_checkpoint", None
        )
    if configured_wm is None or str(configured_wm).strip().lower() in {
        "",
        "null",
        "none",
    }:
        raise ValueError(
            "Crafter planning policy naming requires PPO.checkpoint_path_wm or "
            "domains.crafter.planning_world_model_checkpoint"
        )

    matches = list(_CHECKPOINT_SEED_PATTERN.finditer(Path(str(configured_wm)).stem))
    if not matches:
        raise ValueError(
            "Cannot determine the WM seed from Crafter checkpoint path "
            f"{configured_wm!s}; include '_seed<N>' in the checkpoint filename"
        )
    wm_seed = int(matches[-1].group("seed"))
    task_name = str(cfg.domains.crafter.task_name)
    filename = f"policy_crafter_{task_name}_seed{wm_seed}.ckpt"
    return (Path(str(cfg.PPO.checkpoint_dir)).expanduser() / filename).resolve()


def policy_checkpoint_path(cfg, domain: str | None = None) -> Path:
    """Resolve an explicit override or the source-specific default path."""
    explicit = getattr(cfg.PPO, "checkpoint_path", None)
    if explicit is not None and str(explicit).strip() and str(explicit).lower() != "null":
        return Path(str(explicit)).expanduser().resolve()

    source = policy_training_source(cfg)
    selected_domain = str(domain or cfg.domain)
    validate_acquisition_flags(cfg, selected_domain)
    if source == "planning" and selected_domain == "crafter":
        return _crafter_planning_policy_path(cfg)
    if domain is not None and str(domain) != str(cfg.domain):
        task_name = str(cfg.domains[str(domain)].task_name)
        source_suffix = "_realenv" if source == "real_env" else ""
        filename = f"policy_{domain}_{task_name}{source_suffix}_seed{int(getattr(cfg.PPO, 'seed', 0))}.ckpt"
        path = (
            Path(str(cfg.PPO.checkpoint_dir)).expanduser().resolve() / filename
        )
        if source == "planning" and categorical_effect_enabled(cfg, str(domain)):
            path = add_effect_suffix(path, before_seed=True)
        if source == "planning" and inventory_gate_enabled(cfg, str(domain)):
            path = add_inventory_gate_suffix(path, before_seed=True)
        if source == "planning" and p2e_enabled(cfg, str(domain)):
            path = add_p2e_suffix(path, before_seed=True)
        if source == "planning" and rmax_like_enabled(cfg, str(domain)):
            path = add_rmax_like_suffix(path, before_seed=True)
        path = (
            add_continual_suffix(path, before_seed=True)
            if continual_learning_enabled(cfg, str(domain))
            else path
        )
        return _add_policy_experiment_label(
            _add_crafter_achievement_mask_suffix(path, str(domain)), cfg
        )

    field = f"checkpoint_path_{source}"
    if not hasattr(cfg.PPO, field):
        raise ValueError(f"Missing PPO.{field} for {source} policy training")
    path = Path(str(getattr(cfg.PPO, field))).expanduser().resolve()
    if source == "planning" and categorical_effect_enabled(cfg, str(domain or cfg.domain)):
        path = add_effect_suffix(path, before_seed=True)
    if source == "planning" and inventory_gate_enabled(cfg, str(domain or cfg.domain)):
        path = add_inventory_gate_suffix(path, before_seed=True)
    if source == "planning" and p2e_enabled(cfg, str(domain or cfg.domain)):
        path = add_p2e_suffix(path, before_seed=True)
    if source == "planning" and rmax_like_enabled(cfg, str(domain or cfg.domain)):
        path = add_rmax_like_suffix(path, before_seed=True)
    path = (
        add_continual_suffix(path, before_seed=True)
        if continual_learning_enabled(cfg, str(domain or cfg.domain))
        else path
    )
    return _add_policy_experiment_label(
        _add_crafter_achievement_mask_suffix(path, str(domain or cfg.domain)), cfg
    )


def policy_selection_paths(checkpoint_path: str | Path) -> dict[str, Path]:
    """Return canonical, best, last, and selection-manifest policy artifacts."""
    canonical = Path(checkpoint_path).expanduser().resolve()
    suffix = canonical.suffix or ".ckpt"
    stem = canonical.stem if canonical.suffix else canonical.name
    return {
        "canonical": canonical,
        "best": canonical.with_name(f"{stem}_best{suffix}"),
        "last": canonical.with_name(f"{stem}_last{suffix}"),
        "manifest": canonical.with_name(f"{stem}_selection.json"),
    }


def policy_checkpoint_for_evaluation(cfg, domain: str | None = None) -> Path:
    """Return the canonical artifact, which selection keeps equal to best."""
    return policy_checkpoint_path(cfg, domain=domain)


def policy_validation_stem(cfg, domain: str | None = None) -> str:
    """Derive evaluation artifact names from the exact policy being loaded."""
    return f"{policy_checkpoint_for_evaluation(cfg, domain=domain).stem}_test"


def policy_wandb_identity(cfg) -> tuple[str, str, str]:
    """Return ``(group, run_name, source)`` derived from source and layout."""
    source = policy_training_source(cfg)
    domain = str(cfg.domain)
    task_name = str(cfg.domains[domain].task_name)
    seed = int(getattr(cfg.PPO, "seed", 0))
    # Keep one group per layout so its seed runs are directly comparable.
    # Include the complete acquisition variant in planning names so P2E
    # baselines and Go-Explore imagined-archive runs cannot collide.
    group_base = f"{domain}_{task_name}"
    if source == "planning" and categorical_effect_enabled(cfg, domain):
        group_base += "_effect"
    if source == "planning" and inventory_gate_enabled(cfg, domain):
        group_base += f"_{INVENTORY_GATE_SUFFIX}"
    if source == "planning" and p2e_enabled(cfg, domain):
        group_base += f"_{P2E_SUFFIX}"
    if source == "planning" and rmax_like_enabled(cfg, domain):
        group_base += f"_{RMAX_LIKE_SUFFIX}"
    if continual_learning_enabled(cfg, domain):
        group_base += "_continue"
    if domain == "crafter":
        group_base += f"_{CRAFTER_ACHIEVEMENT_MASK_SUFFIX}"
    group = (
        f"{group_base}_realenv_policy"
        if source == "real_env"
        else f"{group_base}_policy"
    )
    run_base = f"{domain}_{task_name}"
    if source == "planning" and categorical_effect_enabled(cfg, domain):
        run_base += "_effect"
    if source == "planning" and inventory_gate_enabled(cfg, domain):
        run_base += f"_{INVENTORY_GATE_SUFFIX}"
    if source == "planning" and p2e_enabled(cfg, domain):
        run_base += f"_{P2E_SUFFIX}"
    if source == "planning" and rmax_like_enabled(cfg, domain):
        run_base += f"_{RMAX_LIKE_SUFFIX}"
    if continual_learning_enabled(cfg, domain):
        run_base += "_continue"
    if domain == "crafter":
        run_base += f"_{CRAFTER_ACHIEVEMENT_MASK_SUFFIX}"
    if source == "real_env":
        run_base += "_realenv"
    experiment_label = policy_experiment_label(cfg)
    if experiment_label:
        group += f"_{experiment_label}"
        run_base += f"_{experiment_label}"
    group = _bounded_wandb_name(group)
    run_name = _bounded_wandb_name(f"{run_base}_seed{seed}")
    return group, run_name, source


def policy_checkpoint_is_compatible(path, cfg, domain: str = "minigrid") -> bool:
    """Check actor input/output sizes before reusing a policy checkpoint."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return False
    if domain not in {"minigrid", "crafter"}:
        return True

    layout_path = Path(str(cfg.domains[domain].layout_path)).expanduser().resolve()
    try:
        layout_lines = [
            line.strip()
            for line in layout_path.read_text(encoding="utf-8")
            .split("\n\n", 1)[0]
            .splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        height = len(layout_lines)
        width = max(len(line) for line in layout_lines)
        if domain == "crafter":
            inventory_dim = int(getattr(cfg.domains[domain], "inventory_dim", 16))
            expected_state_dim = 2 * height * width + inventory_dim + 22
            expected_action_dim = 17
        else:
            expected_state_dim = 3 * height * width + INVENTORY_TOKEN_COUNT
            expected_action_dim = MODEL_ACTION_COUNT
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return (
            tuple(state_dict["actor.0.weight"].shape)[1] == expected_state_dim
            and tuple(state_dict["actor.4.weight"].shape)[0] == expected_action_dim
        )
    except (OSError, RuntimeError, KeyError, TypeError, ValueError):
        return False
