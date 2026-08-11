"""Consistent policy artifact and WandB naming across training sources."""

from __future__ import annotations

from pathlib import Path

import torch

from domain.minigrid.action_codec import INVENTORY_TOKEN_COUNT, MODEL_ACTION_COUNT
from modelBased.common.artifact_naming import (
    add_continual_suffix,
    add_effect_suffix,
    add_p2e_suffix,
    categorical_effect_enabled,
    continual_learning_enabled,
    p2e_enabled,
)


def policy_training_source(cfg) -> str:
    """Return the automatically selected transition source label."""
    return "real_env" if bool(cfg.PPO.train_in_real_env) else "planning"


def policy_checkpoint_path(cfg, domain: str | None = None) -> Path:
    """Resolve an explicit override or the source-specific default path."""
    explicit = getattr(cfg.PPO, "checkpoint_path", None)
    if explicit is not None and str(explicit).strip() and str(explicit).lower() != "null":
        return Path(str(explicit)).expanduser().resolve()

    source = policy_training_source(cfg)
    if domain is not None and str(domain) != str(cfg.domain):
        task_name = str(cfg.domains[str(domain)].task_name)
        source_suffix = "_realenv" if source == "real_env" else ""
        filename = f"policy_{domain}_{task_name}{source_suffix}_seed{int(getattr(cfg.PPO, 'seed', 0))}.ckpt"
        path = (
            Path(str(cfg.PPO.checkpoint_dir)).expanduser().resolve() / filename
        )
        if source == "planning" and categorical_effect_enabled(cfg, str(domain)):
            path = add_effect_suffix(path, before_seed=True)
        if source == "planning" and p2e_enabled(cfg, str(domain)):
            path = add_p2e_suffix(path, before_seed=True)
        return (
            add_continual_suffix(path, before_seed=True)
            if continual_learning_enabled(cfg, str(domain))
            else path
        )

    field = f"checkpoint_path_{source}"
    if not hasattr(cfg.PPO, field):
        raise ValueError(f"Missing PPO.{field} for {source} policy training")
    path = Path(str(getattr(cfg.PPO, field))).expanduser().resolve()
    if source == "planning" and categorical_effect_enabled(cfg, str(domain or cfg.domain)):
        path = add_effect_suffix(path, before_seed=True)
    if source == "planning" and p2e_enabled(cfg, str(domain or cfg.domain)):
        path = add_p2e_suffix(path, before_seed=True)
    return (
        add_continual_suffix(path, before_seed=True)
        if continual_learning_enabled(cfg, str(domain or cfg.domain))
        else path
    )


def policy_wandb_identity(cfg) -> tuple[str, str, str]:
    """Return ``(group, run_name, source)`` derived from source and layout."""
    source = policy_training_source(cfg)
    domain = str(cfg.domain)
    task_name = str(cfg.domains[domain].task_name)
    seed = int(getattr(cfg.PPO, "seed", 0))
    # Keep one group per layout so its seed runs are directly comparable.
    # Preserve the historical planning group name; only real-environment
    # training receives an extra source marker to keep the two baselines apart.
    group_base = f"{domain}_{task_name}"
    if source == "planning" and categorical_effect_enabled(cfg, domain):
        group_base += "_effect"
    if source == "planning" and p2e_enabled(cfg, domain):
        group_base += "_p2e"
    if continual_learning_enabled(cfg, domain):
        group_base += "_continue"
    group = (
        f"{group_base}_realenv_policy"
        if source == "real_env"
        else f"{group_base}_policy"
    )
    run_base = f"{domain}_{task_name}"
    if source == "planning" and categorical_effect_enabled(cfg, domain):
        run_base += "_effect"
    if source == "planning" and p2e_enabled(cfg, domain):
        run_base += "_p2e"
    if continual_learning_enabled(cfg, domain):
        run_base += "_continue"
    if source == "real_env":
        run_base += "_realenv"
    run_name = f"{run_base}_seed{seed}"
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
            expected_state_dim = 2 * height * width + inventory_dim
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
