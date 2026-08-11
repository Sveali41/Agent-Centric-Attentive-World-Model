"""Shared artifact naming for standard and continual-learning runs."""

from __future__ import annotations

from pathlib import Path


CONTINUAL_SUFFIX = "continue"
EFFECT_SUFFIX = "effect"
P2E_SUFFIX = "p2e"


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


def p2e_enabled(cfg, domain: str | None = None) -> bool:
    """Return whether Crafter WM acquisition uses Plan2Explore."""
    selected_domain = str(domain or cfg.domain)
    p2e_cfg = getattr(cfg, "p2e", None)
    return bool(
        selected_domain == "crafter"
        and p2e_cfg is not None
        and getattr(p2e_cfg, "enabled", False)
    )


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


def world_model_checkpoint_path(cfg, domain: str | None = None) -> Path:
    """Resolve the WM checkpoint matching standard or continual training."""
    selected_domain = str(domain or cfg.domain)
    domain_cfg = cfg.domains[selected_domain]
    if continual_learning_enabled(cfg, selected_domain):
        configured = continual_artifact_path(
            cfg, "final_checkpoint", selected_domain
        )
        if configured is not None:
            return configured
        base = Path(str(domain_cfg.world_model_checkpoint)).expanduser().resolve()
        if p2e_enabled(cfg, selected_domain):
            base = add_p2e_suffix(base)
        return add_continual_suffix(base)
    base = Path(str(domain_cfg.world_model_checkpoint)).expanduser().resolve()
    return add_p2e_suffix(base) if p2e_enabled(cfg, selected_domain) else base


def single_environment_dataset_path(cfg, domain: str | None = None) -> Path:
    """Resolve the dataset used by a non-continual WM run.

    P2E and random acquisition must not silently overwrite or reuse each
    other's data. Continual runs do not use this helper because every phase
    already owns an explicit ``data_dir``.
    """
    selected_domain = str(domain or cfg.domain)
    domain_cfg = cfg.domains[selected_domain]
    if p2e_enabled(cfg, selected_domain):
        configured = getattr(domain_cfg, "p2e_data_save_path", None)
        if configured is not None and str(configured).strip():
            return Path(str(configured)).expanduser().resolve()
        return add_p2e_suffix(domain_cfg.data_save_path)
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
    return Path(str(configured)).expanduser().resolve()
