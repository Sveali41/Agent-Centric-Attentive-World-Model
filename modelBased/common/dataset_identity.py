"""Dataset identity checks to prevent training on the wrong environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def layout_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    file_path = Path(str(path)).expanduser().resolve()
    if not file_path.exists():
        return None
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def identity_from_config(cfg: Any, domain: str | None = None) -> dict[str, Any]:
    domain = domain or str(cfg.domain)
    domain_cfg = cfg.domains[domain]
    # ``layout_path`` is the single source of truth for the environment
    # layout. Keep the fallback for older external configs.
    layout_path = str(
        getattr(domain_cfg, "layout_path", getattr(domain_cfg, "env_path", ""))
    )
    identity = {
        "domain": domain,
        "task_name": str(getattr(domain_cfg, "task_name", "")),
        # env_path is retained as a compatibility key for existing tooling;
        # both fields deliberately contain the same canonical path.
        "env_path": str(Path(layout_path).expanduser().resolve()),
        "layout_path": str(Path(layout_path).expanduser().resolve()),
        "layout_hash": layout_hash(layout_path),
    }
    if domain == "minigrid":
        collect_cfg = getattr(getattr(cfg, "env", None), "collect", None)
        identity["collection_replace_start_with_empty"] = bool(
            getattr(collect_cfg, "replace_start_with_empty", False)
        )
        # Version the transition state representation so datasets collected
        # before colour-aware carried inventory are never silently reused.
        identity["inventory_encoding"] = "key_color_token_v1"
    elif domain == "crafter":
        # Crafter datasets contain native simulator reward and the 16-slot
        # inventory transition used by the auxiliary WM head. Achievements
        # remain planner history and are deliberately not a WM target.
        identity["reward_schema"] = "crafter_native_v1"
        identity["inventory_encoding"] = "crafter_inventory_v1"
        # The per-episode horizon changes how often collection resets before
        # reaching a long-horizon crafting interaction. Include it in the
        # dataset contract so changing max_steps cannot silently reuse an old
        # P2E artifact collected with a shorter horizon.
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
        p2e_enabled = bool(
            p2e_cfg is not None and getattr(p2e_cfg, "enabled", False)
        )
        if p2e_enabled or continual_enabled:
            # Acquisition policy is part of the data contract. This protects
            # both single-environment P2E and continual phase datasets from
            # accidentally reusing random data with the same layout.
            identity["collection_policy"] = (
                "crafter_p2e_task_aware_v3"
                if p2e_enabled
                else "crafter_uniform_random_v1"
            )
        # Non-empty starting inventories define distinct continual-learning
        # phases even when the layout is identical.  Keep the empty/default
        # phase backward compatible with datasets collected before this field
        # existed in the metadata schema.
        initial_inventory = getattr(domain_cfg, "initial_inventory", None)
        if initial_inventory:
            identity["initial_inventory"] = {
                str(key): float(value)
                for key, value in sorted(dict(initial_inventory).items())
            }
    return identity


def dataset_metadata(path: str | Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "metadata" not in data.files:
                return None
            raw = data["metadata"].item() if data["metadata"].shape == () else data["metadata"].tolist()
            return json.loads(str(raw))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def dataset_matches(path: str | Path, cfg: Any, domain: str | None = None) -> bool:
    metadata = dataset_metadata(path)
    return metadata == identity_from_config(cfg, domain)


def metadata_array(cfg: Any, domain: str | None = None) -> np.ndarray:
    return np.asarray(json.dumps(identity_from_config(cfg, domain)), dtype=object)
