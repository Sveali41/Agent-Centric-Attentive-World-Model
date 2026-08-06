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
