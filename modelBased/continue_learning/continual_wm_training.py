"""Continual world-model training over an ordered sequence of environments.

This is the current replacement for the legacy curriculum-learning entry
point. A single AttentionWorldModel is carried through every phase. Each phase
trains on its current transitions plus a bounded replay buffer. EWC and Fisher
estimation are intentionally disabled for this shared-dynamics curriculum.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from modelBased.common.artifacts import dataset_matches, dataset_metadata
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.common.artifacts import (
    continual_artifact_path,
    continual_phase_data_path,
    world_model_checkpoint_path,
)


STATE_VERSION = 1


@dataclass(frozen=True)
class ContinualPhase:
    index: int
    name: str
    task_name: str
    layout_path: str
    data_dir: str
    validation_data_dir: str | None
    initial_inventory: dict[str, float]
    metadata: dict[str, Any] | None
    legacy_dataset: bool
    shape: tuple[int, int, int]
    transitions: int
    fingerprint: dict[str, Any]


def _resolved_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _dataset_shape(path: str | Path, expected_channels: int) -> tuple[int, int, int, int]:
    with np.load(path, allow_pickle=True) as data:
        if "a" not in data.files:
            raise KeyError(f"Dataset has no observation array 'a': {path}")
        observations = np.asarray(data["a"])

    if observations.ndim != 4:
        raise ValueError(
            f"Expected image observations with four dimensions, got "
            f"{observations.shape} in {path}"
        )

    if observations.shape[1] == expected_channels:
        _, channels, height, width = observations.shape
    elif observations.shape[-1] == expected_channels:
        _, height, width, channels = observations.shape
    else:
        raise ValueError(
            f"Cannot identify {expected_channels}-channel observations in "
            f"{path}; shape={observations.shape}"
        )
    return int(channels), int(height), int(width), int(len(observations))


def _phase_cfg(base_cfg: DictConfig, phase: ContinualPhase, max_shape: tuple[int, int]) -> DictConfig:
    """Build a resolved config whose identity matches one phase dataset."""
    raw = _resolved_container(base_cfg)
    domain = str(raw["domain"])
    domain_cfg = raw["domains"][domain]
    domain_cfg["task_name"] = phase.task_name
    domain_cfg["layout_path"] = phase.layout_path
    domain_cfg["data_save_path"] = phase.data_dir
    domain_cfg["validation_data_dir"] = phase.validation_data_dir
    domain_cfg["initial_inventory"] = dict(phase.initial_inventory)

    attention_cfg = raw["attention_model"]
    attention_cfg["data_dir"] = phase.data_dir
    attention_cfg["validation_data_dir"] = phase.validation_data_dir
    attention_cfg["freeze_weight"] = False
    attention_cfg["continue_learning"] = True
    attention_cfg["allow_legacy_dataset"] = bool(phase.legacy_dataset)
    attention_cfg["grid_shape"] = [phase.shape[0], max_shape[0], max_shape[1]]

    continual_cfg = domain_cfg.get("continual_learning", {})
    attention_cfg["replay_frac"] = float(continual_cfg.get("replay_frac", 0.5))
    attention_cfg["ewc_enabled"] = bool(continual_cfg.get("ewc_enabled", False))

    return OmegaConf.create(raw)


def _load_phase_data(path: str | Path, max_shape: tuple[int, int], expected_channels: int) -> dict[str, np.ndarray]:
    """Load a phase and canonicalize its map observations to padded NCHW."""
    with np.load(path, allow_pickle=True) as loaded:
        data = {key: loaded[key] for key in loaded.files if key != "metadata"}

    required = ("a", "b", "c")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset {path} is missing required arrays: {missing}")

    def to_nchw(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim != 4:
            raise ValueError(f"Expected a four-dimensional observation array, got {array.shape}")
        if array.shape[1] == expected_channels:
            return array
        if array.shape[-1] == expected_channels:
            return np.moveaxis(array, -1, 1)
        raise ValueError(
            f"Cannot canonicalize observation shape {array.shape}; "
            f"expected {expected_channels} channels"
        )

    def pad(array: np.ndarray) -> np.ndarray:
        height, width = array.shape[-2:]
        target_h, target_w = max_shape
        if height > target_h or width > target_w:
            raise ValueError(
                f"Phase observation {array.shape} exceeds global map shape {max_shape}"
            )
        if (height, width) == max_shape:
            return array
        return np.pad(
            array,
            ((0, 0), (0, 0), (0, target_h - height), (0, target_w - width)),
            mode="constant",
            constant_values=0,
        )

    data["a"] = pad(to_nchw(data["a"]))
    data["b"] = pad(to_nchw(data["b"]))
    length = len(data["a"])
    for key in ("b", "c", "d", "e", "f", "g", "h"):
        if key in data and len(data[key]) != length:
            raise ValueError(f"Dataset field '{key}' has length {len(data[key])}, expected {length}")
    return data


def _samples_for_replay(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map dataset short keys to the canonical replay-buffer names."""
    mapping = {"a": "obs", "b": "obs_next", "c": "act", "d": "rew", "e": "done", "f": "info", "g": "inv", "h": "inv_next"}
    samples = {}
    for source, target in mapping.items():
        if source in data:
            samples[target] = data[source]
    return samples


def _architecture_signature(cfg: DictConfig, channels: int) -> dict[str, Any]:
    domain_cfg = cfg.domains[str(cfg.domain)]
    schema = OmegaConf.to_container(domain_cfg.observation_schema, resolve=True)
    return {
        "env_type": str(cfg.attention_model.env_type),
        "model_type": str(cfg.attention_model.model_type),
        "data_type": str(cfg.attention_model.data_type),
        "channels": int(channels),
        "attention_mask_size": int(cfg.attention_model.attention_mask_size),
        "action_norm_values": int(cfg.attention_model.action_norm_values),
        "inventory_dim": int(getattr(domain_cfg, "inventory_dim", 0)),
        "observation_schema": schema,
        "continual_regularizer": "replay_only_v1",
    }


def _phase_fingerprint(phase: ContinualPhase) -> dict[str, Any]:
    return {
        "name": phase.name,
        "task_name": phase.task_name,
        "layout_path": phase.layout_path,
        "data": phase.fingerprint,
        "shape": list(phase.shape),
        "transitions": phase.transitions,
        "metadata": phase.metadata,
        "initial_inventory": phase.initial_inventory,
    }


def _resolve_phases(cfg: DictConfig) -> list[ContinualPhase]:
    domain = str(cfg.domain)
    domain_cfg = cfg.domains[domain]
    continual_cfg = getattr(domain_cfg, "continual_learning", None)
    if continual_cfg is None or not bool(getattr(continual_cfg, "enabled", False)):
        raise ValueError(
            f"domains.{domain}.continual_learning.enabled must be true for this entry point"
        )

    configured = list(getattr(continual_cfg, "phases", []))
    if not configured:
        raise ValueError(f"No continual-learning phases configured for domain '{domain}'")

    expected_channels = int(domain_cfg.grid_shape[0])
    phases: list[ContinualPhase] = []
    allow_legacy = bool(getattr(continual_cfg, "allow_legacy_dataset", False))

    for index, raw_phase in enumerate(configured):
        phase = OmegaConf.to_container(raw_phase, resolve=True)
        data_path = continual_phase_data_path(cfg, phase, domain)
        if not data_path.is_file():
            raise FileNotFoundError(f"Phase {index} dataset not found: {data_path}")

        metadata = dataset_metadata(data_path)
        task_name = phase.get("task_name") or (metadata or {}).get("task_name")
        layout_path = phase.get("layout_path") or (metadata or {}).get("layout_path")
        if not task_name or not layout_path:
            if not allow_legacy:
                raise ValueError(
                    f"Phase {index} dataset has no identity metadata. Set "
                    "allow_legacy_dataset=true and provide task_name/layout_path explicitly."
                )
            raise ValueError(f"Legacy phase {index} must provide task_name and layout_path")

        layout_path = str(Path(layout_path).expanduser().resolve())
        if not Path(layout_path).is_file():
            raise FileNotFoundError(f"Phase {index} layout not found: {layout_path}")
        if metadata is not None:
            if metadata.get("domain") != domain:
                raise ValueError(
                    f"Phase {index} dataset domain={metadata.get('domain')!r} does not match {domain!r}"
                )
            phase_cfg = _phase_cfg(cfg, ContinualPhase(
                index=index,
                name=str(phase.get("name", f"phase_{index}")),
                task_name=str(task_name),
                layout_path=layout_path,
                data_dir=str(data_path),
                validation_data_dir=None,
                initial_inventory={
                    str(key): float(value)
                    for key, value in dict(phase.get("initial_inventory", (metadata or {}).get("initial_inventory", {}))).items()
                },
                metadata=metadata,
                legacy_dataset=False,
                shape=(expected_channels, 1, 1),
                transitions=0,
                fingerprint={},
            ), (1, 1))
            if not dataset_matches(data_path, phase_cfg, domain):
                raise ValueError(
                    f"Phase {index} dataset identity does not match task/layout configuration: {data_path}"
                )

        channels, height, width, transitions = _dataset_shape(data_path, expected_channels)
        if channels != expected_channels:
            raise ValueError(
                f"Phase {index} has {channels} channels, expected {expected_channels}"
            )
        validation_data_dir = phase.get("validation_data_dir")
        if validation_data_dir:
            validation_data_dir = str(Path(validation_data_dir).expanduser().resolve())
            if not Path(validation_data_dir).is_file():
                raise FileNotFoundError(f"Phase {index} validation dataset not found: {validation_data_dir}")

        phases.append(ContinualPhase(
            index=index,
            name=str(phase.get("name", f"phase_{index}")),
            task_name=str(task_name),
            layout_path=layout_path,
            data_dir=str(data_path),
            validation_data_dir=validation_data_dir,
            initial_inventory={
                str(key): float(value)
                for key, value in dict(phase.get("initial_inventory", (metadata or {}).get("initial_inventory", {}))).items()
            },
            metadata=metadata,
            legacy_dataset=metadata is None,
            shape=(channels, height, width),
            transitions=transitions,
            fingerprint=_file_fingerprint(data_path),
        ))
    return phases


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_state_dict(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if state is None:
        return None
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _save_replay(buffer: FisherReplayBuffer, path: str | Path) -> None:
    data = buffer.export_dict() if len(buffer) else {}
    _atomic_torch_save({"version": STATE_VERSION, "data": data}, path)


def _load_replay(buffer: FisherReplayBuffer, path: str | Path) -> None:
    replay_path = Path(path).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"Continual-learning replay file not found: {replay_path}")
    payload = torch.load(replay_path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported replay state version in {replay_path}")
    data = payload.get("data", {})
    if data:
        buffer.load_from_dict(data)


def _restore_rng(payload: dict[str, Any]) -> None:
    if payload.get("python_rng") is not None:
        random.setstate(payload["python_rng"])
    if payload.get("numpy_rng") is not None:
        np.random.set_state(payload["numpy_rng"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def _state_payload(
    cfg: DictConfig,
    phases: list[ContinualPhase],
    architecture: dict[str, Any],
    next_phase_index: int,
    net: AttentionWorldModel,
    old_params: dict[str, torch.Tensor] | None,
    fisher: dict[str, torch.Tensor] | None,
    history: list[dict[str, Any]],
    active_phase_index: int | None = None,
    active_fit_ckpt: str | None = None,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "domain": str(cfg.domain),
        "next_phase_index": int(next_phase_index),
        "active_phase_index": active_phase_index,
        "active_fit_ckpt": active_fit_ckpt,
        "completed_phases": [_phase_fingerprint(phase) for phase in phases[:next_phase_index]],
        "phase_fingerprints": [_phase_fingerprint(phase) for phase in phases],
        "model_state_dict": _cpu_state_dict(net.state_dict()),
        "old_params": _cpu_state_dict(old_params),
        "fisher": _cpu_state_dict(fisher),
        "phase_history": history,
        "architecture_signature": architecture,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(f"Continual-learning state not found: {state_path}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported continual-learning state version: {state_path}")
    return payload


def _validate_resume_state(
    state: dict[str, Any],
    cfg: DictConfig,
    phases: list[ContinualPhase],
    architecture: dict[str, Any],
) -> None:
    if state.get("domain") != str(cfg.domain):
        raise ValueError("Continual state domain does not match the selected domain")
    if state.get("architecture_signature") != architecture:
        raise ValueError("Continual state architecture does not match the current configuration")

    next_phase = int(state.get("next_phase_index", 0))
    saved = state.get("completed_phases", [])
    current = [_phase_fingerprint(phase) for phase in phases]
    if next_phase > len(phases) or len(saved) != next_phase:
        raise ValueError("Continual state phase index is incompatible with configured phases")
    if saved != current[:next_phase]:
        raise ValueError(
            "Configured phases changed before the resume point. Keep completed phases "
            "in the same order and append new phases at the end."
        )


def _extract_validation_loss(result: dict[str, Any]) -> float:
    value = result.get("avg_val_loss")
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = value[0].get("val/observation_loss", value[0].get("val_loss", 0.0))
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, (list, tuple)):
        return float(np.asarray(value, dtype=np.float64).mean())
    return float(value or 0.0)


def _write_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in history:
        for validation in entry.get("validations", []):
            rows.append({
                "phase_index": entry["phase_index"],
                "phase_name": entry["phase_name"],
                "validated_phase_index": validation["phase_index"],
                "validated_phase_name": validation["phase_name"],
                "transitions": entry["transitions"],
                "validation_loss": validation["loss"],
                "best_validation_loss": validation.get("best_loss"),
                "forgetting_delta": validation.get("forgetting_delta"),
                "replay_size": entry["replay_size"],
            })
    fieldnames = [
        "phase_index", "phase_name", "validated_phase_index", "validated_phase_name",
        "transitions", "validation_loss", "best_validation_loss", "forgetting_delta",
        "replay_size",
    ]
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _set_map_shape(net: AttentionWorldModel, channels: int, max_shape: tuple[int, int]) -> None:
    net.channel = int(channels)
    net.row, net.col = int(max_shape[0]), int(max_shape[1])
    net.loss_accumulator = [[[] for _ in range(net.col)] for _ in range(net.row)]


def run_continual(cfg: DictConfig) -> dict[str, Any]:
    domain_cfg = cfg.domains[str(cfg.domain)]
    continual_cfg = domain_cfg.continual_learning
    final_checkpoint = world_model_checkpoint_path(cfg, str(cfg.domain))
    force = bool(getattr(getattr(cfg, "pipeline", {}), "force", False))
    reuse_final = bool(getattr(continual_cfg, "skip_if_final_checkpoint_exists", True))
    if reuse_final and final_checkpoint.is_file() and not force:
        configured_phases = list(getattr(continual_cfg, "phases", []))
        print(
            "[Continual WM] Final checkpoint already exists; skipping all "
            f"continual phases: {final_checkpoint}"
        )
        return {
            "next_phase_index": len(configured_phases),
            "history": [],
            "final_checkpoint": str(final_checkpoint),
            "skipped": True,
        }
    phases = _resolve_phases(cfg)
    channels = phases[0].shape[0]
    if any(phase.shape[0] != channels for phase in phases):
        raise ValueError("All continual-learning phases must use the same channel count")
    max_shape = (
        max(phase.shape[1] for phase in phases),
        max(phase.shape[2] for phase in phases),
    )
    architecture = _architecture_signature(cfg, channels)

    state_path = continual_artifact_path(cfg, "state_path", str(cfg.domain))
    replay_path = continual_artifact_path(cfg, "replay_path", str(cfg.domain))
    metrics_path = continual_artifact_path(cfg, "metrics_path", str(cfg.domain))
    if state_path is None or replay_path is None:
        raise ValueError("Continual learning requires state_path and replay_path")
    if metrics_path is None:
        metrics_path = state_path.with_suffix(".csv")

    # Configure the model once for the largest map used by the curriculum.
    cfg.attention_model.grid_shape = [channels, max_shape[0], max_shape[1]]
    net = AttentionWorldModel(cfg.attention_model)
    _set_map_shape(net, channels, max_shape)

    buffer = FisherReplayBuffer(
        max_size=int(getattr(continual_cfg, "replay_buffer_size", 100000))
    )
    old_params = None
    fisher = None
    history: list[dict[str, Any]] = []
    next_phase_index = 0
    active_fit_ckpt = None

    if bool(getattr(continual_cfg, "resume", True)) and state_path.is_file():
        state = _load_state(state_path)
        _validate_resume_state(state, cfg, phases, architecture)
        next_phase_index = int(state["next_phase_index"])
        if state.get("old_params") is not None or state.get("fisher") is not None:
            raise ValueError("Replay-only continual state must not contain EWC/Fisher tensors")
        old_params = None
        fisher = None
        history = list(state.get("phase_history", []))
        if state.get("model_state_dict"):
            net.load_state_dict(state["model_state_dict"], strict=True)
        if replay_path.is_file():
            _load_replay(buffer, replay_path)
        elif len(history) > 0:
            raise FileNotFoundError(
                f"Replay state is required for resume but is missing: {replay_path}"
            )
        active_index = state.get("active_phase_index")
        if active_index is not None and int(active_index) == next_phase_index:
            active_fit_ckpt = state.get("active_fit_ckpt")
        _restore_rng(state)
        print(f"[Continual WM] Resuming at phase {next_phase_index}/{len(phases)}")
    elif state_path.is_file():
        print(f"[Continual WM] Ignoring existing state because resume=false: {state_path}")

    if next_phase_index >= len(phases):
        print("[Continual WM] All configured phases are already complete.")
        return {"next_phase_index": next_phase_index, "history": history, "state_path": str(state_path)}

    for index in range(next_phase_index, len(phases)):
        phase = phases[index]
        phase_cfg = _phase_cfg(cfg, phase, max_shape)
        # Replay-only continual learning never performs the expensive
        # per-sample Fisher backward passes.
        phase_cfg.attention_model.ewc_enabled = False
        phase_cfg.attention_model.compute_fisher = False
        phase_dir = state_path.parent / f"{state_path.stem}_phase_{index}"
        phase_cfg.attention_model.checkpoint_dir = str(phase_dir)
        # Do not expose an intermediate phase as the canonical final model.
        # The final artifact is copied only after every phase completes.
        phase_cfg.attention_model.model_save_path = str(phase_dir / "phase_model.ckpt")
        data = _load_phase_data(phase.data_dir, max_shape, channels)
        replay_data = buffer.export_dict() if len(buffer) else None
        fit_ckpt = active_fit_ckpt if index == next_phase_index else None

        # Persist the previous phase before training.  If this phase is
        # interrupted, its last.ckpt can be resumed without advancing the
        # completed-phase pointer.
        _atomic_torch_save(
            _state_payload(
                cfg, phases, architecture, index, net, old_params, fisher, history,
                active_phase_index=index,
                active_fit_ckpt=str(phase_dir / "last.ckpt"),
            ),
            state_path,
        )
        if len(buffer):
            _save_replay(buffer, replay_path)

        print(
            f"[Continual WM] Phase {index + 1}/{len(phases)}: {phase.name} "
            f"({phase.transitions} transitions; replay={len(buffer)})"
        )
        result, fisher, net = AttentionWM_training.train_api(
            phase_cfg,
            net=net,
            old_params=old_params,
            fisher=fisher,
            replay_data=replay_data,
            direct_data=data,
            fit_ckpt_path=fit_ckpt,
        )
        old_params = result.get("old_params")
        if old_params is not None or fisher is not None:
            raise RuntimeError("Replay-only continual training produced unexpected EWC state")
        _set_map_shape(net, channels, max_shape)

        buffer.update_combined(
            _samples_for_replay(data),
            current_sample_ratio=float(getattr(continual_cfg, "current_sample_ratio", 0.3)),
            fisher_buffer_elements_ratio=float(getattr(continual_cfg, "salient_sample_ratio", 0.5)),
        )

        validations = []
        if bool(getattr(continual_cfg, "validate_after_each_phase", True)):
            best_by_phase = {
                int(item["phase_index"]): float(item["best_validation_loss"])
                for item in history
                for _ in [0]
                if "best_validation_loss" in item
            }
            for validated in phases[: index + 1]:
                validation_cfg = _phase_cfg(phase_cfg, validated, max_shape)
                validation_cfg.attention_model.freeze_weight = True
                validation_data_path = validated.validation_data_dir or validated.data_dir
                validation_cfg.attention_model.data_dir = validation_data_path
                validation_data = _load_phase_data(validation_data_path, max_shape, channels)
                validation_result, _, net = AttentionWM_training.train_api(
                    validation_cfg,
                    net=net,
                    old_params=old_params,
                    fisher=fisher,
                    replay_data=None,
                    direct_data=validation_data,
                )
                loss = _extract_validation_loss(validation_result)
                best_loss = min(best_by_phase.get(validated.index, loss), loss)
                validations.append({
                    "phase_index": validated.index,
                    "phase_name": validated.name,
                    "loss": loss,
                    "best_loss": best_loss,
                    "forgetting_delta": loss - best_loss,
                })
                best_by_phase[validated.index] = best_loss

        history.append({
            "phase_index": index,
            "phase_name": phase.name,
            "transitions": phase.transitions,
            "replay_size": len(buffer),
            "validations": validations,
            "best_validation_loss": next(
                (item["loss"] for item in validations if item["phase_index"] == index),
                None,
            ),
        })
        next_phase_index = index + 1
        active_fit_ckpt = None
        _save_replay(buffer, replay_path)
        _atomic_torch_save(
            _state_payload(
                cfg, phases, architecture, next_phase_index, net, old_params, fisher, history
            ),
            state_path,
        )
        _write_history_csv(history, metrics_path)

    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    final_phase_dir = state_path.parent / f"{state_path.stem}_phase_{len(phases) - 1}"
    final_source = final_phase_dir / "phase_model.ckpt"
    if not final_source.is_file():
        fallback_source = final_phase_dir / f"best-{str(cfg.domain)}.ckpt"
        if fallback_source.is_file():
            final_source = fallback_source
    if final_source.is_file():
        shutil.copy2(final_source, final_checkpoint)
    if not final_checkpoint.is_file():
        raise FileNotFoundError(
            f"Continual training completed but final WM checkpoint is missing: {final_checkpoint}"
        )
    if not bool(getattr(continual_cfg, "keep_phase_checkpoints", False)):
        for phase_index in range(len(phases)):
            phase_dir = state_path.parent / f"{state_path.stem}_phase_{phase_index}"
            if phase_dir.exists():
                shutil.rmtree(phase_dir)
                print(f"[Continual WM] Removed intermediate phase checkpoints: {phase_dir}")
    print(f"[Continual WM] Completed {len(phases)} phases. Final model: {final_checkpoint}")
    return {
        "next_phase_index": next_phase_index,
        "history": history,
        "state_path": str(state_path),
        "replay_path": str(replay_path),
        "final_checkpoint": str(final_checkpoint),
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    run_continual(cfg)


if __name__ == "__main__":
    train()
