"""Collect fixed-budget Crafter datasets with random or P2E acquisition."""

from __future__ import annotations

import tempfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import INVENTORY_SLOTS
from modelBased.common.artifact_naming import (
    continual_phase_data_path,
    single_environment_dataset_path,
    world_model_checkpoint_path,
)
from modelBased.common.dataset_identity import dataset_matches
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.data.data_collect import run_env, save_experiments
from modelBased.exploration.p2e import CrafterP2EExplorer, P2EEnsemble
from modelBased.policy_training.PPO import PPO
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel


def split_collection_budget(total_steps: int, num_phases: int) -> list[int]:
    """Split one global budget exactly, distributing any remainder up front."""
    total_steps = int(total_steps)
    num_phases = int(num_phases)
    if total_steps <= 0 or num_phases <= 0:
        raise ValueError("total_steps and num_phases must both be positive")
    base, remainder = divmod(total_steps, num_phases)
    return [base + (1 if index < remainder else 0) for index in range(num_phases)]


def _phase_config(cfg: DictConfig, phase: dict, budget: int, seed: int) -> DictConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    domain = str(raw["domain"])
    domain_cfg = raw["domains"][domain]
    domain_cfg["task_name"] = str(phase["task_name"])
    domain_cfg["layout_path"] = str(Path(phase["layout_path"]).expanduser().resolve())
    domain_cfg["data_save_path"] = str(Path(phase["data_dir"]).expanduser().resolve())
    domain_cfg["validation_data_dir"] = None
    domain_cfg["initial_inventory"] = {
        str(key): float(value)
        for key, value in dict(phase.get("initial_inventory") or {}).items()
    }

    raw["PPO"]["seed"] = int(seed)
    raw["env"]["env_type"] = domain
    raw["env"]["env_path"] = domain_cfg["layout_path"]
    raw["env"]["max_steps"] = int(raw["domains"][domain]["data_collection"]["max_steps"])
    collect_cfg = raw["env"]["collect"]
    collect_cfg["data_type"] = "random"
    collect_cfg["episodes"] = 0
    collect_cfg["mini_dataset_size"] = int(budget)
    collect_cfg["maximum_dataset_size"] = int(budget)
    collect_cfg["data_save_path"] = domain_cfg["data_save_path"]
    collect_cfg["visualize_filename"] = f"{phase['name']}_coverage.png"
    collect_cfg["save_coverage_visualize"] = False

    return OmegaConf.create(raw)


def _dataset_size(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return int(len(data["a"]))
    except (OSError, KeyError, ValueError):
        return None


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.asarray([])
    return np.concatenate(parts, axis=0)


def _p2e_train_config(
    phase_cfg: DictConfig,
    checkpoint_dir: Path,
    *,
    channels: int,
    height: int,
    width: int,
) -> DictConfig:
    """Create a non-consolidating WM config for one P2E acquisition update."""
    raw = OmegaConf.to_container(phase_cfg, resolve=True)
    p2e_cfg = raw["p2e"]
    attention = raw["attention_model"]
    attention["data_dir"] = raw["env"]["collect"]["data_save_path"]
    attention["validation_data_dir"] = None
    attention["grid_shape"] = [int(channels), int(height), int(width)]
    attention["continue_learning"] = True
    attention["compute_fisher"] = False
    attention["freeze_weight"] = False
    attention["use_wandb"] = False
    attention["enable_progress_bar"] = bool(
        p2e_cfg.get("show_wm_progress", False)
    )
    attention["n_epochs"] = int(p2e_cfg.get("wm_epochs_per_cycle", 3))
    attention["checkpoint_dir"] = str(checkpoint_dir)
    attention["model_save_path"] = str(checkpoint_dir / "exploration_wm.ckpt")
    return OmegaConf.create(raw)


def _p2e_states(
    explorer: CrafterP2EExplorer,
    images: np.ndarray,
    inventories: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    states = []
    for start in range(0, len(images), max(1, int(batch_size))):
        state = explorer.encode_batch(
            images[start : start + batch_size],
            inventories[start : start + batch_size],
        )
        states.append(state.detach().cpu())
    return torch.cat(states, dim=0) if states else torch.empty((0, 0))


def _canonical_p2e_batch(batch: dict) -> dict:
    """Convert collector letter keys to raw replay keys."""
    return {
        "obs": batch["a"],
        "obs_next": batch["b"],
        "act": batch["c"],
        "rew": batch.get("d"),
        "done": batch.get("e"),
        "info": batch.get("f"),
        "inv": batch.get("g"),
        "inv_next": batch.get("h"),
    }


def _p2e_merge_raw_batches(left: dict, right: dict) -> dict:
    """Merge raw replay and current transitions without latent caching."""
    merged = {}
    for key in set(left) | set(right):
        values = [value for value in (left.get(key), right.get(key)) if value is not None]
        if not values:
            continue
        if len(values) == 1:
            merged[key] = values[0]
        else:
            try:
                merged[key] = np.concatenate(values, axis=0)
            except (TypeError, ValueError):
                merged[key] = values[-1]
    return merged


def _p2e_progression_counts(batch: dict) -> dict:
    inventory = batch.get("g")
    inventory_next = batch.get("h")
    if inventory is None or inventory_next is None or len(inventory) == 0:
        return {}
    delta = np.asarray(inventory_next) - np.asarray(inventory)
    result = {
        name: int(np.count_nonzero(delta[:, index]))
        for index, name in enumerate(INVENTORY_SLOTS)
        if index < delta.shape[1]
    }
    tool_start = 10
    result["tools_gained"] = {
        name: int(np.count_nonzero(delta[:, index] > 0.5))
        for index, name in enumerate(INVENTORY_SLOTS[tool_start:], start=tool_start)
        if index < delta.shape[1]
    }
    result["tools_present_end"] = {
        name: float(np.asarray(inventory_next)[-1, index])
        for index, name in enumerate(INVENTORY_SLOTS[tool_start:], start=tool_start)
        if index < delta.shape[1]
    }
    observations = batch.get("a")
    observations_next = batch.get("b")
    actions = np.asarray(batch.get("c", []))
    if observations is not None and observations_next is not None:
        current = np.asarray(observations)
        following = np.asarray(observations_next)
        if current.ndim == 4 and following.shape == current.shape:
            if current.shape[-1] == 2 and current.shape[1] != 2:
                current_objects = current[..., 0]
                following_objects = following[..., 0]
            elif current.shape[1] == 2 and current.shape[-1] != 2:
                current_objects = current[:, 0]
                following_objects = following[:, 0]
            else:
                current_objects = following_objects = None
            if current_objects is not None:
                result["tables_placed"] = int(
                    np.count_nonzero(
                        (following_objects == 11) & (current_objects != 11)
                    )
                )
                result["furnaces_placed"] = int(
                    np.count_nonzero(
                        (following_objects == 12) & (current_objects != 12)
                    )
                )
    if actions.size:
        result["place_table_actions"] = int(np.count_nonzero(actions == 8))
        result["place_furnace_actions"] = int(np.count_nonzero(actions == 9))
    return result


def _collect_crafter_curriculum_p2e(
    cfg: DictConfig,
    phases: list[dict],
    budgets: list[int],
    base_seed: int,
    *,
    final_checkpoint: Path | None = None,
) -> list[str]:
    """Collect P2E data with bootstrap, online WM and imagined actor updates."""
    p2e_cfg = cfg.p2e
    cycles = max(1, int(getattr(p2e_cfg, "cycles_per_phase", 24)))
    bootstrap_steps = max(1, int(getattr(p2e_cfg, "bootstrap_steps", 4000)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exploration_wm = AttentionWorldModel(cfg.attention_model).to(device)
    replay = FisherReplayBuffer(max_size=int(getattr(p2e_cfg, "replay_buffer_size", 30000)))
    explorer = None
    ensemble = None
    outputs: list[str] = []

    with tempfile.TemporaryDirectory(prefix="agebt_p2e_wm_") as temp_dir:
        temp_root = Path(temp_dir)
        for phase_index, (phase, budget) in enumerate(zip(phases, budgets)):
            layout_path = Path(str(phase["layout_path"])).expanduser().resolve()
            data_path = Path(str(phase["data_dir"])).expanduser().resolve()
            budget = int(budget)
            if explorer is None:
                if budget <= bootstrap_steps:
                    raise ValueError("The first phase budget must exceed bootstrap_steps")
                chunk_budgets = [bootstrap_steps] + split_collection_budget(
                    budget - bootstrap_steps, cycles
                )
            else:
                chunk_budgets = split_collection_budget(budget, cycles)
            phase_parts = {key: [] for key in "abcdefgh"}
            initial_inventory = {
                str(key): float(value)
                for key, value in dict(phase.get("initial_inventory") or {}).items()
            }
            env = CustomCrafterEnv(
                txt_file_path=str(layout_path),
                max_steps=int(cfg.domains.crafter.data_collection.max_steps),
                seed=base_seed + phase_index,
                initial_inventory=initial_inventory,
            )

            for cycle_index, chunk_budget in enumerate(chunk_budgets):
                is_bootstrap = explorer is None
                phase_cfg = _phase_config(cfg, phase, chunk_budget, base_seed + phase_index)
                phase_cfg.env.collect.data_type = "p2e"
                print(
                    f"[Crafter P2E] Phase {phase_index + 1}/{len(phases)} "
                    f"{'bootstrap' if is_bootstrap else 'cycle'} {cycle_index + 1}: collect {chunk_budget}"
                )
                if is_bootstrap:
                    batch = run_env(
                        env, phase_cfg, wandb_run=None,
                        log_name=f"p2e_{phase['name']}_bootstrap",
                        policy=None, intrinsic_reward_fn=None,
                        store_intrinsic_reward=False, save_img=False,
                        randomize_inventory=False,
                    )
                else:
                    exploration_wm.eval()
                    ensemble.eval()
                    batch = run_env(
                        env, phase_cfg, wandb_run=None,
                        log_name=f"p2e_{phase['name']}_{cycle_index}",
                        policy=explorer,
                        intrinsic_reward_fn=lambda obs, action, obs_next: explorer.compute_intrinsic_reward(obs, action, obs_next),
                        store_intrinsic_reward=False, save_img=False,
                        randomize_inventory=False,
                    )
                obs, obs_next, actions, rewards, done, info, inv, inv_next = batch
                if len(obs) != chunk_budget:
                    raise RuntimeError(
                        f"P2E phase {phase['name']} produced {len(obs)} transitions; expected {chunk_budget}"
                    )
                for key, value in zip("abcdefgh", batch):
                    phase_parts[key].append(value)
                cumulative = {key: _concat(value) for key, value in phase_parts.items()}
                save_experiments(
                    phase_cfg, cumulative["a"], cumulative["b"], cumulative["c"],
                    cumulative["d"], cumulative["e"], cumulative["f"],
                    inv=cumulative["g"], inv_next=cumulative["h"],
                )
                direct_data = {
                    "a": obs, "b": obs_next, "c": actions, "d": rewards,
                    "e": done, "f": info, "g": inv, "h": inv_next,
                }
                previous_replay = replay.export_dict() if len(replay) else None
                height, width = int(obs.shape[1]), int(obs.shape[2])
                train_cfg = _p2e_train_config(
                    phase_cfg,
                    temp_root / f"phase_{phase_index}_cycle_{cycle_index}",
                    channels=2, height=height, width=width,
                )
                train_cfg.attention_model.n_epochs = int(
                    getattr(p2e_cfg, "bootstrap_wm_epochs", 10)
                    if is_bootstrap else getattr(p2e_cfg, "wm_epochs_per_cycle", 3)
                )
                exploration_wm.train()
                _, _, exploration_wm = AttentionWM_training.train_api(
                    train_cfg, net=exploration_wm, old_params=None, fisher=None,
                    replay_data=previous_replay, direct_data=direct_data,
                )
                exploration_wm.to(device).eval()

                if is_bootstrap:
                    p2e_state_dim = int(cfg.attention_model.embed_dim) + int(cfg.domains.crafter.inventory_dim)
                    ensemble = P2EEnsemble(
                        p2e_state_dim, int(cfg.attention_model.action_norm_values),
                        num_models=int(getattr(p2e_cfg, "num_models", 10)),
                        learning_rate=float(getattr(p2e_cfg, "disag_lr", 1e-4)),
                        target_type=str(getattr(p2e_cfg, "target_type", "delta")),
                        bootstrap_heads=bool(getattr(p2e_cfg, "bootstrap_heads", True)),
                        device=device,
                    )
                    ppo = PPO(
                        state_dim=p2e_state_dim,
                        action_dim=int(cfg.attention_model.action_norm_values),
                        lr_actor=float(getattr(p2e_cfg, "ppo_lr_actor", 3e-4)),
                        lr_critic=float(getattr(p2e_cfg, "ppo_lr_critic", 1e-3)),
                        gamma=float(getattr(p2e_cfg, "ppo_gamma", 0.99)),
                        K_epochs=int(getattr(p2e_cfg, "ppo_k_epochs", 2)),
                        eps_clip=float(getattr(p2e_cfg, "ppo_eps_clip", 0.2)),
                        has_continuous_action_space=False,
                        entropy_coef=float(getattr(p2e_cfg, "ppo_entropy_coef", 0.03)),
                        normalize_advantages=bool(getattr(p2e_cfg, "ppo_normalize_advantages", True)),
                        normalize_returns=bool(getattr(p2e_cfg, "ppo_normalize_returns", False)),
                        max_grad_norm=float(getattr(p2e_cfg, "ppo_max_grad_norm", 0.5)),
                    )
                    explorer = CrafterP2EExplorer(ppo, exploration_wm, ensemble, cfg)
                    explorer.set_representation_anchor(obs[:256], inv[:256])
                    raw_view = _canonical_p2e_batch(direct_data)
                    initial_epochs = int(getattr(p2e_cfg, "bootstrap_disag_epochs", 5))
                else:
                    explorer.set_world_model(exploration_wm)
                    # Re-encode every historical raw sample plus the complete
                    # current chunk. The bounded replay insertion below may
                    # retain only a saliency-weighted subset for the next WM
                    # update, but ensemble fitting must see this full target.
                    raw_view = _p2e_merge_raw_batches(
                        previous_replay or {}, _canonical_p2e_batch(direct_data)
                    )
                    initial_epochs = int(getattr(p2e_cfg, "disag_epochs", 1))

                encode_batch_size = int(getattr(p2e_cfg, "encode_batch_size", 512))
                state = _p2e_states(explorer, raw_view["obs"], raw_view["inv"], encode_batch_size)
                next_state = _p2e_states(explorer, raw_view["obs_next"], raw_view["inv_next"], encode_batch_size)
                ensemble_loss = ensemble.train_step(
                    state, raw_view["act"], next_state,
                    epochs=initial_epochs,
                    batch_size=int(getattr(p2e_cfg, "batch_size", 256)),
                )
                with torch.no_grad():
                    prediction = ensemble.predictions(state, raw_view["act"])
                    state_device = state.to(ensemble.device)
                    next_state_device = next_state.to(ensemble.device)
                    target = (
                        next_state_device - state_device
                        if ensemble.target_type == "delta"
                        else next_state_device
                    )
                    ensemble_prediction_error = float(
                        (prediction - target.unsqueeze(0)).abs().mean().cpu()
                    )
                drift = {"representation_cosine": 1.0, "representation_l2": 0.0}
                if not is_bootstrap:
                    drift = explorer.measure_representation_drift()
                actor_metrics = explorer.adapt_actor_imagined(raw_view["obs"], raw_view["inv"])
                replay.add_from_batch(
                    direct_data,
                    current_sample_ratio=1.0 if is_bootstrap else float(getattr(p2e_cfg, "replay_add_ratio", 0.5)),
                    fisher_buffer_elements_ratio=float(getattr(p2e_cfg, "salient_sample_ratio", 0.5)),
                )
                metrics = explorer.metrics(reset=True)
                progression = _p2e_progression_counts(direct_data)
                print(
                    f"[Crafter P2E] ensemble_loss={ensemble_loss:.6f} "
                    f"prediction_error={ensemble_prediction_error:.6f} "
                    f"intrinsic={metrics['intrinsic_reward_mean']:.6f} "
                    f"imagined_reward={actor_metrics['imagined_reward_mean']:.6f} "
                    f"disagreement_p50={actor_metrics['imagined_reward_p50']:.6f} "
                    f"imagined_delta={actor_metrics['imagined_parameter_delta']:.6e} "
                    f"action_entropy={actor_metrics['imagined_entropy']:.4f} "
                    f"real_action_entropy={metrics['real_action_entropy']:.4f} "
                    f"representation_cosine={drift['representation_cosine']:.5f} "
                    f"encoder_version={metrics['representation_version']} "
                    f"replay={len(replay)} progression={progression}"
                )

            env.close()
            outputs.append(str(data_path))
            print(f"[Crafter P2E] Saved phase '{phase['name']}' with {_dataset_size(data_path)} transitions: {data_path}")

        if final_checkpoint is not None:
            if len(phases) != 1 or len(outputs) != 1:
                raise ValueError(
                    "Final P2E WM consolidation currently supports one environment"
                )
            data_path = Path(outputs[0])
            with np.load(data_path, allow_pickle=True) as loaded:
                full_data = {
                    key: loaded[key]
                    for key in "abcdefgh"
                    if key in loaded.files
                }
            observations = full_data["a"]
            if observations.ndim != 4:
                raise ValueError(
                    f"Expected four-dimensional P2E observations, got "
                    f"{observations.shape}"
                )
            if observations.shape[1] == 2:
                height, width = observations.shape[2:4]
            elif observations.shape[-1] == 2:
                height, width = observations.shape[1:3]
            else:
                raise ValueError(
                    f"Cannot identify Crafter channels in {observations.shape}"
                )

            final_phase_cfg = _phase_config(
                cfg, phases[0], len(observations), base_seed
            )
            final_phase_cfg.env.collect.data_type = "p2e"
            final_cfg = _p2e_train_config(
                final_phase_cfg,
                temp_root / "final_consolidation",
                channels=2,
                height=int(height),
                width=int(width),
            )
            final_cfg.attention_model.n_epochs = int(
                getattr(p2e_cfg, "final_wm_epochs", 5)
            )
            final_checkpoint = final_checkpoint.expanduser().resolve()
            final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            final_cfg.attention_model.model_save_path = str(final_checkpoint)
            print(
                f"[Crafter P2E] Consolidating the same online WM for "
                f"{final_cfg.attention_model.n_epochs} epochs over "
                f"all {len(observations)} transitions."
            )
            exploration_wm.train()
            _, _, exploration_wm = AttentionWM_training.train_api(
                final_cfg,
                net=exploration_wm,
                old_params=None,
                fisher=None,
                replay_data=None,
                direct_data=full_data,
            )
            if not final_checkpoint.is_file():
                raise RuntimeError(
                    f"P2E consolidation did not create the final WM: "
                    f"{final_checkpoint}"
                )
            print(f"[Crafter P2E] Final planning WM: {final_checkpoint}")

    return outputs


def collect_crafter_single_p2e(cfg: DictConfig) -> list[str]:
    """Run P2E acquisition on only the currently selected Crafter layout."""
    if str(cfg.domain) != "crafter":
        raise ValueError("Single-environment P2E currently supports Crafter only")
    if not bool(getattr(getattr(cfg, "p2e", None), "enabled", False)):
        raise ValueError("Single-environment P2E requires p2e.enabled=true")

    domain_cfg = cfg.domains.crafter
    continual_cfg = getattr(domain_cfg, "continual_learning", None)
    if continual_cfg is not None and bool(getattr(continual_cfg, "enabled", False)):
        raise ValueError(
            "Single-environment P2E requires continual_learning.enabled=false"
        )

    budget = int(domain_cfg.data_collection.maximum_dataset_size)
    seed = int(getattr(cfg.PPO, "seed", 0))
    data_path = single_environment_dataset_path(cfg, "crafter")
    phase = {
        "name": str(domain_cfg.task_name),
        "task_name": str(domain_cfg.task_name),
        "layout_path": str(Path(str(domain_cfg.layout_path)).expanduser().resolve()),
        "data_dir": str(data_path),
        "initial_inventory": OmegaConf.to_container(
            getattr(domain_cfg, "initial_inventory", OmegaConf.create({})),
            resolve=True,
        ),
    }
    phase_cfg = _phase_config(cfg, phase, budget, seed)
    final_checkpoint = world_model_checkpoint_path(cfg, "crafter")
    force = bool(
        continual_cfg is not None
        and getattr(continual_cfg, "force_collection", False)
    )
    if (
        not force
        and _dataset_size(data_path) == budget
        and dataset_matches(data_path, phase_cfg, "crafter")
        and final_checkpoint.is_file()
    ):
        print(
            f"[Crafter P2E] Reusing single-environment dataset and WM: "
            f"{data_path}, {final_checkpoint}"
        )
        return [str(data_path)]

    return _collect_crafter_curriculum_p2e(
        cfg,
        [phase],
        [budget],
        seed,
        final_checkpoint=final_checkpoint,
    )


def collect_crafter_curriculum(cfg: DictConfig) -> list[str]:
    domain = str(cfg.domain)
    if domain != "crafter":
        raise ValueError("The fixed-inventory curriculum collector currently supports Crafter only")

    domain_cfg = cfg.domains[domain]
    continual_cfg = domain_cfg.continual_learning
    phases = [OmegaConf.to_container(item, resolve=True) for item in continual_cfg.phases]
    for phase in phases:
        phase["data_dir"] = str(
            continual_phase_data_path(cfg, phase, domain)
        )
    required_count = getattr(continual_cfg, "required_phase_count", None)
    if required_count is not None and len(phases) != int(required_count):
        raise ValueError(
            f"Expected exactly {required_count} Crafter curriculum phases, got {len(phases)}"
        )

    total_steps = int(
        getattr(
            continual_cfg,
            "total_collection_steps",
            domain_cfg.data_collection.maximum_dataset_size,
        )
    )
    budgets = split_collection_budget(total_steps, len(phases))
    base_seed = int(getattr(cfg.PPO, "seed", 0))
    force = bool(getattr(continual_cfg, "force_collection", False))
    outputs = []

    p2e_enabled = bool(
        hasattr(cfg, "p2e") and getattr(cfg.p2e, "enabled", False)
    )
    if p2e_enabled:
        ready = all(
            _dataset_size(Path(str(phase["data_dir"])).expanduser().resolve())
            == budget
            and dataset_matches(
                Path(str(phase["data_dir"])).expanduser().resolve(),
                _phase_config(cfg, phase, budget, base_seed + index),
                domain,
            )
            for index, (phase, budget) in enumerate(zip(phases, budgets))
        )
        if ready and not force:
            reused = [
                str(Path(str(phase["data_dir"])).expanduser().resolve())
                for phase in phases
            ]
            print("[Crafter P2E] Reusing the complete P2E curriculum dataset set.")
            return reused
        # P2E is one stateful acquisition process: if any phase is stale,
        # recollect every phase so its actor/ensemble trajectory is coherent.
        return _collect_crafter_curriculum_p2e(
            cfg, phases, budgets, base_seed
        )

    for index, phase in enumerate(phases):
        budget = budgets[index]
        data_path = Path(str(phase["data_dir"])).expanduser().resolve()
        layout_path = Path(str(phase["layout_path"])).expanduser().resolve()
        if not layout_path.is_file():
            raise FileNotFoundError(f"Curriculum layout not found: {layout_path}")

        phase_cfg = _phase_config(cfg, phase, budget, base_seed + index)
        if not force and _dataset_size(data_path) == budget and dataset_matches(data_path, phase_cfg, domain):
            print(f"[Crafter Curriculum] Reusing phase {index + 1}: {data_path}")
            outputs.append(str(data_path))
            continue

        print(
            f"[Crafter Curriculum] Collecting phase {index + 1}/{len(phases)} "
            f"'{phase['name']}' with {budget} transitions; "
            f"initial_inventory={phase.get('initial_inventory') or {}}"
        )
        initial_inventory = {
            str(key): float(value)
            for key, value in dict(phase.get("initial_inventory") or {}).items()
        }
        env = CustomCrafterEnv(
            txt_file_path=str(layout_path),
            max_steps=int(domain_cfg.data_collection.max_steps),
            seed=base_seed + index,
            initial_inventory=initial_inventory,
        )
        obs, obs_next, actions, rewards, done, info, inv, inv_next = run_env(
            env,
            phase_cfg,
            wandb_run=None,
            log_name=f"collect_{phase['name']}",
            save_img=False,
            randomize_inventory=False,
        )
        if len(obs) != budget:
            env.close()
            raise RuntimeError(
                f"Phase {phase['name']} produced {len(obs)} transitions, expected exactly {budget}"
            )
        save_experiments(
            phase_cfg,
            obs,
            obs_next,
            actions,
            rewards,
            done,
            info,
            inv=inv,
            inv_next=inv_next,
        )
        env.close()
        outputs.append(str(data_path))
        print(f"[Crafter Curriculum] Saved {data_path}")

    return outputs


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    continual_cfg = getattr(cfg.domains.crafter, "continual_learning", None)
    continual_enabled = bool(
        continual_cfg is not None
        and getattr(continual_cfg, "enabled", False)
    )
    if continual_enabled:
        collect_crafter_curriculum(cfg)
    else:
        collect_crafter_single_p2e(cfg)


if __name__ == "__main__":
    train()
