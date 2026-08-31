"""Persistent RMax acquisition over an ordered Crafter curriculum."""

from __future__ import annotations

import json
from pathlib import Path
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from domain.crafter.crafter_reward import INVENTORY_SLOTS
from modelBased.common.artifacts import continual_phase_data_path, dataset_matches
from modelBased.data.data_collect import save_dataset_coverage, save_experiments
from modelBased.exploration.count_based import CrafterStateActionCounter, build_crafter_policy_state
from modelBased.exploration.crafter_rmax_collect import _make_ppo
from modelBased.policy_training.crafter_vector_env import CrafterSubprocessVectorEnv


def _phase_cfg(cfg: DictConfig, phase: dict, data_path: Path) -> DictConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    domain = raw["domains"]["crafter"]
    domain["task_name"] = str(phase["task_name"])
    domain["layout_path"] = str(Path(phase["layout_path"]).expanduser().resolve())
    domain["data_save_path"] = str(data_path)
    domain["initial_inventory"] = dict(phase.get("initial_inventory") or {})
    raw["env"]["env_path"] = domain["layout_path"]
    raw["env"]["collect"]["data_save_path"] = str(data_path)
    raw["env"]["collect"]["data_type"] = "rmax"
    raw["env"]["collect"]["visualize_filename"] = f"{phase['name']}_rmax_coverage.png"
    return OmegaConf.create(raw)


def _save_state(path: Path, ppo, counter: CrafterStateActionCounter, phase_index: int, total_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ppo": ppo.policy_old.state_dict(), "counts": dict(counter._counts),
                "phase_index": int(phase_index), "total_steps": int(total_steps)}, path)


def collect_crafter_rmax_continual(cfg: DictConfig) -> list[str]:
    if str(cfg.domain) != "crafter":
        raise ValueError("Continual RMax currently supports Crafter only")
    if not bool(getattr(cfg.rmax_like, "enabled", False)):
        raise ValueError("rmax_like.enabled must be true")
    continual = cfg.domains.crafter.continual_learning
    if not bool(getattr(continual, "enabled", False)):
        raise ValueError("continual_learning.enabled must be true")
    phases = [OmegaConf.to_container(item, resolve=True) for item in continual.phases]
    if not phases:
        raise ValueError("No continual phases configured")
    total_steps = int(getattr(continual, "total_collection_steps", cfg.domains.crafter.data_collection.maximum_dataset_size))
    num_envs = int(cfg.rmax_like.num_envs)
    base, rem = divmod(total_steps, len(phases))
    budgets = [base + (index < rem) for index in range(len(phases))]
    if any(budget % num_envs for budget in budgets):
        raise ValueError(f"Continual phase budgets {budgets} must be divisible by num_envs={num_envs}")
    train_steps = int(cfg.rmax_like.train_steps)
    frozen_steps = int(cfg.rmax_like.frozen_steps)
    if train_steps + frozen_steps != total_steps:
        raise ValueError("rmax_like.train_steps + frozen_steps must equal continual total_collection_steps")
    seed = int(cfg.rmax_like.seed); np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    counter = CrafterStateActionCounter(mask_size=int(cfg.rmax_like.mask_size), reward_scale=float(cfg.rmax_like.reward_scale))
    ppo = None; state_dim = None; global_step = 0; phase_outputs=[]
    configured_checkpoint = getattr(cfg.rmax_like, "continual_checkpoint_path", None)
    checkpoint = Path(str(configured_checkpoint)) if configured_checkpoint is not None else Path("")
    if str(configured_checkpoint).strip() in {"", ".", "None", "null"}:
        base_checkpoint = Path(str(cfg.rmax_like.checkpoint_path)).expanduser()
        checkpoint = base_checkpoint.with_name(base_checkpoint.stem + "_continual.pt")
    resume = bool(getattr(continual, "resume", False)) and checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if resume else None
    start_phase = 0
    if saved is not None:
        global_step = int(saved.get("total_steps", 0)); counter._counts.update(saved.get("counts", {}))
        start_phase = int(saved.get("phase_index", 0))

    for phase_index, (phase, budget) in enumerate(zip(phases, budgets)):
        if phase_index < start_phase:
            phase_outputs.append(str(continual_phase_data_path(cfg, phase, "crafter")))
            continue
        data_path = continual_phase_data_path(cfg, phase, "crafter")
        phase_cfg = _phase_cfg(cfg, phase, data_path)
        layout = str(Path(str(phase["layout_path"])).expanduser().resolve())
        initial_inventory = {str(k): float(v) for k, v in dict(phase.get("initial_inventory") or {}).items()}
        envs = CrafterSubprocessVectorEnv(layout, int(cfg.domains.crafter.data_collection.max_steps), num_envs, seed + phase_index * num_envs, str(cfg.rmax_like.start_method), initial_inventory=initial_inventory)
        arrays = {key: [] for key in ("obs", "obs_next", "act", "rew", "done", "info", "inv", "inv_next")}
        try:
            obs, _ = envs.reset()
            initial_state = build_crafter_policy_state(obs["image"], obs["inventory"], cfg.attention_model.obs_norm_values)
            if ppo is None:
                state_dim = int(initial_state.shape[1]); ppo = _make_ppo(cfg, state_dim)
                if saved is not None: ppo.policy_old.load_state_dict(saved["ppo"]); ppo.policy.load_state_dict(saved["ppo"])
            for _ in range(budget // num_envs):
                state = build_crafter_policy_state(obs["image"], obs["inventory"], cfg.attention_model.obs_norm_values)
                actions, states_buf, actions_buf, logprobs_buf, values_buf = ppo.select_action_batch(state)
                action_np = actions.detach().cpu().numpy().reshape(-1)
                next_obs, native_rewards, terminated, truncated, infos = envs.step(action_np)
                dones = np.asarray(terminated) | np.asarray(truncated)
                count_rewards = counter.rewards(obs["image"], obs["inventory"], action_np, terminated=terminated, update=True)
                if float(cfg.rmax_like.death_penalty) > 0:
                    count_rewards[np.asarray(terminated)] = -float(cfg.rmax_like.death_penalty)
                arrays["obs"].append(obs["image"].copy()); arrays["obs_next"].append(next_obs["image"].copy()); arrays["act"].append(action_np.astype(np.int64)); arrays["rew"].append(np.asarray(native_rewards, dtype=np.float32)); arrays["done"].append(np.asarray(terminated, dtype=np.bool_)); arrays["info"].append(np.asarray(infos, dtype=object)); arrays["inv"].append(obs["inventory"].copy()); arrays["inv_next"].append(next_obs["inventory"].copy())
                training = global_step < train_steps
                if training:
                    ppo.save_buffer_batch(states_buf, actions_buf, logprobs_buf, values_buf, count_rewards, dones)
                    if ppo.buffer.transition_count() >= int(cfg.rmax_like.rollout_steps):
                        bootstrap = ppo.estimate_old_values_batch(build_crafter_policy_state(next_obs["image"], next_obs["inventory"], cfg.attention_model.obs_norm_values))
                        bootstrap[torch.as_tensor(terminated)] = 0.0
                        ppo.update(bootstrap_value=bootstrap)
                global_step += num_envs
                obs = {"image": next_obs["image"].copy(), "inventory": next_obs["inventory"].copy()}
                if np.any(dones):
                    reset_obs, _ = envs.reset_at(np.flatnonzero(dones)); obs["image"][dones] = reset_obs["image"]; obs["inventory"][dones] = reset_obs["inventory"]
            if ppo.buffer.transition_count() > 0 and global_step >= train_steps:
                ppo.update(bootstrap_value=0.0)
        finally:
            envs.close()
        merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
        save_experiments(phase_cfg, merged["obs"], merged["obs_next"], merged["act"], merged["rew"], merged["done"], merged["info"], merged["inv"], merged["inv_next"])
        save_dataset_coverage(phase_cfg, data_path=data_path)
        phase_outputs.append(str(data_path)); _save_state(checkpoint, ppo, counter, phase_index + 1, global_step)
        print(f"[Crafter RMax continual] phase={phase_index + 1}/{len(phases)} steps={len(merged['act'])} unique_sa={counter.unique_state_actions} data={data_path}")
    configured_metrics = getattr(cfg.rmax_like, "continual_metrics_path", None)
    metrics_path = (
        Path(str(configured_metrics)).expanduser().resolve()
        if configured_metrics is not None
        else Path(str(cfg.paths.results)).expanduser().resolve()
        / "exploration"
        / f"{checkpoint.stem}.metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"total_steps": global_step, "unique_state_actions": counter.unique_state_actions, "phases": phase_outputs}, indent=2), encoding="utf-8")
    return phase_outputs


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    collect_crafter_rmax_continual(cfg)


if __name__ == "__main__":
    main()
