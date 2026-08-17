"""Single-environment Crafter Plan2Explore collection.

This module intentionally has no curriculum, archive, PPO, learning-progress,
or count-novelty path.  The collector is the DreamerV2 online ratio: random
prefill, uniform episodic replay, one WM/ensemble/imagined actor update every
five real interactions.
"""

from __future__ import annotations

from pathlib import Path
import random
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import INVENTORY_SLOTS
from modelBased.common.artifacts import single_environment_dataset_path, world_model_checkpoint_path
from modelBased.data.data_collect import save_dataset_coverage, save_experiments
from modelBased.exploration.p2e import AttentionWMP2EAdapter, DreamerP2EActorCritic, EpisodeReplay, P2EEnsemble
from modelBased.world_model.AttentionWM import AttentionWorldModel


def split_collection_budget(total_steps: int, num_phases: int) -> list[int]:
    total_steps, num_phases = int(total_steps), int(num_phases)
    if total_steps <= 0 or num_phases <= 0: raise ValueError("budgets must be positive")
    base, remainder = divmod(total_steps, num_phases)
    return [base + (i < remainder) for i in range(num_phases)]


def _cfg_with_paths(cfg: DictConfig, obs: dict, data_path: Path) -> DictConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw["attention_model"]["grid_shape"] = [2, int(obs["image"].shape[0]), int(obs["image"].shape[1])]
    raw["attention_model"]["data_dir"] = str(data_path)
    raw["attention_model"]["model_save_path"] = str(world_model_checkpoint_path(cfg, "crafter"))
    raw["env"]["env_type"] = "crafter"
    raw["env"]["collect"]["data_save_path"] = str(data_path)
    raw["env"]["collect"]["data_type"] = "p2e"
    raw["env"]["collect"]["save_coverage_visualize"] = True
    task_name = str(raw["domains"]["crafter"].get("task_name", "crafter"))
    raw["env"]["collect"]["visualize_filename"] = f"{task_name}_p2e_coverage.png"
    return OmegaConf.create(raw)


def _flatten_sequence(batch: dict[str, np.ndarray], key: str) -> np.ndarray:
    value = batch[key]
    return value.reshape((-1,) + value.shape[2:])


def _save_checkpoint(path: Path, step: int, wm, adapter, ensemble, actor, replay: EpisodeReplay, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    replay_path = path.with_suffix(".replay.npz"); replay.save(replay_path)
    torch.save({"step": int(step), "world_model": wm.state_dict(), "wm_optimizer": adapter.optimizer.state_dict(),
                "ensemble": ensemble.state_dict(), "ensemble_optimizer": ensemble.optimizer.state_dict(),
                "actor_critic": actor.state_dict(), "actor_optimizer": actor.actor_optimizer.state_dict(),
                "critic_optimizer": actor.critic_optimizer.state_dict(), "replay_path": str(replay_path),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}, path)


def collect_crafter_single_p2e(cfg: DictConfig) -> list[str]:
    if str(cfg.domain) != "crafter": raise ValueError("P2E currently supports Crafter only")
    if not bool(getattr(cfg.p2e, "enabled", False)): raise ValueError("p2e.enabled must be true")
    continual = getattr(cfg.domains.crafter, "continual_learning", None)
    if continual is not None and bool(getattr(continual, "enabled", False)):
        raise ValueError("official single-environment P2E cannot run with continual_learning")

    p2e = cfg.p2e; total_steps = int(getattr(p2e, "total_steps", 1_000_000)); prefill = int(getattr(p2e, "prefill_steps", 10_000))
    seed = int(getattr(cfg.PPO, "seed", 0)); np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    data_path = single_environment_dataset_path(cfg, "crafter"); data_path.parent.mkdir(parents=True, exist_ok=True)
    layout = str(Path(str(cfg.domains.crafter.layout_path)).expanduser().resolve())
    configured_inventory = getattr(cfg.domains.crafter, "initial_inventory", {})
    if not isinstance(configured_inventory, dict):
        configured_inventory = OmegaConf.to_container(configured_inventory, resolve=True)
    env = CustomCrafterEnv(txt_file_path=layout, max_steps=int(cfg.domains.crafter.data_collection.max_steps), seed=seed,
                           initial_inventory=configured_inventory or {})
    obs, _ = env.reset()
    run_cfg = _cfg_with_paths(cfg, obs, data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = AttentionWorldModel(run_cfg.attention_model).to(device)
    adapter = AttentionWMP2EAdapter(wm, inventory_dim=int(cfg.domains.crafter.inventory_dim), inventory_scale=float(p2e.inventory_scale), amp=bool(getattr(p2e, "amp", True)))
    state_dim = int(cfg.attention_model.embed_dim) + int(cfg.domains.crafter.inventory_dim); action_dim = int(cfg.attention_model.action_norm_values)
    ensemble = P2EEnsemble(state_dim, action_dim, num_models=int(p2e.num_models), learning_rate=float(p2e.ensemble_lr), device=device)
    actor = DreamerP2EActorCritic(state_dim, action_dim, actor_lr=float(p2e.actor_lr), critic_lr=float(p2e.critic_lr), entropy_coef=float(p2e.actor_entropy), gamma=float(p2e.gamma), discount_lambda=float(p2e.discount_lambda), horizon=int(p2e.imagined_horizon), slow_target_update=int(p2e.slow_target_update), device=device)
    replay = EpisodeReplay(int(p2e.replay_capacity)); rng = np.random.default_rng(seed)
    checkpoint = Path(str(getattr(p2e, "checkpoint_path", world_model_checkpoint_path(cfg, "crafter")))).expanduser().resolve()
    start_step = 0
    if bool(getattr(p2e, "resume", True)) and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        wm.load_state_dict(saved["world_model"]); adapter.optimizer.load_state_dict(saved["wm_optimizer"])
        ensemble.load_state_dict(saved["ensemble"]); ensemble.optimizer.load_state_dict(saved["ensemble_optimizer"])
        actor.load_state_dict(saved["actor_critic"]); actor.actor_optimizer.load_state_dict(saved["actor_optimizer"]); actor.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        replay_file = Path(str(saved.get("replay_path", "")))
        if replay_file.is_file(): replay = EpisodeReplay.load(replay_file, int(p2e.replay_capacity))
        start_step = int(saved.get("step", 0))
        if saved.get("numpy_rng") is not None: rng.bit_generator.state = saved["numpy_rng"]
        if saved.get("torch_rng") is not None: torch.set_rng_state(saved["torch_rng"])
        if saved.get("python_rng") is not None: random.setstate(saved["python_rng"])
        print(f"[Crafter P2E] Resuming online state at step {start_step}")
    rewards=[]; dones=[]; infos=[]; observations=[]; observations_next=[]; actions=[]; inventories=[]; inventories_next=[]; update_history=[]
    train_every, batch_size, seq_len = int(p2e.train_every), int(p2e.replay_batch_size), int(p2e.replay_sequence_length)
    target_mode = next((str(field.get("target_mode", "absolute")) for field in cfg.domains.crafter.observation_schema if str(field.get("name", "")) == "inventory"), "absolute")
    last_obs = obs
    for step in range(start_step, total_steps):
        if step < prefill: action = int(env.action_space.sample())
        else: action = actor.act(adapter.encode_states(last_obs["image"], last_obs["inventory"]))
        next_obs, native_reward, terminated, truncated, info = env.step(action); done = bool(terminated or truncated)
        replay.add(last_obs["image"], next_obs["image"], action, native_reward, done, info, last_obs["inventory"], next_obs["inventory"])
        observations.append(last_obs["image"]); observations_next.append(next_obs["image"]); actions.append(action); rewards.append(native_reward); dones.append(done); infos.append(info); inventories.append(last_obs["inventory"]); inventories_next.append(next_obs["inventory"])
        last_obs = env.reset()[0] if done else next_obs
        real_step = step + 1
        if real_step >= prefill and real_step % train_every == 0 and replay.ready:
            sample = replay.sample(batch_size, seq_len, rng)
            flat = {key: _flatten_sequence(sample, key) for key in ("obs", "obs_next", "act", "rew", "done", "inv", "inv_next")}
            wm_loss = adapter.train_batch({"obs": flat["obs"], "obs_next": flat["obs_next"], "act": flat["act"], "inv": flat["inv"], "inv_next": flat["inv_next"], "info": None})
            current_state = adapter.encode_states(flat["obs"], flat["inv"]); next_state = adapter.encode_states(flat["obs_next"], flat["inv_next"])
            ensemble_loss = ensemble.train_batch(current_state, flat["act"], next_state)
            images_chw = torch.as_tensor(flat["obs"], device=device).permute(0, 3, 1, 2).float()
            actor_metrics = actor.imagine_update(current_state, images_chw, torch.as_tensor(flat["inv"], device=device), adapter, ensemble, target_mode=target_mode, start_terminal=torch.as_tensor(flat["done"], device=device))
            update_history.append({"step": real_step, "wm_loss": wm_loss, "ensemble_loss": ensemble_loss, **actor_metrics})
            if real_step % max(1, int(getattr(p2e, "log_every", 1000))) == 0:
                print(
                    f"[Crafter P2E] step={real_step} wm={wm_loss:.5f} ensemble={ensemble_loss:.5f} "
                    f"disagreement={actor_metrics['imagined_reward_mean']:.6f} "
                    f"actor_loss={actor_metrics['actor_loss']:.5f} "
                    f"critic_loss={actor_metrics['critic_loss']:.5f} "
                    f"entropy={actor_metrics['imagined_entropy']:.4f}"
                )
        if real_step % int(getattr(p2e, "checkpoint_every", 100_000)) == 0:
            _save_checkpoint(checkpoint, real_step, wm, adapter, ensemble, actor, replay, rng)
    env.close()
    exported = replay.export_dict()
    inventory_array = np.asarray(exported["inv_next"], dtype=np.float32)
    unique_inventory = int(np.unique(inventory_array, axis=0).shape[0]) if len(inventory_array) else 0
    presence = {
        name: bool(np.any(inventory_array[:, index] > 0))
        for index, name in enumerate(INVENTORY_SLOTS)
        if len(inventory_array) and index < inventory_array.shape[1]
    }
    print(f"[Crafter P2E] final unique_inventory_states={unique_inventory} presence={presence}")
    def _action_summary(values):
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        counts = np.bincount(values, minlength=action_dim).astype(np.float64) if len(values) else np.zeros(action_dim)
        probabilities = counts / max(counts.sum(), 1.0)
        nonzero = probabilities > 0
        entropy = float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()) if nonzero.any() else 0.0
        return {"entropy": entropy, "effective_actions": float(np.exp(entropy)), "coverage": float(nonzero.mean()), "counts": counts.astype(int).tolist()}
    all_action_summary = _action_summary(exported["act"])
    explorer_action_summary = _action_summary(exported["act"][int(prefill):])
    metrics_path = data_path.with_suffix(".p2e_metrics.json")
    metrics_path.write_text(json.dumps({
        "artifact": "p2e_official_dv2_attnwm_v1", "total_steps": int(total_steps), "prefill_steps": int(prefill),
        "unique_inventory_states": unique_inventory, "inventory_presence": presence,
        "all_action_summary": all_action_summary, "explorer_action_summary": explorer_action_summary,
        "updates": update_history,
    }, indent=2), encoding="utf-8")
    print(f"[Crafter P2E] metrics saved to {metrics_path}")
    save_experiments(run_cfg, exported["obs"], exported["obs_next"], exported["act"], exported["rew"], exported["done"], exported["info"], inv=exported["inv"], inv_next=exported["inv_next"])
    wm_path = world_model_checkpoint_path(cfg, "crafter"); wm_path.parent.mkdir(parents=True, exist_ok=True); torch.save(wm.state_dict(), wm_path)
    _save_checkpoint(checkpoint, total_steps, wm, adapter, ensemble, actor, replay, rng)
    save_dataset_coverage(run_cfg, data_path=data_path)
    return [str(data_path)]


def collect_crafter_curriculum(cfg: DictConfig) -> list[str]:
    raise ValueError("Official P2E does not support curriculum collection; disable continual_learning")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    collect_crafter_single_p2e(cfg)


if __name__ == "__main__": train()
