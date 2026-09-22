"""Evaluate a trained Crafter PPO policy in the real Crafter environment."""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from modelBased.common import utils
from modelBased.common.utils import WM_OUTPUTS_PATH
from modelBased.common.artifacts import append_mean_row, detail_rows
from modelBased.policy_training.PPO import PPO
from modelBased.policy_training.experiment_naming import (
    policy_checkpoint_is_compatible,
    policy_checkpoint_for_evaluation,
    policy_training_source,
    policy_validation_stem,
)
from modelBased.policy_training.PPO_crafter_training import (
    CRAFTER_ACTION_COUNT,
    CRAFTER_ACHIEVEMENT_MASK_DIM,
    build_policy_state,
    crafter_achievement_mask_from_names,
    seed_policy_training,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_action(ppo_agent: PPO, state: torch.Tensor, deterministic: bool) -> int:
    if deterministic:
        with torch.no_grad():
            probs = ppo_agent.policy_old.actor(state.to(device))
        return int(torch.argmax(probs, dim=-1).item())
    action, _, _, _, _ = ppo_agent.select_action(state)
    return int(action)


def _parallel_validate_crafter_policy(
    *, cfg: DictConfig, ppo_agent: PPO, total_episodes: int, test_num_envs: int,
    max_ep_len: int, inventory_dim: int, obs_norm_values, deterministic: bool,
    seed: int,
) -> tuple[list[tuple], float]:
    """Batch policy inference over independent, reproducibly seeded episodes."""
    domain_cfg = cfg.domains["crafter"]
    slots: list[dict | None] = [None] * test_num_envs
    results: list[tuple] = []
    total_reward = 0.0
    next_episode = 1
    environment_steps = 0
    vector_steps = 0
    started = time.perf_counter()
    progress_every = max(1, int(getattr(cfg.PPO, "test_progress_every_steps", 100)))

    def reset_slot(index: int, episode: int) -> None:
        env = CustomCrafterEnv(
            txt_file_path=str(domain_cfg.layout_path), max_steps=max_ep_len,
            seed=seed + episode - 1,
        )
        obs, _ = env.reset()
        slots[index] = {
            "episode": episode, "env": env, "obs": obs, "steps": 0,
            "reward": 0.0, "achievements": 0, "success": False,
            "unlocked": torch.zeros((1, CRAFTER_ACHIEVEMENT_MASK_DIM), device=device, dtype=torch.bool),
        }

    for index in range(min(test_num_envs, total_episodes)):
        reset_slot(index, next_episode)
        next_episode += 1

    while any(slot is not None for slot in slots):
        active_indices = [index for index, slot in enumerate(slots) if slot is not None]
        active_slots = [slots[index] for index in active_indices]
        states = torch.as_tensor(
            np.stack([np.transpose(slot["obs"]["image"], (2, 0, 1)) for slot in active_slots]),
            device=device, dtype=torch.float32,
        )
        inventories = torch.as_tensor(
            np.stack([slot["obs"]["inventory"] for slot in active_slots]),
            device=device, dtype=torch.float32,
        )
        unlocked = torch.cat([slot["unlocked"] for slot in active_slots])
        policy_states = build_policy_state(states, inventories, obs_norm_values, unlocked)
        with torch.no_grad():
            probabilities = ppo_agent.policy_old.actor(policy_states)
            if deterministic:
                actions = torch.argmax(probabilities, dim=-1)
            else:
                actions = torch.distributions.Categorical(probabilities).sample()

        vector_steps += 1
        environment_steps += len(active_slots)
        for batch_index, (slot_index, slot) in enumerate(zip(active_indices, active_slots)):
            obs, native_reward, terminated, truncated, info = slot["env"].step(
                int(actions[batch_index].item())
            )
            newly_unlocked = info.get("newly_unlocked", [])
            slot["unlocked"] |= crafter_achievement_mask_from_names(
                newly_unlocked, device_=device
            ).unsqueeze(0)
            slot["obs"] = obs
            slot["steps"] += 1
            slot["reward"] += float(native_reward)
            slot["achievements"] += len(newly_unlocked)
            slot["success"] |= "collect_diamond" in newly_unlocked
            if terminated or truncated or slot["steps"] >= max_ep_len:
                results.append((
                    slot["episode"], slot["steps"], slot["reward"],
                    slot["success"], slot["achievements"],
                ))
                total_reward += slot["reward"]
                slot["env"].close()
                if next_episode <= total_episodes:
                    reset_slot(slot_index, next_episode)
                    next_episode += 1
                else:
                    slots[slot_index] = None
        if vector_steps % progress_every == 0 or len(results) == total_episodes:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[Parallel Crafter test] completed={len(results)}/{total_episodes}, "
                f"active={sum(slot is not None for slot in slots)}, "
                f"env_steps={environment_steps}, throughput={environment_steps / elapsed:.1f} steps/s",
                flush=True,
            )
    results.sort(key=lambda row: row[0])
    return results, total_reward


def validate_crafter_policy(cfg: DictConfig) -> float:
    if str(cfg.domain).lower() != "crafter":
        raise ValueError("PPO_crafter_test currently supports the crafter domain only.")

    ppo_cfg = cfg.PPO
    domain_cfg = cfg.domains["crafter"]
    hparams_wm = cfg.attention_model
    training_source = policy_training_source(cfg)

    checkpoint_path = policy_checkpoint_for_evaluation(cfg, domain="crafter")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    if not policy_checkpoint_is_compatible(checkpoint_path, cfg, domain="crafter"):
        raise RuntimeError(
            "Crafter policy checkpoint is incompatible with the current 22-bit "
            "achievement-aware state input (or the current layout shape). Legacy "
            "map+inventory policies cannot be loaded; retrain the policy for "
            f"{domain_cfg.task_name}: {checkpoint_path}"
        )

    inventory_dim   = int(getattr(domain_cfg, "inventory_dim", 16))
    obs_norm_values = list(hparams_wm.obs_norm_values)
    max_ep_len      = int(ppo_cfg.max_ep_len)
    total_episodes  = int(ppo_cfg.total_test_episodes)
    deterministic   = bool(getattr(ppo_cfg, "test_deterministic", False))
    save_csv        = bool(ppo_cfg.save_csv)
    save_gif        = bool(getattr(ppo_cfg, "save_gif", False))
    render          = bool(ppo_cfg.render)
    render_delay    = float(getattr(ppo_cfg, "render_delay", 0.05))
    gif_fps         = int(getattr(ppo_cfg, "gif_fps", 10))
    eval_seed       = int(getattr(ppo_cfg, "eval_seed", 0))
    test_num_envs   = max(1, int(getattr(ppo_cfg, "test_num_envs", 1)))

    seed_policy_training(eval_seed)
    env = CustomCrafterEnv(
        txt_file_path=str(domain_cfg.layout_path),
        max_steps=max_ep_len,
        seed=eval_seed,
    )

    # Determine state_dim from one reset
    sample_obs, _ = env.reset()
    sample_grid = sample_obs["image"]   # (H, W, 2)
    H, W = sample_grid.shape[0], sample_grid.shape[1]
    state_dim = 2 * H * W + inventory_dim + CRAFTER_ACHIEVEMENT_MASK_DIM

    # Build and load PPO agent
    ppo_agent = PPO(
        state_dim,
        CRAFTER_ACTION_COUNT,
        ppo_cfg.lr_actor,
        ppo_cfg.lr_critic,
        ppo_cfg.gamma,
        ppo_cfg.K_epochs,
        ppo_cfg.eps_clip,
        ppo_cfg.has_continuous_action_space,
        ppo_cfg.action_std,
        minibatch_size=int(getattr(ppo_cfg, "minibatch_size", 0)),
    )
    print(f"Loading policy: {checkpoint_path}")
    print(f"Training source: {training_source}")
    ppo_agent.load(str(checkpoint_path))
    ppo_agent.policy_old.eval()

    if test_num_envs > 1:
        if render or save_gif:
            env.close()
            raise ValueError(
                "PPO.test_num_envs > 1 does not support render or save_gif; "
                "set PPO.test_num_envs=1 for visual evaluation."
            )
        env.close()
        results, total_reward = _parallel_validate_crafter_policy(
            cfg=cfg, ppo_agent=ppo_agent, total_episodes=total_episodes,
            test_num_envs=test_num_envs, max_ep_len=max_ep_len,
            inventory_dim=inventory_dim, obs_norm_values=obs_norm_values,
            deterministic=deterministic, seed=eval_seed,
        )
        for episode, steps, episode_reward, success, achievements in results:
            print(
                f"Episode {episode}: steps={steps}, native_reward={episode_reward:.5f}, "
                f"achievements={achievements}, collect_diamond_success={success}"
            )
    else:
        results = []
        total_reward = 0.0

    csv_dir = Path(str(ppo_cfg.save_path_csv)).expanduser().resolve()
    gif_dir = Path(
        str(getattr(ppo_cfg, "save_path_gif", WM_OUTPUTS_PATH / "evaluation" / "ppo" / "gif"))
    ).expanduser().resolve()
    if save_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, total_episodes + 1) if test_num_envs == 1 else ():
        # CustomCrafterEnv owns its RNG seed at construction time. Recreate it
        # per episode so serial and batched evaluation use the same fixed seed
        # schedule regardless of PPO.test_num_envs.
        env.close()
        env = CustomCrafterEnv(
            txt_file_path=str(domain_cfg.layout_path),
            max_steps=max_ep_len,
            seed=eval_seed + episode - 1,
        )
        obs, _ = env.reset()
        grid = obs["image"]                                # (H, W, 2)
        state_chw = torch.as_tensor(
            np.transpose(grid, (2, 0, 1)), device=device, dtype=torch.float32
        ).unsqueeze(0)                                     # (1, 2, H, W)
        inv = torch.as_tensor(
            obs["inventory"], device=device, dtype=torch.float32
        ).unsqueeze(0)                                     # (1, 16)
        unlocked_mask = torch.zeros(
            (1, CRAFTER_ACHIEVEMENT_MASK_DIM), device=device, dtype=torch.bool
        )

        episode_reward = 0.0
        episode_achievements = 0
        success = False
        frames = []
        if save_gif and episode == 1:
            frames.append(env.render())

        for step in range(1, max_ep_len + 1):
            state_flat = build_policy_state(
                state_chw, inv, obs_norm_values, unlocked_mask
            ).squeeze(0)
            action = _select_action(ppo_agent, state_flat, deterministic)

            obs, native_reward, done, trunc, info = env.step(action)
            episode_reward += float(native_reward)
            newly_unlocked = info.get("newly_unlocked", [])
            unlocked_mask |= crafter_achievement_mask_from_names(
                newly_unlocked, device_=device
            ).unsqueeze(0)
            episode_achievements += len(newly_unlocked)
            success |= "collect_diamond" in newly_unlocked
            grid = obs["image"]
            state_chw = torch.as_tensor(
                np.transpose(grid, (2, 0, 1)), device=device, dtype=torch.float32
            ).unsqueeze(0)
            inv = torch.as_tensor(
                obs["inventory"], device=device, dtype=torch.float32
            ).unsqueeze(0)

            if render or (save_gif and episode == 1):
                frame = env.render()
            else:
                frame = None

            if render:
                time.sleep(render_delay)
            if frame is not None and save_gif and episode == 1:
                frames.append(frame)

            if done or trunc:
                break

        total_reward += episode_reward
        results.append((episode, step, episode_reward, success, episode_achievements))
        print(
            f"Episode {episode}: steps={step}, native_reward={episode_reward:.5f}, "
            f"achievements={episode_achievements}, collect_diamond_success={success}"
        )

        if save_gif and episode == 1 and frames:
            gif_path = gif_dir / f"{policy_validation_stem(cfg, 'crafter')}.gif"
            imageio.mimsave(gif_path, frames, fps=gif_fps)
            print(f"Saved GIF: {gif_path}")

    env.close()

    if save_csv:
        csv_filename = f"{policy_validation_stem(cfg, 'crafter')}.csv"
        csv_path = csv_dir / csv_filename
        current_results = pd.DataFrame(
            results,
            columns=[
                "episode",
                "steps",
                "native_reward",
                "collect_diamond_success",
                "newly_unlocked",
            ],
        )
        current_results.insert(0, "seed", eval_seed)

        # Keep one multi-seed validation artifact. Re-running a seed replaces
        # that seed's rows while preserving every other completed seed, so a
        # retry cannot silently duplicate its episodes.
        if csv_path.is_file():
            previous_results = detail_rows(pd.read_csv(csv_path))
            if "seed" in previous_results.columns:
                previous_results["seed"] = pd.to_numeric(
                    previous_results["seed"], errors="raise"
                ).astype(int)
                previous_results = previous_results.loc[
                    previous_results["seed"] != eval_seed
                ]
                current_results = pd.concat(
                    [previous_results, current_results], ignore_index=True
                )
        current_results = current_results.sort_values(
            ["seed", "episode"], kind="stable"
        )
        current_results = append_mean_row(
            current_results,
            mean_columns=[
                "steps",
                "native_reward",
                "collect_diamond_success",
                "newly_unlocked",
            ],
            labels={"seed": "all", "episode": "mean"},
        )
        temporary_path = csv_path.with_name(f".{csv_path.name}.tmp")
        current_results.to_csv(temporary_path, index=False)
        temporary_path.replace(csv_path)
        print(f"Saved multi-seed CSV: {csv_path} (seed={eval_seed})")

    avg_reward   = total_reward / max(total_episodes, 1)
    success_rate = sum(r[3] for r in results) / max(total_episodes, 1)
    achievement_rate = sum(r[4] for r in results) / max(total_episodes, 1)
    print(f"Average real-environment native reward: {avg_reward:.5f}")
    print(f"Average newly unlocked achievements: {achievement_rate:.2f}")
    print(f"Collect-diamond success rate: {success_rate:.1%}")
    return avg_reward


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "modelBased/config"),
    config_name="config",
)
def test(cfg: DictConfig) -> None:
    validate_crafter_policy(cfg)


if __name__ == "__main__":
    test()
