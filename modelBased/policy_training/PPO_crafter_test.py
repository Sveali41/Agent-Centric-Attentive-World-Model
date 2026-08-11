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
from modelBased.policy_training.PPO import PPO
from modelBased.policy_training.experiment_naming import (
    policy_checkpoint_is_compatible,
    policy_checkpoint_path,
    policy_training_source,
)
from modelBased.policy_training.PPO_crafter_training import (
    CRAFTER_ACTION_COUNT,
    CRAFTER_PLAYER_ID,
    build_policy_state,
    find_success_tile_position,
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


def validate_crafter_policy(cfg: DictConfig) -> float:
    if str(cfg.domain).lower() != "crafter":
        raise ValueError("PPO_crafter_test currently supports the crafter domain only.")

    ppo_cfg = cfg.PPO
    domain_cfg = cfg.domains["crafter"]
    hparams_wm = cfg.attention_model

    checkpoint_path = policy_checkpoint_path(cfg, domain="crafter")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    if not policy_checkpoint_is_compatible(checkpoint_path, cfg, domain="crafter"):
        raise RuntimeError(
            "Crafter policy checkpoint does not match the current layout shape. "
            f"Retrain the policy for {domain_cfg.task_name}: {checkpoint_path}"
        )

    success_obj_id  = int(getattr(domain_cfg, "success_obj_id", 10))
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
    seed            = int(getattr(ppo_cfg, "seed", 0))

    seed_policy_training(seed)
    env = CustomCrafterEnv(
        txt_file_path=str(domain_cfg.layout_path),
        max_steps=max_ep_len,
        seed=seed,
    )

    # Determine state_dim from one reset
    sample_obs, _ = env.reset()
    sample_grid = sample_obs["image"]   # (H, W, 2)
    H, W = sample_grid.shape[0], sample_grid.shape[1]
    state_dim = 2 * H * W + inventory_dim

    # Locate success tile
    sample_chw = np.transpose(sample_grid, (2, 0, 1))
    goal_yx = find_success_tile_position(sample_chw, success_obj_id)
    if goal_yx is None:
        print(
            f"[Crafter Test] No target tile (obj={success_obj_id}) found; "
            "target-tile metric disabled."
        )
    else:
        print(f"[Crafter Test] Target tile (obj={success_obj_id}) at yx={goal_yx}")

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
    )
    print(f"Loading policy: {checkpoint_path}")
    print(f"Training source: {policy_training_source(cfg)}")
    ppo_agent.load(str(checkpoint_path))
    ppo_agent.policy_old.eval()

    csv_dir = Path(str(ppo_cfg.save_path_csv)).expanduser().resolve()
    gif_dir = Path(str(getattr(ppo_cfg, "save_path_gif", "outputs/PPO_gif"))).expanduser().resolve()
    if save_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_reward = 0.0

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset()
        grid = obs["image"]                                # (H, W, 2)
        state_chw = torch.as_tensor(
            np.transpose(grid, (2, 0, 1)), device=device, dtype=torch.float32
        ).unsqueeze(0)                                     # (1, 2, H, W)
        inv = torch.as_tensor(
            obs["inventory"], device=device, dtype=torch.float32
        ).unsqueeze(0)                                     # (1, 16)

        episode_reward = 0.0
        episode_achievements = 0
        success = False
        frames = []
        if save_gif and episode == 1:
            frames.append(env.render())

        for step in range(1, max_ep_len + 1):
            state_flat = build_policy_state(state_chw, inv, obs_norm_values).squeeze(0)
            action = _select_action(ppo_agent, state_flat, deterministic)

            obs, native_reward, done, trunc, info = env.step(action)
            episode_reward += float(native_reward)
            episode_achievements += len(info.get("newly_unlocked", []))
            grid = obs["image"]
            state_chw = torch.as_tensor(
                np.transpose(grid, (2, 0, 1)), device=device, dtype=torch.float32
            ).unsqueeze(0)
            inv = torch.as_tensor(
                obs["inventory"], device=device, dtype=torch.float32
            ).unsqueeze(0)

            # Check success: player stands on the success tile
            chw_np = np.transpose(obs["image"], (2, 0, 1))
            player_yx = None
            hits = np.argwhere(chw_np[0] == CRAFTER_PLAYER_ID)
            if len(hits) > 0:
                player_yx = tuple(hits[0])

            # In Crafter minitask the player *moves onto* the tile, so the obj channel
            # will show the player at goal_yx when they overlap.
            if goal_yx is not None and player_yx is not None and tuple(player_yx) == tuple(goal_yx):
                success = True

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
            f"achievements={episode_achievements}, target_success={success}"
        )

        if save_gif and episode == 1 and frames:
            task_name = str(domain_cfg.task_name)
            source = policy_training_source(cfg)
            gif_path = gif_dir / f"ppo_crafter_{task_name}_{source}_seed{seed}.gif"
            imageio.mimsave(gif_path, frames, fps=gif_fps)
            print(f"Saved GIF: {gif_path}")

    env.close()

    if save_csv:
        csv_path = csv_dir / "ppo_crafter_test.csv"
        pd.DataFrame(
            results,
            columns=["episode", "steps", "native_reward", "target_success", "newly_unlocked"],
        ).to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    avg_reward   = total_reward / max(total_episodes, 1)
    success_rate = sum(r[3] for r in results) / max(total_episodes, 1)
    achievement_rate = sum(r[4] for r in results) / max(total_episodes, 1)
    print(f"Average real-environment native reward: {avg_reward:.5f}")
    print(f"Average newly unlocked achievements: {achievement_rate:.2f}")
    print(f"Target-tile success rate: {success_rate:.1%}")
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
