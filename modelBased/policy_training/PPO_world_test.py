"""Evaluate a trained compact-action PPO policy in the real MiniGrid env."""

from __future__ import annotations

import os
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
from minigrid.wrappers import FullyObsWrapper
from omegaconf import DictConfig

from domain.minigrid.action_codec import (
    COMPACT_ACTION_NAMES,
    INVENTORY_TOKEN_COUNT,
    MODEL_ACTION_COUNT,
    carrying_token_from_env,
    compact_to_native,
)
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from domain.minigrid.minigrid_support import ColRowCanl_to_CanlRowCol
from modelBased.common.utils import normalize_obs
from modelBased.policy_training.PPO import PPO
from modelBased.policy_training.experiment_naming import (
    policy_checkpoint_is_compatible,
    policy_checkpoint_path,
    policy_training_source,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_eval_action(ppo_agent: PPO, state: torch.Tensor, deterministic: bool) -> int:
    """Select a compact action without writing to the PPO rollout buffer."""
    if deterministic:
        with torch.no_grad():
            probabilities = ppo_agent.policy_old.actor(state.to(device))
        return int(torch.argmax(probabilities, dim=-1).item())
    action, _, _, _, _ = ppo_agent.select_action(state)
    return int(action)


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "modelBased/config"),
    config_name="config",
)
def test(cfg: DictConfig) -> None:
    validate_policy(cfg)


def validate_policy(cfg: DictConfig) -> float:
    if str(cfg.domain).lower() != "minigrid":
        raise ValueError("PPO_world_test currently supports the MiniGrid policy only.")

    ppo_cfg = cfg.PPO
    checkpoint_path = policy_checkpoint_path(cfg)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    if not policy_checkpoint_is_compatible(checkpoint_path, cfg):
        raise RuntimeError(
            "Policy checkpoint uses an old observation/action shape. Retrain "
            f"the current inventory-aware policy: {checkpoint_path}"
        )

    render = bool(ppo_cfg.render)
    save_gif = bool(ppo_cfg.save_gif)
    save_csv = bool(ppo_cfg.save_csv)
    deterministic = bool(getattr(ppo_cfg, "test_deterministic", True))
    render_delay = float(getattr(ppo_cfg, "render_delay", 0.05))
    gif_fps = int(getattr(ppo_cfg, "gif_fps", 10))
    total_episodes = int(ppo_cfg.total_test_episodes)
    max_ep_len = int(ppo_cfg.max_ep_len)

    # Human mode opens a live window. RGB mode is used for headless/GIF runs.
    render_mode = "human" if render else "rgb_array"
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=str(ppo_cfg.env_path),
            custom_mission="Reach the goal.",
            max_steps=max_ep_len,
            render_mode=render_mode,
        )
    )

    state_dim = int(np.prod(env.observation_space["image"].shape)) + INVENTORY_TOKEN_COUNT
    ppo_agent = PPO(
        state_dim,
        MODEL_ACTION_COUNT,
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
    print(f"Real layout:   {ppo_cfg.env_path}")
    print(f"Evaluation:    {'deterministic' if deterministic else 'stochastic'}")
    ppo_agent.load(str(checkpoint_path))
    ppo_agent.policy_old.eval()

    gif_dir = Path(str(ppo_cfg.save_path_gif)).expanduser().resolve()
    csv_dir = Path(str(ppo_cfg.save_path_csv)).expanduser().resolve()
    if save_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_reward = 0.0
    for episode in range(1, total_episodes + 1):
        observation, _ = env.reset()
        state = ColRowCanl_to_CanlRowCol(observation["image"])
        episode_reward = 0.0
        frames = []
        if save_gif and not render and episode == 1:
            frames.append(env.render())

        for step in range(1, max_ep_len + 1):
            # normalize_obs mutates its input, so preserve the raw discrete state.
            normalized = normalize_obs(state.copy(), cfg.attention_model.obs_norm_values)
            state_tensor = torch.as_tensor(
                normalized, dtype=torch.float32, device=device
            ).flatten()
            inventory = torch.nn.functional.one_hot(
                torch.as_tensor(
                    carrying_token_from_env(env), device=device
                ).long(),
                num_classes=INVENTORY_TOKEN_COUNT,
            ).float()
            state_tensor = torch.cat((state_tensor, inventory), dim=0)
            compact_action = _select_eval_action(
                ppo_agent, state_tensor, deterministic
            )
            native_action = compact_to_native(compact_action)
            observation, reward, terminated, truncated, _ = env.step(native_action)
            episode_reward += float(reward)
            state = ColRowCanl_to_CanlRowCol(observation["image"])

            if render:
                env.render()
                print(
                    f"\rEpisode {episode} | step {step} | "
                    f"action {compact_action}:{COMPACT_ACTION_NAMES[compact_action]} | "
                    f"reward {episode_reward:.5f}",
                    end="",
                    flush=True,
                )
                if render_delay > 0:
                    time.sleep(render_delay)
            elif save_gif and episode == 1:
                frames.append(env.render())

            if terminated or truncated:
                break

        if render:
            print()
        success = episode_reward > 0.0
        total_reward += episode_reward
        results.append((episode, step, episode_reward, success))
        print(
            f"Episode {episode}: steps={step}, reward={episode_reward:.5f}, "
            f"success={success}"
        )

        if save_gif and not render and episode == 1 and frames:
            gif_path = gif_dir / "ppo_real_env_test.gif"
            imageio.mimsave(gif_path, frames, fps=gif_fps)
            print(f"Saved GIF: {gif_path}")

    env.close()

    if save_csv:
        csv_path = csv_dir / "ppo_real_env_test.csv"
        pd.DataFrame(
            results, columns=["episode", "steps", "reward", "success"]
        ).to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    average_reward = total_reward / max(total_episodes, 1)
    success_rate = sum(row[3] for row in results) / max(total_episodes, 1)
    print(f"Average real-environment reward: {average_reward:.5f}")
    print(f"Success rate: {success_rate:.1%}")
    return average_reward


if __name__ == "__main__":
    test()
