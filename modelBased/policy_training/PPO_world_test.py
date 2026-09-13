"""Evaluate a trained compact-action PPO policy in the real MiniGrid env."""

from __future__ import annotations

import os
import random
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
from domain.minigrid.minigrid_support import (
    ColRowCanl_to_CanlRowCol,
    stochastic_env_kwargs,
)
from modelBased.common.artifacts import append_mean_row
from modelBased.common.utils import normalize_obs
from modelBased.policy_training.PPO import PPO
from modelBased.policy_training.minigrid_dense_reward import (
    build_door_topology,
    build_goal_distance_map,
    build_goal_region_mask,
    main_dense_rewards,
    reward_settings as main_dense_reward_settings,
)
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


def _make_real_env(ppo_cfg, max_ep_len: int, stochastic_kwargs: dict):
    """Construct one independent real MiniGrid evaluation environment."""
    return FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=str(ppo_cfg.env_path),
            custom_mission="Reach the goal.",
            max_steps=max_ep_len,
            render_mode="rgb_array",
            **stochastic_kwargs,
        )
    )


def _parallel_validate_policy(
    *,
    cfg: DictConfig,
    ppo_cfg,
    ppo_agent: PPO,
    total_episodes: int,
    test_num_envs: int,
    max_ep_len: int,
    eval_seed: int,
    deterministic: bool,
    reward_mode: str,
    use_main_dense_reward: bool,
    main_reward: dict,
    save_csv: bool,
    wandb_run,
) -> float:
    """Evaluate independent episodes in parallel batches of policy inference.

    MiniGrid itself is Python/CPU based, so its ``step`` calls remain one per
    environment.  The expensive policy forward pass and dense-reward tensor
    work are batched, while each slot keeps its own seed and reward state.
    """
    envs = [
        _make_real_env(ppo_cfg, max_ep_len, stochastic_env_kwargs(cfg))
        for _ in range(test_num_envs)
    ]
    results: list[tuple] = []
    total_reward = 0.0
    next_episode = 1
    vector_steps = 0
    environment_steps = 0
    progress_every_steps = max(
        1, int(getattr(ppo_cfg, "test_progress_every_steps", 100))
    )
    evaluation_started = time.perf_counter()

    # The target layout is static across reset seeds.  Build static reward
    # structures once, exactly as the serial evaluator does per episode.
    first_observation, _ = envs[0].reset(seed=eval_seed)
    first_state = ColRowCanl_to_CanlRowCol(first_observation["image"])
    goals = np.argwhere(first_state[0] == 8)
    if len(goals) != 1:
        raise ValueError("The real MiniGrid layout must contain exactly one goal")
    goal_yx = tuple(map(int, goals[0]))
    initial_tensor = torch.as_tensor(first_state, dtype=torch.float32, device=device)
    goal_distance_map = build_goal_distance_map(initial_tensor, goal_yx)
    goal_region_mask = build_goal_region_mask(initial_tensor, goal_yx)
    door_topology = build_door_topology(initial_tensor, goal_yx)

    slots: list[dict | None] = [None] * test_num_envs

    def reset_slot(index: int, episode: int, observation=None) -> None:
        if observation is None:
            observation, _ = envs[index].reset(seed=eval_seed + episode - 1)
        state = ColRowCanl_to_CanlRowCol(observation["image"])
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        position = (state_tensor[0] == 10).nonzero(as_tuple=False)[0]
        initial_distance = goal_distance_map[position[0], position[1]]
        inventory = carrying_token_from_env(envs[index])
        seen_states = set()
        if deterministic:
            seen_states.add(
                state.tobytes() + np.asarray([inventory], dtype=np.int16).tobytes()
            )
        slots[index] = {
            "episode": episode,
            "state": state,
            "step": 0,
            "episode_reward": 0.0,
            "native_reward": 0.0,
            "shaped_reward": 0.0,
            "goal_distance_sum": 0.0,
            "goal_distance_count": 0,
            "goal_distance_min": float("inf"),
            "goal_region_entry_count": 0,
            "progress_milestone_count": 0,
            "progress_milestone_reward_sum": 0.0,
            "critical_door_crossing_count": 0,
            "critical_door_crossing_reward_sum": 0.0,
            "succeeded": False,
            "reason": "max_steps",
            "seen_states": seen_states,
            "goal_region_seen": torch.zeros(1, dtype=torch.bool, device=device),
            "rewarded_key_positions": torch.zeros(
                (1, *state_tensor.shape[-2:]), dtype=torch.bool, device=device
            ),
            "rewarded_door_positions": torch.zeros(
                (1, *state_tensor.shape[-2:]), dtype=torch.bool, device=device
            ),
            "progress_milestone_seen": torch.zeros(1, dtype=torch.bool, device=device),
            "pending_door_ids": torch.full((1,), -1, dtype=torch.long, device=device),
            "pending_door_origins": torch.full((1,), -1, dtype=torch.long, device=device),
            "rewarded_door_crossings": torch.zeros(
                (1, door_topology.num_doors), dtype=torch.bool, device=device
            ),
            "initial_distance": initial_distance.reshape(1),
            "best_distance": initial_distance.reshape(1).clone(),
        }

    initial_slots = min(test_num_envs, total_episodes)
    if initial_slots:
        # Reuse the reset used to build the static maps, rather than resetting
        # seed 0 twice.  This preserves the serial evaluator's seed sequence.
        reset_slot(0, next_episode, observation=first_observation)
        next_episode += 1
    for slot_index in range(1, initial_slots):
        reset_slot(slot_index, next_episode)
        next_episode += 1

    while any(slot is not None for slot in slots):
        vector_steps += 1
        active_indices = [i for i, slot in enumerate(slots) if slot is not None]
        active_slots = [slots[i] for i in active_indices]
        environment_steps += len(active_indices)
        previous_states = np.stack([slot["state"] for slot in active_slots])
        inventories = torch.as_tensor(
            [carrying_token_from_env(envs[i]) for i in active_indices],
            dtype=torch.long,
            device=device,
        )
        normalized = normalize_obs(
            previous_states.copy(), cfg.attention_model.obs_norm_values
        )
        state_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        state_tensor = state_tensor.flatten(start_dim=1)
        inventory_one_hot = torch.nn.functional.one_hot(
            inventories, num_classes=INVENTORY_TOKEN_COUNT
        ).float()
        state_tensor = torch.cat((state_tensor, inventory_one_hot), dim=1)
        with torch.no_grad():
            probabilities = ppo_agent.policy_old.actor(state_tensor)
            if deterministic:
                compact_actions = torch.argmax(probabilities, dim=-1)
            else:
                compact_actions = torch.distributions.Categorical(probabilities).sample()
        compact_action_values = compact_actions.detach().cpu().tolist()

        current_states = []
        native_rewards = []
        succeeded = []
        lava_terminated = []
        ended = []
        current_inventories = []
        for batch_index, (slot_index, slot) in enumerate(zip(active_indices, active_slots)):
            native_action = compact_to_native(compact_action_values[batch_index])
            observation, native_reward, terminated, truncated, _ = envs[slot_index].step(native_action)
            state = ColRowCanl_to_CanlRowCol(observation["image"])
            step_succeeded = bool(terminated and native_reward > 0.0)
            current_states.append(state)
            native_rewards.append(float(native_reward))
            succeeded.append(step_succeeded)
            lava_terminated.append(bool(terminated and not step_succeeded))
            ended.append(bool(terminated or truncated))
            current_inventories.append(carrying_token_from_env(envs[slot_index]))
            slot["state"] = state
            slot["step"] += 1
            slot["native_reward"] += float(native_reward)
            slot["succeeded"] |= step_succeeded
            if terminated:
                slot["reason"] = "goal" if step_succeeded else "terminated"
            elif truncated:
                slot["reason"] = "truncated"

        if use_main_dense_reward:
            previous_tensor = torch.as_tensor(previous_states, device=device)
            current_tensor = torch.as_tensor(np.stack(current_states), device=device)
            dense_rewards, updated_goal_region_seen, reward_events = main_dense_rewards(
                previous_tensor,
                current_tensor,
                compact_actions,
                goal_distance_map,
                goal_region_mask,
                torch.cat([slot["goal_region_seen"] for slot in active_slots]),
                torch.as_tensor(succeeded, device=device),
                torch.as_tensor(lava_terminated, device=device),
                best_goal_distances=torch.cat([slot["best_distance"] for slot in active_slots]),
                initial_goal_distances=torch.cat([slot["initial_distance"] for slot in active_slots]),
                progress_milestone_seen=torch.cat([slot["progress_milestone_seen"] for slot in active_slots]),
                previous_carrying_tokens=inventories,
                current_carrying_tokens=torch.as_tensor(
                    current_inventories, device=device
                ),
                door_topology=door_topology,
                pending_door_ids=torch.cat([slot["pending_door_ids"] for slot in active_slots]),
                pending_door_origins=torch.cat([slot["pending_door_origins"] for slot in active_slots]),
                rewarded_door_crossings=torch.cat([slot["rewarded_door_crossings"] for slot in active_slots]),
                rewarded_goal_region_key_positions=torch.cat(
                    [slot["rewarded_key_positions"] for slot in active_slots]
                ),
                rewarded_goal_region_door_positions=torch.cat(
                    [slot["rewarded_door_positions"] for slot in active_slots]
                ),
                **main_reward,
            )
        else:
            dense_rewards = torch.as_tensor(native_rewards, dtype=torch.float32, device=device)
            updated_goal_region_seen = None
            reward_events = None

        dense_reward_values = dense_rewards.detach().cpu().tolist()
        scalar_event_values = None
        if reward_events is not None:
            scalar_event_values = {
                key: reward_events[key].detach().cpu().tolist()
                for key in (
                    "current_distance",
                    "goal_region_entered",
                    "progress_milestone_crossed",
                    "progress_milestone_reward",
                    "critical_door_crossed",
                    "critical_door_crossing_reward",
                )
            }

        for batch_index, (slot_index, slot) in enumerate(zip(active_indices, active_slots)):
            dense_reward = float(dense_reward_values[batch_index])
            reward = dense_reward if reward_mode == "wm" else native_rewards[batch_index]
            slot["shaped_reward"] += dense_reward
            slot["episode_reward"] += reward
            if reward_events is not None:
                slot["goal_region_seen"] = updated_goal_region_seen[batch_index:batch_index + 1]
                slot["best_distance"] = reward_events["new_best_goal_distance"][batch_index:batch_index + 1]
                slot["progress_milestone_seen"] = reward_events["progress_milestone_seen"][batch_index:batch_index + 1]
                slot["pending_door_ids"] = reward_events["pending_door_ids"][batch_index:batch_index + 1]
                slot["pending_door_origins"] = reward_events["pending_door_origins"][batch_index:batch_index + 1]
                slot["rewarded_door_crossings"] = reward_events["rewarded_door_crossings"][batch_index:batch_index + 1]
                slot["rewarded_key_positions"] = reward_events["rewarded_goal_region_key_positions"][batch_index:batch_index + 1]
                slot["rewarded_door_positions"] = reward_events["rewarded_goal_region_door_positions"][batch_index:batch_index + 1]
                distance = float(scalar_event_values["current_distance"][batch_index])
                if np.isfinite(distance):
                    slot["goal_distance_sum"] += distance
                    slot["goal_distance_count"] += 1
                    slot["goal_distance_min"] = min(slot["goal_distance_min"], distance)
                slot["goal_region_entry_count"] += int(
                    scalar_event_values["goal_region_entered"][batch_index]
                )
                slot["progress_milestone_count"] += int(
                    scalar_event_values["progress_milestone_crossed"][batch_index]
                )
                slot["progress_milestone_reward_sum"] += float(
                    scalar_event_values["progress_milestone_reward"][batch_index]
                )
                slot["critical_door_crossing_count"] += int(
                    scalar_event_values["critical_door_crossed"][batch_index]
                )
                slot["critical_door_crossing_reward_sum"] += float(
                    scalar_event_values["critical_door_crossing_reward"][batch_index]
                )

            if not ended[batch_index] and deterministic:
                inventory = carrying_token_from_env(envs[slot_index])
                state_key = slot["state"].tobytes() + np.asarray([inventory], dtype=np.int16).tobytes()
                if state_key in slot["seen_states"]:
                    slot["reason"] = "cycle_detected"
                    ended[batch_index] = True
                else:
                    slot["seen_states"].add(state_key)
            if slot["step"] >= max_ep_len:
                ended[batch_index] = True

            if ended[batch_index]:
                mean_distance = slot["goal_distance_sum"] / max(slot["goal_distance_count"], 1)
                min_distance = slot["goal_distance_min"]
                results.append((
                    slot["episode"], slot["step"], slot["episode_reward"], slot["native_reward"],
                    slot["succeeded"], slot["reason"], mean_distance,
                    min_distance if np.isfinite(min_distance) else None,
                    slot["goal_region_entry_count"], slot["progress_milestone_count"],
                    slot["progress_milestone_reward_sum"], slot["critical_door_crossing_count"],
                    slot["critical_door_crossing_reward_sum"], slot["shaped_reward"],
                ))
                total_reward += slot["episode_reward"]
                if next_episode <= total_episodes:
                    reset_slot(slot_index, next_episode)
                    next_episode += 1
                else:
                    slots[slot_index] = None

        if vector_steps % progress_every_steps == 0 or len(results) == total_episodes:
            elapsed = max(time.perf_counter() - evaluation_started, 1e-9)
            throughput = environment_steps / elapsed
            print(
                f"[Parallel test] completed={len(results)}/{total_episodes}, "
                f"active={sum(slot is not None for slot in slots)}, "
                f"env_steps={environment_steps}, "
                f"throughput={throughput:.1f} steps/s",
                flush=True,
            )

    for env in envs:
        env.close()
    results.sort(key=lambda row: row[0])
    for row in results:
        print(
            f"Episode {row[0]}: steps={row[1]}, reward={row[2]:.5f}, "
            f"native_reward={row[3]:.5f}, wm_shaped_reward={row[13]:.5f}, "
            f"success={row[4]}, goal_region_entries={row[8]}, "
            f"progress_milestones={row[9]}, critical_door_crossings={row[11]}, reason={row[5]}"
        )
        if wandb_run is not None:
            wandb_run.log({
                "episode": row[0], "steps": row[1], "reward": row[2],
                "native_reward": row[3], "wm_shaped_reward": row[13], "success": row[4],
                "goal_dist_mean": row[6], "goal_dist_min": row[7],
                "goal_region_entry_count": row[8], "progress_milestone_count": row[9],
                "progress_milestone_reward_sum": row[10],
                "critical_door_crossing_count": row[11],
                "critical_door_crossing_reward_sum": row[12],
            }, step=row[0])

    if save_csv:
        csv_dir = Path(str(ppo_cfg.save_path_csv)).expanduser().resolve()
        csv_dir.mkdir(parents=True, exist_ok=True)
        evaluation_results = pd.DataFrame(results, columns=[
            "episode", "steps", "reward", "native_reward", "success", "reason",
            "goal_dist_mean", "goal_dist_min", "goal_region_entry_count",
            "progress_milestone_count", "progress_milestone_reward_sum",
            "critical_door_crossing_count", "critical_door_crossing_reward_sum",
            "wm_shaped_reward",
        ])
        evaluation_results = append_mean_row(
            evaluation_results,
            mean_columns=[column for column in evaluation_results.columns if column != "episode" and column != "reason"],
            labels={"episode": "mean"},
        )
        csv_path = csv_dir / "ppo_real_env_test.csv"
        evaluation_results.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    average_reward = total_reward / max(total_episodes, 1)
    success_rate = sum(row[4] for row in results) / max(total_episodes, 1)
    print(f"Average real-environment reward: {average_reward:.5f}")
    print(f"Success rate: {success_rate:.1%}")
    if wandb_run is not None:
        wandb_run.summary["mean_reward"] = average_reward
        wandb_run.summary["success_rate"] = success_rate
        wandb_run.finish()
    return average_reward


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
    eval_seed = int(getattr(ppo_cfg, "eval_seed", 0))
    reward_mode = str(getattr(ppo_cfg, "test_reward_mode", "wm")).lower()
    if reward_mode not in {"wm", "native"}:
        raise ValueError("PPO.test_reward_mode must be 'wm' or 'native'")
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    render_delay = float(getattr(ppo_cfg, "render_delay", 0.05))
    gif_fps = int(getattr(ppo_cfg, "gif_fps", 10))
    gif_max_frames = max(2, int(getattr(ppo_cfg, "gif_max_frames", 300)))
    gif_downsample = max(1, int(getattr(ppo_cfg, "gif_downsample", 2)))
    total_episodes = int(ppo_cfg.total_test_episodes)
    test_num_envs = max(1, int(getattr(ppo_cfg, "test_num_envs", 1)))
    max_ep_len = int(ppo_cfg.max_ep_len)
    use_main_dense_reward = bool(
        getattr(ppo_cfg, "use_main_dense_reward", False)
    )
    main_reward = main_dense_reward_settings(ppo_cfg)
    wandb_run = None
    if bool(getattr(ppo_cfg, "use_wandb", False)):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "PPO.use_wandb=True requires wandb; install it or disable W&B"
            ) from exc
        wandb_kwargs = {
            "project": str(getattr(ppo_cfg, "wandb_project", "minigrid-ppo")),
            "job_type": "policy-evaluation",
        }
        entity = getattr(ppo_cfg, "wandb_entity", None)
        if entity not in (None, "", "auto"):
            wandb_kwargs["entity"] = str(entity)
        wandb_run = wandb.init(**wandb_kwargs)

    # Human mode opens a live window. RGB mode is used for headless/GIF runs.
    render_mode = "human" if render else "rgb_array"
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=str(ppo_cfg.env_path),
            custom_mission="Reach the goal.",
            max_steps=max_ep_len,
            render_mode=render_mode,
            **stochastic_env_kwargs(cfg),
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
        minibatch_size=int(getattr(ppo_cfg, "minibatch_size", 0)),
    )
    print(f"Loading policy: {checkpoint_path}")
    print(f"Training source: {policy_training_source(cfg)}")
    print(f"Real layout:   {ppo_cfg.env_path}")
    print(f"Evaluation:    {'deterministic' if deterministic else 'stochastic'}")
    print(f"Reward:        {'main_dense' if use_main_dense_reward else 'native'}")
    print(f"Test envs:     {test_num_envs}")
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
        return _parallel_validate_policy(
            cfg=cfg,
            ppo_cfg=ppo_cfg,
            ppo_agent=ppo_agent,
            total_episodes=total_episodes,
            test_num_envs=test_num_envs,
            max_ep_len=max_ep_len,
            eval_seed=eval_seed,
            deterministic=deterministic,
            reward_mode=reward_mode,
            use_main_dense_reward=use_main_dense_reward,
            main_reward=main_reward,
            save_csv=save_csv,
            wandb_run=wandb_run,
        )

    gif_dir = Path(str(ppo_cfg.save_path_gif)).expanduser().resolve()
    csv_dir = Path(str(ppo_cfg.save_path_csv)).expanduser().resolve()
    if save_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_reward = 0.0

    def capture_frame():
        frame = env.render()
        if gif_downsample > 1:
            frame = frame[::gif_downsample, ::gif_downsample]
        return frame

    for episode in range(1, total_episodes + 1):
        observation, _ = env.reset(seed=eval_seed + episode - 1)
        state = ColRowCanl_to_CanlRowCol(observation["image"])
        episode_reward = 0.0
        native_episode_reward = 0.0
        shaped_episode_reward = 0.0
        goal_distance_sum = 0.0
        goal_distance_count = 0
        goal_distance_min = float("inf")
        goal_region_entry_count = 0
        progress_milestone_count = 0
        progress_milestone_reward_sum = 0.0
        critical_door_crossing_count = 0
        critical_door_crossing_reward_sum = 0.0
        episode_succeeded = False
        goals = np.argwhere(state[0] == 8)
        if len(goals) != 1:
            raise ValueError("The real MiniGrid layout must contain exactly one goal")
        goal_yx = tuple(map(int, goals[0]))
        initial_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        goal_distance_map = build_goal_distance_map(initial_state, goal_yx)
        goal_region_mask = build_goal_region_mask(initial_state, goal_yx)
        door_topology = build_door_topology(initial_state, goal_yx)
        goal_region_seen = torch.zeros(1, dtype=torch.bool, device=device)
        rewarded_goal_region_key_positions = torch.zeros(
            (1, *initial_state.shape[-2:]), dtype=torch.bool, device=device
        )
        rewarded_goal_region_door_positions = torch.zeros_like(
            rewarded_goal_region_key_positions
        )
        progress_milestone_seen = torch.zeros(1, dtype=torch.bool, device=device)
        pending_door_ids = torch.full((1,), -1, dtype=torch.long, device=device)
        pending_door_origins = torch.full_like(pending_door_ids, -1)
        rewarded_door_crossings = torch.zeros(
            (1, door_topology.num_doors), dtype=torch.bool, device=device
        )
        initial_position = (initial_state[0] == 10).nonzero(as_tuple=False)[0]
        initial_goal_distance = goal_distance_map[
            initial_position[0], initial_position[1]
        ].reshape(1)
        best_goal_distance = initial_goal_distance.clone()
        termination_reason = "max_steps"
        seen_states = set()
        if deterministic:
            initial_inventory = carrying_token_from_env(env)
            seen_states.add(
                state.tobytes()
                + np.asarray([initial_inventory], dtype=np.int16).tobytes()
            )
        frames = []
        if save_gif and not render and episode == 1:
            frames.append(capture_frame())

        for step in range(1, max_ep_len + 1):
            previous_state = state
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
            previous_inventory_token = carrying_token_from_env(env)
            observation, native_reward, terminated, truncated, _ = env.step(native_action)
            state = ColRowCanl_to_CanlRowCol(observation["image"])
            step_succeeded = bool(terminated and native_reward > 0.0)
            lava_terminated = bool(terminated and not step_succeeded)
            episode_succeeded |= step_succeeded
            native_episode_reward += float(native_reward)
            if use_main_dense_reward:
                dense_reward, goal_region_seen, reward_events = main_dense_rewards(
                    torch.as_tensor(previous_state, device=device).unsqueeze(0),
                    torch.as_tensor(state, device=device).unsqueeze(0),
                    torch.as_tensor([compact_action], device=device),
                    goal_distance_map,
                    goal_region_mask,
                    goal_region_seen,
                    torch.as_tensor([step_succeeded], device=device),
                    torch.as_tensor([lava_terminated], device=device),
                    best_goal_distances=best_goal_distance,
                    initial_goal_distances=initial_goal_distance,
                    progress_milestone_seen=progress_milestone_seen,
                    previous_carrying_tokens=torch.as_tensor(
                        [previous_inventory_token], device=device
                    ),
                    current_carrying_tokens=torch.as_tensor(
                        [carrying_token_from_env(env)], device=device
                    ),
                    door_topology=door_topology,
                    pending_door_ids=pending_door_ids,
                    pending_door_origins=pending_door_origins,
                    rewarded_door_crossings=rewarded_door_crossings,
                    rewarded_goal_region_key_positions=rewarded_goal_region_key_positions,
                    rewarded_goal_region_door_positions=rewarded_goal_region_door_positions,
                    **main_reward,
                )
                best_goal_distance = reward_events["new_best_goal_distance"]
                progress_milestone_seen = reward_events["progress_milestone_seen"]
                pending_door_ids = reward_events["pending_door_ids"]
                pending_door_origins = reward_events["pending_door_origins"]
                rewarded_door_crossings = reward_events["rewarded_door_crossings"]
                rewarded_goal_region_key_positions = reward_events[
                    "rewarded_goal_region_key_positions"
                ]
                rewarded_goal_region_door_positions = reward_events[
                    "rewarded_goal_region_door_positions"
                ]
                reward = float(dense_reward.item())
                current_distance = float(reward_events["current_distance"].item())
                if np.isfinite(current_distance):
                    goal_distance_sum += current_distance
                    goal_distance_count += 1
                    goal_distance_min = min(goal_distance_min, current_distance)
                goal_region_entry_count += int(
                    reward_events["goal_region_entered"].item()
                )
                progress_milestone_count += int(
                    reward_events["progress_milestone_crossed"].item()
                )
                progress_milestone_reward_sum += float(
                    reward_events["progress_milestone_reward"].item()
                )
                critical_door_crossing_count += int(
                    reward_events["critical_door_crossed"].item()
                )
                critical_door_crossing_reward_sum += float(
                    reward_events["critical_door_crossing_reward"].item()
                )
            else:
                reward = float(native_reward)
            shaped_episode_reward += reward
            episode_reward += (
                reward if reward_mode == "wm" else float(native_reward)
            )

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
                if len(frames) < gif_max_frames:
                    frames.append(capture_frame())

            if terminated:
                termination_reason = (
                    "goal" if step_succeeded else "terminated"
                )
                break
            if truncated:
                termination_reason = "truncated"
                break
            if deterministic:
                inventory_token = carrying_token_from_env(env)
                state_key = (
                    state.tobytes()
                    + np.asarray([inventory_token], dtype=np.int16).tobytes()
                )
                if state_key in seen_states:
                    termination_reason = "cycle_detected"
                    break
                seen_states.add(state_key)

        if render:
            print()
        success = episode_succeeded
        total_reward += episode_reward
        results.append(
            (
                episode, step, episode_reward, native_episode_reward,
                success, termination_reason,
                goal_distance_sum / max(goal_distance_count, 1),
                goal_distance_min if np.isfinite(goal_distance_min) else None,
                goal_region_entry_count,
                progress_milestone_count,
                progress_milestone_reward_sum,
                critical_door_crossing_count,
                critical_door_crossing_reward_sum,
                shaped_episode_reward,
            )
        )
        print(
            f"Episode {episode}: steps={step}, reward={episode_reward:.5f}, "
            f"native_reward={native_episode_reward:.5f}, "
            f"wm_shaped_reward={shaped_episode_reward:.5f}, "
            f"success={success}, goal_region_entries={goal_region_entry_count}, "
            f"progress_milestones={progress_milestone_count}, "
            f"critical_door_crossings={critical_door_crossing_count}, "
            f"reason={termination_reason}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "episode": episode,
                    "steps": step,
                    "reward": episode_reward,
                    "native_reward": native_episode_reward,
                    "wm_shaped_reward": shaped_episode_reward,
                    "success": success,
                    "goal_dist_mean": goal_distance_sum
                    / max(goal_distance_count, 1),
                    "goal_dist_min": goal_distance_min
                    if np.isfinite(goal_distance_min)
                    else None,
                    "goal_region_entry_count": goal_region_entry_count,
                    "progress_milestone_count": progress_milestone_count,
                    "progress_milestone_reward_sum": progress_milestone_reward_sum,
                    "critical_door_crossing_count": critical_door_crossing_count,
                    "critical_door_crossing_reward_sum": critical_door_crossing_reward_sum,
                },
                step=episode,
            )

        if save_gif and not render and episode == 1 and frames:
            # For a long failed episode, preserve its final state without
            # retaining thousands of full-resolution frames in memory.
            if step + 1 > gif_max_frames:
                frames[-1] = capture_frame()
            gif_path = gif_dir / "ppo_real_env_test.gif"
            imageio.mimsave(gif_path, frames, fps=gif_fps)
            print(f"Saved GIF: {gif_path}")

    env.close()

    if save_csv:
        csv_path = csv_dir / "ppo_real_env_test.csv"
        evaluation_results = pd.DataFrame(
            results,
            columns=[
                "episode", "steps", "reward", "native_reward", "success", "reason",
                "goal_dist_mean", "goal_dist_min", "goal_region_entry_count",
                "progress_milestone_count", "progress_milestone_reward_sum",
                "critical_door_crossing_count", "critical_door_crossing_reward_sum",
                "wm_shaped_reward",
            ],
        )
        evaluation_results = append_mean_row(
            evaluation_results,
            mean_columns=[
                "steps", "reward", "native_reward", "success",
                "goal_dist_mean", "goal_dist_min", "goal_region_entry_count",
                "progress_milestone_count", "progress_milestone_reward_sum",
                "critical_door_crossing_count", "critical_door_crossing_reward_sum",
                "wm_shaped_reward",
            ],
            labels={"episode": "mean"},
        )
        evaluation_results.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    average_reward = total_reward / max(total_episodes, 1)
    success_rate = sum(row[4] for row in results) / max(total_episodes, 1)
    print(f"Average real-environment reward: {average_reward:.5f}")
    print(f"Success rate: {success_rate:.1%}")
    if wandb_run is not None:
        wandb_run.summary["mean_reward"] = average_reward
        wandb_run.summary["success_rate"] = success_rate
        wandb_run.finish()
    return average_reward


if __name__ == "__main__":
    test()
