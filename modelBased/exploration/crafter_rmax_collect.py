"""Train a real-environment count explorer and collect one Crafter WM dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from domain.crafter.crafter_support import CrafterCoverageTracker
from modelBased.common.artifacts import single_environment_dataset_path
from modelBased.data.data_collect import save_experiments
from modelBased.exploration.count_based import (
    CrafterStateActionCounter,
    build_crafter_policy_state,
)
from modelBased.policy_training.ppo.PPO import PPO
from modelBased.policy_training.common.crafter_vector_env import CrafterSubprocessVectorEnv


def apply_death_penalty(
    intrinsic_rewards: np.ndarray,
    terminated: np.ndarray,
    death_penalty: float,
) -> np.ndarray:
    """Replace true-death rewards with a negative penalty.

    ``terminated`` deliberately excludes time-limit truncation.  The argument
    is a non-negative magnitude so ``death_penalty=1`` produces reward ``-1``.
    """
    if death_penalty < 0:
        raise ValueError("rmax_like.death_penalty must be non-negative")
    rewards = np.asarray(intrinsic_rewards, dtype=np.float32).copy()
    deaths = np.asarray(terminated, dtype=np.bool_)
    if rewards.shape != deaths.shape:
        raise ValueError("intrinsic_rewards and terminated must have equal shapes")
    rewards[deaths] = -float(death_penalty)
    return rewards


def _validate_config(cfg: DictConfig) -> tuple[int, int, int]:
    if str(cfg.domain) != "crafter":
        raise ValueError("RMax-like acquisition currently supports Crafter only")
    rmax_cfg = cfg.rmax_like
    if not bool(getattr(rmax_cfg, "enabled", False)):
        raise ValueError("RMax-like collector requires rmax_like.enabled=true")
    if bool(getattr(cfg.p2e, "enabled", False)):
        raise ValueError("rmax_like.enabled and p2e.enabled are mutually exclusive")

    train_steps = int(rmax_cfg.train_steps)
    frozen_steps = int(rmax_cfg.frozen_steps)
    num_envs = int(rmax_cfg.num_envs)
    total_steps = int(cfg.domains.crafter.data_collection.maximum_dataset_size)
    if min(train_steps, frozen_steps, num_envs) <= 0:
        raise ValueError("train_steps, frozen_steps and num_envs must be positive")
    if train_steps + frozen_steps != total_steps:
        raise ValueError(
            "Crafter maximum_dataset_size must equal "
            "rmax_like.train_steps + rmax_like.frozen_steps"
        )
    if train_steps % num_envs or frozen_steps % num_envs:
        raise ValueError("Both RMax-like stage budgets must be divisible by num_envs")
    rollout_steps = int(rmax_cfg.rollout_steps)
    if int(rmax_cfg.mask_size) != 5:
        raise ValueError(
            "rmax_count_local5_inv12_v1 requires rmax_like.mask_size=5"
        )
    if rollout_steps < num_envs or rollout_steps % num_envs:
        raise ValueError("rmax_like.rollout_steps must be divisible by num_envs")
    if float(getattr(rmax_cfg, "death_penalty", 0.0)) < 0:
        raise ValueError("rmax_like.death_penalty must be non-negative")
    return train_steps, frozen_steps, num_envs


def _make_ppo(cfg: DictConfig, state_dim: int) -> PPO:
    rmax_cfg = cfg.rmax_like
    print(
        "[Crafter RMax-like] PPO config "
        f"rollout={int(rmax_cfg.rollout_steps)} "
        f"gamma={float(rmax_cfg.gamma):.4f} "
        f"K_epochs={int(rmax_cfg.K_epochs)} "
        f"lr_actor={float(rmax_cfg.lr_actor):.1e} "
        f"lr_critic={float(rmax_cfg.lr_critic):.1e} "
        f"normalize_returns={bool(rmax_cfg.normalize_returns)}"
    )
    return PPO(
        state_dim=state_dim,
        action_dim=int(cfg.attention_model.action_norm_values),
        lr_actor=float(rmax_cfg.lr_actor),
        lr_critic=float(rmax_cfg.lr_critic),
        gamma=float(rmax_cfg.gamma),
        K_epochs=int(rmax_cfg.K_epochs),
        eps_clip=float(rmax_cfg.eps_clip),
        has_continuous_action_space=False,
        action_std_init=float(getattr(rmax_cfg, "action_std", 0.1)),
        entropy_coef=float(rmax_cfg.entropy_coef),
        normalize_advantages=bool(rmax_cfg.normalize_advantages),
        normalize_returns=bool(rmax_cfg.normalize_returns),
        max_grad_norm=float(rmax_cfg.max_grad_norm),
        minibatch_size=int(getattr(rmax_cfg, "minibatch_size", 0)),
    )


def _parameter_snapshot(ppo: PPO) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in ppo.policy_old.parameters()]


def _parameters_equal(snapshot: list[torch.Tensor], ppo: PPO) -> bool:
    current = [parameter.detach().cpu() for parameter in ppo.policy_old.parameters()]
    return len(snapshot) == len(current) and all(
        torch.equal(before, after) for before, after in zip(snapshot, current)
    )


def collect_crafter_rmax(
    cfg: DictConfig,
    *,
    envs=None,
    ppo_factory=None,
) -> Path:
    """Run train/frozen acquisition and return the saved dataset path.

    ``envs`` and ``ppo_factory`` are injectable only to keep the small-budget
    integration test independent of multiprocessing.
    """
    train_steps, frozen_steps, num_envs = _validate_config(cfg)
    rmax_cfg = cfg.rmax_like
    dataset_path = single_environment_dataset_path(cfg, "crafter")
    # The mode-specific domain path is authoritative for direct module runs;
    # run_pipeline resolves and forwards the same path explicitly.
    with open_dict(cfg):
        cfg.env.collect.data_save_path = str(dataset_path)
    seed = int(rmax_cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    owns_envs = envs is None
    if envs is None:
        envs = CrafterSubprocessVectorEnv(
            layout_path=str(cfg.domains.crafter.layout_path),
            max_steps=int(cfg.domains.crafter.data_collection.max_steps),
            num_envs=num_envs,
            seed=seed,
            start_method=str(getattr(rmax_cfg, "start_method", "spawn")),
        )

    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "obs", "obs_next", "act", "rew", "done",
            "info", "inv", "inv_next", "phase",
        )
    }
    coverage = CrafterCoverageTracker()
    counter = CrafterStateActionCounter(
        mask_size=int(rmax_cfg.mask_size),
        reward_scale=float(rmax_cfg.reward_scale),
    )
    death_penalty = float(getattr(rmax_cfg, "death_penalty", 0.0))
    try:
        obs, _ = envs.reset()
        initial_state = build_crafter_policy_state(
            obs["image"], obs["inventory"], cfg.attention_model.obs_norm_values
        )
        state_dim = int(initial_state.shape[1])
        ppo = ppo_factory(cfg, state_dim) if ppo_factory else _make_ppo(cfg, state_dim)
        rollout_steps = int(rmax_cfg.rollout_steps)
        total_collected = 0

        def collect_stage(stage_steps: int, *, training: bool, phase_id: int):
            nonlocal obs, total_collected
            stage_rewards: list[float] = []
            stage_deaths = 0
            last_terminated = np.zeros(num_envs, dtype=np.bool_)
            for _ in range(stage_steps // num_envs):
                coverage.update(obs["image"], obs["inventory"])
                state = build_crafter_policy_state(
                    obs["image"], obs["inventory"], cfg.attention_model.obs_norm_values
                )
                actions, states_buf, actions_buf, logprobs_buf, values_buf = \
                    ppo.select_action_batch(state)
                action_np = actions.detach().cpu().numpy().reshape(-1)
                next_obs, native_rewards, terminated, truncated, infos = envs.step(
                    action_np
                )
                dones = np.asarray(terminated) | np.asarray(truncated)
                count_rewards = counter.rewards(
                    obs["image"], obs["inventory"], action_np,
                    terminated=terminated, update=True,
                )
                count_rewards = apply_death_penalty(
                    count_rewards, terminated, death_penalty
                )

                arrays["obs"].append(np.asarray(obs["image"]).copy())
                arrays["obs_next"].append(np.asarray(next_obs["image"]).copy())
                # Match the canonical run_env NPZ contract: scalar transition
                # fields are (N,), while maps and inventories retain features.
                arrays["act"].append(action_np.astype(np.int64))
                arrays["rew"].append(np.asarray(native_rewards, dtype=np.float32))
                arrays["done"].append(np.asarray(terminated, dtype=np.bool_))
                arrays["info"].append(np.asarray(infos, dtype=object))
                arrays["inv"].append(
                    np.asarray(obs["inventory"], dtype=np.float32).copy()
                )
                arrays["inv_next"].append(
                    np.asarray(next_obs["inventory"], dtype=np.float32).copy()
                )
                arrays["phase"].append(np.full(num_envs, phase_id, dtype=np.int8))

                if training:
                    ppo.save_buffer_batch(
                        states_buf, actions_buf, logprobs_buf, values_buf,
                        count_rewards, dones,
                    )
                stage_rewards.extend(count_rewards.tolist())
                stage_deaths += int(np.count_nonzero(terminated))
                total_collected += num_envs
                last_terminated = np.asarray(terminated, dtype=np.bool_)

                if training and total_collected % rollout_steps == 0:
                    next_state = build_crafter_policy_state(
                        next_obs["image"], next_obs["inventory"],
                        cfg.attention_model.obs_norm_values,
                    )
                    bootstrap = ppo.estimate_old_values_batch(next_state)
                    bootstrap[torch.as_tensor(last_terminated)] = 0.0
                    ppo.update(bootstrap_value=bootstrap)

                done_indices = np.flatnonzero(dones)
                obs = {
                    "image": np.asarray(next_obs["image"]).copy(),
                    "inventory": np.asarray(next_obs["inventory"]).copy(),
                }
                if done_indices.size:
                    reset_obs, _ = envs.reset_at(done_indices)
                    obs["image"][done_indices] = reset_obs["image"]
                    obs["inventory"][done_indices] = reset_obs["inventory"]

            mean_reward = float(np.mean(stage_rewards)) if stage_rewards else 0.0
            print(
                f"[Crafter RMax-like] phase={'train' if training else 'frozen'} "
                f"steps={stage_steps} mean_intrinsic_reward={mean_reward:.6f} "
                f"deaths={stage_deaths} death_penalty={death_penalty:.3f} "
                f"unique_sa={counter.unique_state_actions}"
            )
            return last_terminated

        last_terminated = collect_stage(train_steps, training=True, phase_id=0)
        if ppo.buffer.transition_count() > 0:
            final_state = build_crafter_policy_state(
                obs["image"], obs["inventory"], cfg.attention_model.obs_norm_values
            )
            bootstrap = ppo.estimate_old_values_batch(final_state)
            bootstrap[torch.as_tensor(last_terminated)] = 0.0
            ppo.update(bootstrap_value=bootstrap)

        checkpoint = Path(str(rmax_cfg.checkpoint_path)).expanduser().resolve()
        ppo.save(checkpoint)
        frozen_snapshot = _parameter_snapshot(ppo)
        collect_stage(frozen_steps, training=False, phase_id=1)
        if not _parameters_equal(frozen_snapshot, ppo):
            raise RuntimeError("RMax-like PPO parameters changed during frozen collection")

        merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
        expected = train_steps + frozen_steps
        if any(len(value) != expected for value in merged.values()):
            lengths = {key: len(value) for key, value in merged.items()}
            raise RuntimeError(f"RMax-like dataset arrays are misaligned: {lengths}")
        permutation = np.random.default_rng(seed).permutation(expected)
        merged = {key: value[permutation] for key, value in merged.items()}

        save_experiments(
            cfg,
            merged["obs"], merged["obs_next"], merged["act"], merged["rew"],
            merged["done"], merged["info"], merged["inv"], merged["inv_next"],
            extra_arrays={"acquisition_phase": merged["phase"]},
        )
        coverage_path = getattr(rmax_cfg, "coverage_path", None)
        if coverage_path and str(coverage_path).lower() != "null":
            coverage.save(
                Path(str(coverage_path)).expanduser().resolve(),
                title=(
                    f"Crafter RMax-like coverage ({cfg.domains.crafter.task_name}, "
                    f"steps={expected})"
                ),
            )
        print(
            f"[Crafter RMax-like] saved dataset={cfg.env.collect.data_save_path} "
            f"checkpoint={checkpoint} transitions={expected} "
            f"unique_sa={counter.unique_state_actions}"
        )
        return dataset_path
    finally:
        if owns_envs:
            envs.close()


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "modelBased/config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    collect_crafter_rmax(cfg)


if __name__ == "__main__":
    main()
