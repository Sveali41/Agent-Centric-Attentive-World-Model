"""Hydra entry point for the data -> world model -> policy pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from modelBased.common.artifacts import dataset_matches
from modelBased.policy_training.common.experiment_naming import (
    policy_checkpoint_is_compatible,
    policy_checkpoint_path,
)
from modelBased.common.artifacts import (
    p2e_enabled,
    rmax_like_enabled,
    single_environment_dataset_path,
    validate_acquisition_flags,
    world_model_checkpoint_path,
)
from domain.crafter.crafter_support import inspect_crafter_planning_checkpoint

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modelBased" / "models"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT.parent / ".env", override=False)
except ImportError:
    pass

# Allow the pipeline to start even when it is launched without sourcing the
# repository-level .env file first.
os.environ.setdefault("PROJECT_ROOT", str(ROOT.parent))
os.environ.setdefault("WM_ROOT", str(ROOT))
os.environ.setdefault("ENV_PATH", str(ROOT.parent / "level"))
os.environ.setdefault("WORLD_MODEL_PATH", str(ROOT / "modelBased"))
os.environ.setdefault(
    "TRAIN_DATASET_PATH", str(ROOT / "modelBased" / "data" / "train_world_model")
)
os.environ.setdefault("MODEL_FPATH", str(ROOT / "modelBased" / "models"))
os.environ.setdefault("GENERATOR_PATH", str(ROOT.parent / "generator"))
os.environ.setdefault("TRAINER_PATH", str(ROOT.parent / "trainer"))




def run_command(args: list[str], label: str, cfg: DictConfig) -> None:
    print(f"\n{'=' * 72}\n[{label}] {' '.join(args)}\n{'=' * 72}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True, env=os.environ.copy())


def dataset_transition_count(path: Path) -> int | None:
    """Return the saved transition count without trusting metadata alone."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return int(len(data["a"]))
    except (OSError, KeyError, ValueError):
        return None


def policy_runtime_overrides(cfg: DictConfig) -> list[str]:
    """Forward policy train/test settings to Hydra subprocesses.

    The pipeline launches each stage in a fresh process, so command-line
    overrides on the pipeline itself are not otherwise visible to PPO.
    """
    ppo_cfg = cfg.PPO
    overrides = []
    for name in (
        "wm_control_mode",
        "max_ep_len",
        "rollout_steps",
        "max_training_timesteps",
        "save_model_freq",
        "total_test_episodes",
        "render_delay",
        "gif_fps",
    ):
        if hasattr(ppo_cfg, name):
            overrides.append(f"PPO.{name}={getattr(ppo_cfg, name)}")
    for name in (
        "use_wandb",
        "render",
        "test_deterministic",
        "save_csv",
        "save_gif",
    ):
        if hasattr(ppo_cfg, name):
            overrides.append(
                f"PPO.{name}={str(bool(getattr(ppo_cfg, name))).lower()}"
            )
    experiment_label = getattr(ppo_cfg, "experiment_label", None)
    if (
        experiment_label is not None
        and str(experiment_label).strip().lower() not in {"", "null", "none"}
    ):
        overrides.append(f"PPO.experiment_label={experiment_label}")
    return overrides


def domain_runtime_overrides(cfg: DictConfig, domain: str) -> list[str]:
    """Forward domain task and collection budgets to stage subprocesses."""
    domain_cfg = cfg.domains[domain]
    overrides = [
        f"domains.{domain}.task_name={domain_cfg.task_name}",
        # task_name alone does not rewrite a literal layout_path. Forward the
        # fully resolved layout so collection, WM training and PPO all use the
        # same environment file in their fresh Hydra subprocesses.
        f"domains.{domain}.layout_path={domain_cfg.layout_path}",
    ]
    if domain == "minigrid" and hasattr(domain_cfg, "minigrid_transition_mode"):
        overrides.append(
            f"domains.{domain}.minigrid_transition_mode="
            f"{domain_cfg.minigrid_transition_mode}"
        )
    if domain == "minigrid":
        stochastic_cfg = getattr(domain_cfg, "stochastic", None)
        if stochastic_cfg is not None:
            overrides.extend(
                [
                    "domains.minigrid.stochastic.enabled="
                    f"{str(bool(getattr(stochastic_cfg, 'enabled', False))).lower()}",
                    "domains.minigrid.stochastic.move_failure_prob="
                    f"{float(getattr(stochastic_cfg, 'move_failure_prob', 0.2))}",
                ]
            )
    collection_cfg = getattr(domain_cfg, "data_collection", None)
    if collection_cfg is not None:
        for name in (
            "max_steps",
            "data_type",
            "episodes",
            "mini_dataset_size",
            "maximum_dataset_size",
            "uniform_reset_steps",
        ):
            if hasattr(collection_cfg, name):
                overrides.append(
                    f"domains.{domain}.data_collection.{name}="
                    f"{getattr(collection_cfg, name)}"
                )
    attention_cfg = getattr(cfg, "attention_model", None)
    if attention_cfg is not None and hasattr(attention_cfg, "n_cpu"):
        overrides.append(f"attention_model.n_cpu={int(attention_cfg.n_cpu)}")
    if hasattr(domain_cfg, "collection_replace_start_with_empty"):
        value = bool(domain_cfg.collection_replace_start_with_empty)
        overrides.append(
            "domains."
            f"{domain}.collection_replace_start_with_empty="
            f"{str(value).lower()}"
        )
    return overrides


def rmax_like_runtime_overrides(cfg: DictConfig, domain: str) -> list[str]:
    """Forward count-acquisition settings to pipeline subprocesses."""
    if not hasattr(cfg, "rmax_like"):
        return []
    rmax_cfg = cfg.rmax_like
    overrides = [
        f"rmax_like.enabled={str(bool(getattr(rmax_cfg, 'enabled', False))).lower()}"
    ]
    if domain != "crafter" or not bool(getattr(rmax_cfg, "enabled", False)):
        return overrides
    for name in (
        "train_steps", "frozen_steps", "num_envs", "mask_size", "reward_scale",
        "death_penalty", "seed", "rollout_steps", "start_method", "checkpoint_path", "coverage_path",
        "lr_actor", "lr_critic", "gamma", "K_epochs", "eps_clip", "action_std",
        "entropy_coef", "normalize_advantages", "normalize_returns", "max_grad_norm",
    ):
        if hasattr(rmax_cfg, name):
            value = getattr(rmax_cfg, name)
            if isinstance(value, bool):
                value = str(value).lower()
            overrides.append(f"rmax_like.{name}={value}")
    return overrides


def p2e_runtime_overrides(cfg: DictConfig) -> list[str]:
    """Forward the online Dreamer P2E budget/schedule to its subprocess."""
    if not hasattr(cfg, "p2e"):
        return []
    names = (
        "enabled", "total_steps", "prefill_steps", "train_every", "train_steps",
        "replay_capacity", "replay_batch_size", "replay_sequence_length",
        "checkpoint_every", "checkpoint_path", "resume",
    )
    return [f"p2e.{name}={getattr(cfg.p2e, name)}" for name in names if hasattr(cfg.p2e, name)]


def run_domain(domain: str, cfg: DictConfig) -> None:
    domain_cfg = cfg.domains[domain]
    force = bool(cfg.pipeline.force)
    force_policy = bool(getattr(cfg.pipeline, "force_policy", False))
    skip_world_model = bool(getattr(cfg.pipeline, "skip_world_model", False))
    python = sys.executable
    layout_path = Path(str(domain_cfg.layout_path)).expanduser().resolve()
    validate_acquisition_flags(cfg, domain)
    if not layout_path.exists():
        raise FileNotFoundError(
            f"[{domain}] Canonical layout does not exist: {layout_path}"
        )
    print(f"[{domain}] Canonical layout: {layout_path}", flush=True)

    train_in_real_env = getattr(cfg.PPO, "train_in_real_env", False)
    policy_source_override = (
        f"PPO.train_in_real_env={str(bool(train_in_real_env)).lower()}"
    )
    policy_identity_overrides = [
        policy_source_override,
        f"PPO.seed={int(cfg.PPO.seed)}",
        f"PPO.action_set={cfg.PPO.action_set}",
        f"domains.{domain}.task_name={domain_cfg.task_name}",
        f"domains.{domain}.layout_path={domain_cfg.layout_path}",
    ]
    if domain == "minigrid" and hasattr(domain_cfg, "minigrid_transition_mode"):
        policy_identity_overrides.append(
            f"domains.{domain}.minigrid_transition_mode="
            f"{domain_cfg.minigrid_transition_mode}"
        )
    if domain == "minigrid":
        stochastic_cfg = getattr(domain_cfg, "stochastic", None)
        if stochastic_cfg is not None:
            policy_identity_overrides.extend(
                [
                    "domains.minigrid.stochastic.enabled="
                    f"{str(bool(getattr(stochastic_cfg, 'enabled', False))).lower()}",
                    "domains.minigrid.stochastic.move_failure_prob="
                    f"{float(getattr(stochastic_cfg, 'move_failure_prob', 0.2))}",
                ]
            )
    if hasattr(cfg, "p2e"):
        policy_identity_overrides.append(
            f"p2e.enabled={str(bool(cfg.p2e.enabled)).lower()}"
        )
    if hasattr(cfg, "rmax_like"):
        policy_identity_overrides.append(
            f"rmax_like.enabled={str(bool(cfg.rmax_like.enabled)).lower()}"
        )
    configured_continual = getattr(domain_cfg, "continual_learning", None)
    if configured_continual is not None:
        policy_identity_overrides.append(
            f"domains.{domain}.continual_learning.enabled="
            f"{str(bool(getattr(configured_continual, 'enabled', False))).lower()}"
        )
    domain_overrides = domain_runtime_overrides(cfg, domain)
    policy_overrides = policy_identity_overrides + policy_runtime_overrides(cfg)
    continual_cfg = configured_continual
    continual_enabled = bool(
        continual_cfg is not None and getattr(continual_cfg, "enabled", False)
    )
    world_model_path = world_model_checkpoint_path(cfg, domain)
    configured_policy_wm = getattr(cfg.PPO, "checkpoint_path_wm", None)
    policy_world_model_path = (
        Path(str(configured_policy_wm)).expanduser().resolve()
        if configured_policy_wm is not None
        and str(configured_policy_wm).strip()
        and str(configured_policy_wm).lower() != "null"
        else world_model_path
    )
    if continual_enabled and train_in_real_env:
        raise ValueError(
            f"[{domain}] continual WM training cannot be combined with "
            "PPO.train_in_real_env=true. Disable continual_learning or use WM planning."
        )
    if train_in_real_env and rmax_like_enabled(cfg, domain):
        raise ValueError(
            "RMax-like WM acquisition requires PPO.train_in_real_env=false; "
            "the acquisition explorer is trained separately in the real environment"
        )

    if skip_world_model:
        if not policy_world_model_path.is_file():
            raise FileNotFoundError(
                f"[{domain}] pipeline.skip_world_model=true but the configured "
                f"WM checkpoint does not exist: {policy_world_model_path}"
            )
        print(
            f"[{domain}] Reusing existing world-model checkpoint and skipping "
            f"all data/WM stages: {policy_world_model_path}"
        )
        data_recollected = False
    elif continual_enabled:
        print(
            f"[{domain}] Continual WM training enabled. "
            "Using the configured phase datasets and skipping single-environment collection."
        )
        final_checkpoint = world_model_path
        reuse_final = bool(
            getattr(continual_cfg, "skip_if_final_checkpoint_exists", True)
        )
        if reuse_final and final_checkpoint.is_file() and not force:
            print(
                f"[{domain}] Final continual WM checkpoint already exists; "
                f"skipping curriculum collection and WM training: {final_checkpoint}"
            )
            data_recollected = False
            continual_enabled = False
        else:
            world_model_path_before = world_model_path
            world_model_mtime_before = (
                world_model_path_before.stat().st_mtime_ns
                if world_model_path_before.exists()
                else None
            )
            continual_overrides = [
                f"domains.{domain}.continual_learning.enabled=true",
            ]
            if hasattr(cfg, "p2e"):
                continual_overrides.append(
                    f"p2e.enabled={str(bool(cfg.p2e.enabled)).lower()}"
                )
            if hasattr(cfg, "rmax_like"):
                continual_overrides.append(
                    f"rmax_like.enabled={str(bool(cfg.rmax_like.enabled)).lower()}"
                )
            if force:
                # pipeline.force means start a fresh continual run. The state and
                # replay files are still replaced atomically by the trainer.
                continual_overrides.append(
                    f"domains.{domain}.continual_learning.resume=false"
                )
                continual_overrides.append(
                    f"domains.{domain}.continual_learning.force_collection=true"
                )
            if bool(getattr(continual_cfg, "collect_data", True)):
                run_command(
                    [
                        python,
                        "-m",
                        (
                            "modelBased.exploration.crafter_rmax_continual_collect"
                            if rmax_like_enabled(cfg, domain)
                            else "modelBased.continue_learning.crafter_curriculum_collect"
                        ),
                        f"domain={domain}",
                        *continual_overrides,
                    ],
                    f"{domain} / collect continual curriculum data",
                    cfg,
                )
            run_command(
                [
                    python,
                    "-m",
                    "modelBased.continue_learning.continual_wm_training",
                    f"domain={domain}",
                    *continual_overrides,
                ],
                f"{domain} / continual world-model training",
                cfg,
            )
            world_model_mtime_after = (
                world_model_path_before.stat().st_mtime_ns
                if world_model_path_before.exists()
                else None
            )
            data_recollected = (
                world_model_mtime_after is not None
                and world_model_mtime_after != world_model_mtime_before
            )
    elif train_in_real_env:
        print(f"[{domain}] PPO.train_in_real_env is True. Skipping data collection and world-model training.")
        data_recollected = False
    else:
        # --- 1. Dataset Collection ---
        use_single_p2e = p2e_enabled(cfg, domain)
        use_rmax_like = rmax_like_enabled(cfg, domain)
        data_path = single_environment_dataset_path(cfg, domain)
        data_overrides = [
            *domain_overrides,
            f"domains.{domain}.data_save_path={data_path}",
            f"env.collect.data_save_path={data_path}",
            f"attention_model.data_dir={data_path}",
        ]
        # The mode-specific path is authoritative inside the acquisition
        # subprocess too. Without forwarding it, a child composed with its
        # default PPO.seed can silently write every RMax/P2E run to seed 0
        # while the parent trains the WM from a different seed path.
        if use_single_p2e:
            data_overrides.append(
                f"domains.{domain}.p2e_data_save_path={data_path}"
            )
        elif use_rmax_like:
            data_overrides.append(
                f"domains.{domain}.rmax_like_data_save_path={data_path}"
            )
        if hasattr(cfg, "p2e"):
            data_overrides.append(
                f"p2e.enabled={str(bool(cfg.p2e.enabled)).lower()}"
            )
        if use_single_p2e:
            data_overrides.append("env.collect.data_type=p2e")
        data_overrides.extend(rmax_like_runtime_overrides(cfg, domain))
        data_overrides.extend(p2e_runtime_overrides(cfg) if use_single_p2e else [])
        if configured_continual is not None:
            data_overrides.append(
                f"domains.{domain}.continual_learning.enabled="
                f"{str(continual_enabled).lower()}"
            )
        if use_single_p2e:
            collector_module = "modelBased.continue_learning.crafter_curriculum_collect"
            collection_mode = "single-environment P2E"
        elif use_rmax_like:
            collector_module = "modelBased.exploration.crafter_rmax_collect"
            collection_mode = "RMax-like count exploration"
        else:
            collector_module = "modelBased.data.data_collect"
            collection_mode = "standard"
        force_collection_override = []
        if use_single_p2e and force:
            force_collection_override.append(
                f"domains.{domain}.continual_learning.force_collection=true"
            )
        data_recollected = False
        # P2E owns its one-million-interaction budget; random/RMax retain the
        # domain collection budget so acquisition modes cannot overwrite one
        # another's identity.
        expected_data_size = int(
            cfg.p2e.total_steps if use_single_p2e else domain_cfg.data_collection.maximum_dataset_size
        )
        p2e_artifacts_ready = (
            not use_single_p2e
            or (
                dataset_transition_count(data_path) == expected_data_size
                and world_model_path.is_file()
            )
        )
        rmax_artifacts_ready = (
            not use_rmax_like
            or (
                dataset_transition_count(data_path) == expected_data_size
                and Path(str(cfg.rmax_like.checkpoint_path)).expanduser().resolve().is_file()
            )
        )
        if (
            data_path.exists()
            and not force
            and dataset_matches(data_path, cfg, domain)
            and p2e_artifacts_ready
            and rmax_artifacts_ready
        ):
            print(f"[SKIP] Dataset already exists: {data_path}")
        elif data_path.exists() and not force:
            reason = (
                "P2E dataset size/final WM is incomplete"
                if use_single_p2e and dataset_matches(data_path, cfg, domain)
                else "RMax-like dataset/explorer artifact is incomplete"
                if use_rmax_like and dataset_matches(data_path, cfg, domain)
                else "dataset identity does not match"
            )
            print(f"[RECOLLECT] {reason} for {domain}: {data_path}")
            run_command(
                [
                    python,
                    "-m",
                    collector_module,
                    f"domain={domain}",
                    *data_overrides,
                    *force_collection_override,
                ],
                f"{domain} / recollect {collection_mode} data",
                cfg,
            )
            data_recollected = True
        else:
            run_command(
                [
                    python,
                    "-m",
                    collector_module,
                    f"domain={domain}",
                    *data_overrides,
                    *force_collection_override,
                ],
                f"{domain} / collect {collection_mode} data",
                cfg,
            )
            data_recollected = True

        # --- 2. World Model Training ---
        if use_single_p2e:
            if not world_model_path.is_file():
                raise RuntimeError(
                    "Single-environment P2E collection completed without its "
                    f"persistent WM checkpoint: {world_model_path}"
                )
            print(
                "[SKIP] P2E already trained and consolidated the same online "
                f"WM for planning: {world_model_path}"
            )
        elif world_model_path.exists() and not force and not data_recollected:
            print(f"[SKIP] World-model checkpoint already exists: {world_model_path}")
        else:
            if world_model_path.exists() and data_recollected:
                print(
                    "[RETRAIN] Dataset was recollected for the canonical layout; "
                    "the existing world-model checkpoint is stale."
                )
            run_command(
                [
                    python,
                    "-m",
                    "modelBased.world_model.AttentionWM_training",
                    f"domain={domain}",
                    f"attention_model.model_save_path={world_model_path}",
                    *data_overrides,
                ],
                f"{domain} / train world model",
                cfg,
            )

    if bool(cfg.pipeline.skip_policy):
        print(f"[SKIP] Policy stage disabled for {domain}")
    elif domain == "minigrid" and str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower() in {"mpc", "mcts", "astar"}:
        control_mode = str(cfg.PPO.wm_control_mode).lower()
        print(f"[{domain}] Running online WM {control_mode.upper()} evaluation; PPO training is disabled by flag.")
        control_cfg = getattr(cfg.PPO, control_mode)
        if control_mode == "mpc":
            control_names = (
                "horizon", "cpu_threads", "execute_steps", "population", "elite_count", "iterations", "gamma",
                "fallback_action", "episodes", "print_every_steps", "output_dir",
            )
        elif control_mode == "mcts":
            control_names = (
                "simulations", "max_depth", "rollout_depth", "exploration_constant", "gamma",
                "goal_reward", "progress_reward", "inventory_change_bonus", "door_open_bonus",
                "step_penalty", "invalid_action_penalty", "uncertainty_penalty", "heuristic_weight",
                "fallback_action", "episodes", "print_every_steps", "output_dir",
            )
        else:
            control_names = (
                "execution_mode", "max_depth", "max_expansions",
                "progress_every_expansions", "max_real_steps",
                "initial_directions", "seeds", "validate_real", "save_gif",
                "rollout_plans_in_real_env", "evaluate_all_targets", "target_dir",
                "gif_fps", "print_every_steps", "output_dir",
            )
        control_overrides = [
            f"PPO.{control_mode}.{name}={getattr(control_cfg, name)}"
            for name in (
                *control_names,
            )
            if hasattr(control_cfg, name)
        ]
        if control_mode == "mpc":
            control_overrides.append(
                "PPO.use_main_dense_reward="
                f"{str(bool(cfg.PPO.use_main_dense_reward)).lower()}"
            )
            control_overrides.extend(
                f"PPO.main_dense_reward.{name}={value}"
                for name, value in cfg.PPO.main_dense_reward.items()
            )
        run_command(
            [
                python,
                "-m",
                (
                    "modelBased.policy_training.planners.dijkstra_planner"
                    if control_mode == "astar"
                    else f"modelBased.policy_training.planners.{control_mode}_planner"
                ),
                "domain=minigrid",
                f"PPO.checkpoint_path_wm={policy_world_model_path}",
                *policy_overrides,
                *control_overrides,
            ],
            f"{domain} / online WM {control_mode.upper()}",
            cfg,
        )
    elif domain == "minigrid":
        policy_path = policy_checkpoint_path(cfg, domain=domain)
        policy_compatible = policy_checkpoint_is_compatible(
            policy_path, cfg, domain=domain
        )
        if policy_path.exists() and not policy_compatible:
            print(
                f"[RETRAIN] Existing policy uses an incompatible observation/action "
                f"shape: {policy_path}"
            )
        if policy_compatible and not force and not force_policy and not data_recollected:
            print(f"[SKIP] Policy checkpoint already exists: {policy_path}")
        else:
            if policy_path.exists() and data_recollected:
                print(
                    "[RETRAIN] The world model was retrained for the canonical "
                    "layout; the existing policy checkpoint is stale."
                )
            run_command(
                [
                    python,
                    "-m",
                    "modelBased.policy_training.ppo.PPO_world_training",
                    "domain=minigrid",
                    f"PPO.checkpoint_path={policy_path}",
                    f"PPO.checkpoint_path_wm={policy_world_model_path}",
                    *policy_overrides,
                ],
                f"{domain} / train policy",
                cfg,
            )

            # Auto-run validation
            print(f"[{domain}] Starting policy validation in real environment...")
            run_command(
                [
                    python,
                    "-m",
                    "modelBased.policy_training.ppo.PPO_world_test",
                    "domain=minigrid",
                    f"PPO.checkpoint_path={policy_path}",
                    *policy_overrides,
                ],
                f"{domain} / test policy",
                cfg,
            )
    elif domain == "crafter":
        # A Crafter imagined-policy run must consume the exact WM semantics
        # saved with this checkpoint, rather than the pipeline's training cfg.
        # Real-environment PPO intentionally has no WM dependency.
        if not bool(getattr(cfg.PPO, "train_in_real_env", False)):
            crafter_spec = inspect_crafter_planning_checkpoint(policy_world_model_path)
            print(f"[crafter] Planning WM contract: {crafter_spec}")
        policy_path = policy_checkpoint_path(cfg, domain=domain)
        policy_compatible = policy_checkpoint_is_compatible(
            policy_path, cfg, domain=domain
        )
        if policy_path.exists() and not policy_compatible:
            print(
                f"[RETRAIN] Existing Crafter policy does not match the current "
                f"layout shape: {policy_path}"
            )
        if policy_compatible and not force and not force_policy and not data_recollected:
            print(f"[SKIP] Crafter policy checkpoint already exists: {policy_path}")
        else:
            if policy_path.exists() and data_recollected:
                print(
                    "[RETRAIN] The world model was retrained; "
                    "the existing Crafter policy checkpoint is stale."
                )
            run_command(
                [
                    python,
                    "-m",
                    "modelBased.policy_training.ppo.PPO_crafter_training",
                    "domain=crafter",
                    f"PPO.checkpoint_path={policy_path}",
                    f"PPO.checkpoint_path_wm={policy_world_model_path}",
                    *policy_overrides,
                ],
                f"{domain} / train policy",
                cfg,
            )

            # Training completed; validation runs below for both newly trained
            # and already compatible checkpoints.

        # Always validate the policy as the final Crafter stage. This also
        # produces the optional GIF when PPO.save_gif=true.
        print(f"[{domain}] Starting policy validation in real environment...")
        run_command(
            [
                python,
                "-m",
                "modelBased.policy_training.ppo.PPO_crafter_test",
                "domain=crafter",
                f"PPO.checkpoint_path={policy_path}",
                *policy_overrides,
            ],
            f"{domain} / test policy",
            cfg,
        )
    else:
        print(
            f"[SKIP] Policy stage for {domain}: no PPO world-model "
            "implementation available for this domain."
        )


@hydra.main(
    version_base=None,
    config_path="modelBased/config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run one domain or all domains according to Hydra configuration."""
    label = str(cfg.pipeline.label).lower()
    valid_labels = {"minigrid", "crafter", "bipedalwalker", "full"}
    if label not in valid_labels:
        raise ValueError(f"pipeline.label must be one of {sorted(valid_labels)}, got {label!r}")

    import time
    from datetime import timedelta
    
    print("Pipeline configuration:")
    print(OmegaConf.to_yaml(cfg.pipeline, resolve=True))
    
    pipeline_start = time.time()

    domains = ["minigrid", "crafter", "bipedalwalker"] if label == "full" else [label]
    for domain in domains:
        run_domain(domain, cfg)
        
    pipeline_end = time.time()
    elapsed = pipeline_end - pipeline_start
    print("\n========================================================")
    print("🎉 MBRL Pipeline Completed!")
    print(f"Total time (Data Collection -> WM Train -> Planning): {timedelta(seconds=int(elapsed))}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
