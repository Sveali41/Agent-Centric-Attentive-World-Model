"""Hydra entry point for the data -> world model -> policy pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from modelBased.common.dataset_identity import dataset_matches
from modelBased.policy_training.experiment_naming import (
    policy_checkpoint_is_compatible,
    policy_checkpoint_path,
)
from modelBased.common.artifact_naming import (
    p2e_enabled,
    single_environment_dataset_path,
    world_model_checkpoint_path,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modelBased" / "models"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass




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
    return overrides


def domain_runtime_overrides(cfg: DictConfig, domain: str) -> list[str]:
    """Forward domain task and collection budgets to stage subprocesses."""
    domain_cfg = cfg.domains[domain]
    overrides = [f"domains.{domain}.task_name={domain_cfg.task_name}"]
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
    if hasattr(domain_cfg, "collection_replace_start_with_empty"):
        value = bool(domain_cfg.collection_replace_start_with_empty)
        overrides.append(
            "domains."
            f"{domain}.collection_replace_start_with_empty="
            f"{str(value).lower()}"
        )
    return overrides


def run_domain(domain: str, cfg: DictConfig) -> None:
    domain_cfg = cfg.domains[domain]
    force = bool(cfg.pipeline.force)
    force_policy = bool(getattr(cfg.pipeline, "force_policy", False))
    skip_world_model = bool(getattr(cfg.pipeline, "skip_world_model", False))
    python = sys.executable
    layout_path = Path(str(domain_cfg.layout_path)).expanduser().resolve()
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
    ]
    if hasattr(cfg, "p2e"):
        policy_identity_overrides.append(
            f"p2e.enabled={str(bool(cfg.p2e.enabled)).lower()}"
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
    if continual_enabled and train_in_real_env:
        raise ValueError(
            f"[{domain}] continual WM training cannot be combined with "
            "PPO.train_in_real_env=true. Disable continual_learning or use WM planning."
        )

    if skip_world_model:
        if not world_model_path.is_file():
            raise FileNotFoundError(
                f"[{domain}] pipeline.skip_world_model=true but the configured "
                f"WM checkpoint does not exist: {world_model_path}"
            )
        print(
            f"[{domain}] Reusing existing world-model checkpoint and skipping "
            f"all data/WM stages: {world_model_path}"
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
                        "modelBased.continue_learning.crafter_curriculum_collect",
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
        data_path = single_environment_dataset_path(cfg, domain)
        data_overrides = [
            *domain_overrides,
            f"domains.{domain}.data_save_path={data_path}",
            f"env.collect.data_save_path={data_path}",
            f"attention_model.data_dir={data_path}",
        ]
        if hasattr(cfg, "p2e"):
            data_overrides.append(
                f"p2e.enabled={str(bool(cfg.p2e.enabled)).lower()}"
            )
        if configured_continual is not None:
            data_overrides.append(
                f"domains.{domain}.continual_learning.enabled="
                f"{str(continual_enabled).lower()}"
            )
        collector_module = (
            "modelBased.continue_learning.crafter_curriculum_collect"
            if use_single_p2e
            else "modelBased.data.data_collect"
        )
        collection_mode = "single-environment P2E" if use_single_p2e else "standard"
        force_collection_override = []
        if use_single_p2e and force:
            force_collection_override.append(
                f"domains.{domain}.continual_learning.force_collection=true"
            )
        data_recollected = False
        expected_data_size = int(domain_cfg.data_collection.maximum_dataset_size)
        p2e_artifacts_ready = (
            not use_single_p2e
            or (
                dataset_transition_count(data_path) == expected_data_size
                and world_model_path.is_file()
            )
        )
        if (
            data_path.exists()
            and not force
            and dataset_matches(data_path, cfg, domain)
            and p2e_artifacts_ready
        ):
            print(f"[SKIP] Dataset already exists: {data_path}")
        elif data_path.exists() and not force:
            reason = (
                "P2E dataset size/final WM is incomplete"
                if use_single_p2e and dataset_matches(data_path, cfg, domain)
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
                    "modelBased.policy_training.PPO_world_training",
                    "domain=minigrid",
                    f"PPO.checkpoint_path={policy_path}",
                    f"PPO.checkpoint_path_wm={world_model_path}",
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
                    "modelBased.policy_training.PPO_world_test",
                    "domain=minigrid",
                    f"PPO.checkpoint_path={policy_path}",
                    *policy_overrides,
                ],
                f"{domain} / test policy",
                cfg,
            )
    elif domain == "crafter":
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
                    "modelBased.policy_training.PPO_crafter_training",
                    "domain=crafter",
                    f"PPO.checkpoint_path={policy_path}",
                    f"PPO.checkpoint_path_wm={world_model_path}",
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
                "modelBased.policy_training.PPO_crafter_test",
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
