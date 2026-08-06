"""Hydra entry point for the data -> world model -> policy pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from modelBased.common.dataset_identity import dataset_matches

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


def run_domain(domain: str, cfg: DictConfig) -> None:
    domain_cfg = cfg.domains[domain]
    pipeline_cfg = cfg.pipeline.domains[domain]
    force = bool(cfg.pipeline.force)
    force_policy = bool(getattr(cfg.pipeline, "force_policy", False))
    python = sys.executable
    layout_path = Path(str(domain_cfg.layout_path)).expanduser().resolve()
    if not layout_path.exists():
        raise FileNotFoundError(
            f"[{domain}] Canonical layout does not exist: {layout_path}"
        )
    print(f"[{domain}] Canonical layout: {layout_path}", flush=True)

    train_in_real_env = getattr(cfg.PPO, "train_in_real_env", False)
    if train_in_real_env:
        print(f"[{domain}] PPO.train_in_real_env is True. Skipping data collection and world-model training.")
        data_recollected = False
    else:
        # --- 1. Dataset Collection ---
        data_path = Path(str(domain_cfg.data_save_path))
        data_recollected = False
        if data_path.exists() and not force and dataset_matches(data_path, cfg, domain):
            print(f"[SKIP] Dataset already exists: {data_path}")
        elif data_path.exists() and not force:
            print(f"[RECOLLECT] Dataset identity does not match {domain}: {data_path}")
            run_command(
                [python, "-m", "modelBased.data.data_collect", f"domain={domain}"],
                f"{domain} / recollect data",
                cfg,
            )
            data_recollected = True
        else:
            run_command(
                [python, "-m", "modelBased.data.data_collect", f"domain={domain}"],
                f"{domain} / collect data",
                cfg,
            )
            data_recollected = True

        # --- 2. World Model Training ---
        world_model_path = Path(str(domain_cfg.world_model_checkpoint))
        if world_model_path.exists() and not force and not data_recollected:
            print(f"[SKIP] World-model checkpoint already exists: {world_model_path}")
        else:
            if world_model_path.exists() and data_recollected:
                print(
                    "[RETRAIN] Dataset was recollected for the canonical layout; "
                    "the existing world-model checkpoint is stale."
                )
            run_command(
                [python, "-m", "modelBased.world_model.AttentionWM_training", f"domain={domain}"],
                f"{domain} / train world model",
                cfg,
            )

    world_model_path = Path(str(domain_cfg.world_model_checkpoint))

    if bool(cfg.pipeline.skip_policy):
        print(f"[SKIP] Policy stage disabled for {domain}")
    elif domain != "minigrid":
        print(
            f"[SKIP] Policy stage for {domain}: the current PPO world-model "
            "implementation is MiniGrid-specific."
        )
    else:
        policy_path = Path(str(pipeline_cfg.policy_checkpoint))
        if policy_path.exists() and not force and not force_policy and not data_recollected:
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
                ],
                f"{domain} / test policy",
                cfg,
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

    print("Pipeline configuration:")
    print(OmegaConf.to_yaml(cfg.pipeline, resolve=True))

    domains = ["minigrid", "crafter", "bipedalwalker"] if label == "full" else [label]
    for domain in domains:
        run_domain(domain, cfg)


if __name__ == "__main__":
    main()
