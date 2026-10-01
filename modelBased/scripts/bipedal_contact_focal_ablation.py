"""Compare next-contact BCE with KEEP/CHANGE focal loss on Bipedal target 1."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WM_ROOT = ROOT / "wm"
DATA_ROOT = ROOT / "trainer" / "data" / "bipedalwalker" / "target_tasks"
TRAIN_PATH = DATA_ROOT / "bipedal_target_task_1_random.npz"
VALIDATION_PATH = DATA_ROOT / "bipedal_target_task_1_uniform.npz"
SEED = int(os.environ.get("BIPEDAL_SEED", "0"))
FOCAL_GAMMA = float(os.environ.get("BIPEDAL_FOCAL_GAMMA", "1.0"))
GAMMA_TAG = f"{FOCAL_GAMMA:g}".replace(".", "p")
OUTPUT_NAME = (
    f"bipedal_contact_focal_target1_seed{SEED}"
    if FOCAL_GAMMA == 1.0
    else f"bipedal_contact_focal_gamma{GAMMA_TAG}_target1_seed{SEED}"
)
OUTPUT_ROOT = ROOT / "trainer" / "results" / OUTPUT_NAME
CONFIG_PATH = WM_ROOT / "modelBased" / "config" / "config.yaml"

for key, value in {
    "WM_ROOT": str(WM_ROOT),
    "TRAINER_PATH": str(ROOT / "trainer"),
    "TRAIN_DATASET_PATH": str(ROOT / "trainer" / "data"),
    "MODEL_FPATH": str(OUTPUT_ROOT / "models"),
    "ENV_PATH": str(ROOT / "trainer" / "data" / "bipedalwalker"),
}.items():
    os.environ[key] = value

if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.world_model.AttentionWM_training import train_api

EPOCHS = 10
BATCH_SIZE = 64


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ("a", "b", "c")}


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def metrics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "avg_val_loss",
        "best_loss",
        "val/observation_loss",
        "val/natural_ce",
        "val/leg1_contact_next_contact_bce",
        "val/leg2_contact_next_contact_bce",
        "val/leg1_contact_next_contact_accuracy",
        "val/leg2_contact_next_contact_accuracy",
        "val/leg1_contact_switch_focal_loss",
        "val/leg2_contact_switch_focal_loss",
        "val/leg1_contact_switch_recall",
        "val/leg2_contact_switch_recall",
        "val/leg1_contact_switch_precision",
        "val/leg2_contact_switch_precision",
        "val/leg1_contact_keep_false_switch_rate",
        "val/leg2_contact_keep_false_switch_rate",
        "val/hull_pose_nll",
        "val/hull_vel_nll",
        "val/leg1_hip_nll",
        "val/leg1_knee_nll",
        "val/leg2_hip_nll",
        "val/leg2_knee_nll",
        "val/lidar_near_nll",
        "val/lidar_far_nll",
    )
    metrics = dict(result)
    # Lightning validation-only mode returns callback metrics inside the first
    # item of ``avg_val_loss``; training mode exposes them at the top level.
    validation_rows = result.get("avg_val_loss")
    if (
        isinstance(validation_rows, (list, tuple))
        and validation_rows
        and isinstance(validation_rows[0], dict)
    ):
        metrics.update(validation_rows[0])
    return {key: serializable(metrics[key]) for key in keys if key in metrics}


def make_config(*, arm: str, schema: list[dict], gamma: float, freeze: bool = False):
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.domain = "bipedalwalker"
    cfg.domains.bipedalwalker.task_name = "bipedal_target_task_1"
    cfg.domains.bipedalwalker.data_save_path = str(TRAIN_PATH)
    cfg.domains.bipedalwalker.validation_data_dir = str(VALIDATION_PATH)
    cfg.domains.bipedalwalker.focal_gamma = float(gamma)
    cfg.domains.bipedalwalker.observation_schema = copy.deepcopy(schema)
    cfg.env.collect.data_type = "random"

    model_dir = OUTPUT_ROOT / arm
    cfg.attention_model.env_type = "bipedalwalker"
    cfg.attention_model.data_dir = str(TRAIN_PATH)
    cfg.attention_model.validation_data_dir = str(VALIDATION_PATH)
    cfg.attention_model.model_save_path = str(model_dir / "model.ckpt")
    cfg.attention_model.checkpoint_dir = str(model_dir / "checkpoints")
    cfg.attention_model.metrics_dir = str(model_dir / "metrics")
    cfg.attention_model.observation_schema = copy.deepcopy(schema)
    cfg.attention_model.focal_gamma = float(gamma)
    cfg.attention_model.seed = SEED
    cfg.attention_model.batch_size = BATCH_SIZE
    cfg.attention_model.n_epochs = EPOCHS
    cfg.attention_model.lr = 5e-4
    cfg.attention_model.wd = 1e-5
    cfg.attention_model.n_cpu = 0
    cfg.attention_model.freeze_weight = freeze
    cfg.attention_model.allow_legacy_dataset = True
    cfg.attention_model.save_local_metrics = True
    cfg.attention_model.use_wandb = False
    cfg.attention_model.enable_progress_bar = True
    cfg.attention_model.transition_balanced_sampling = False
    cfg.attention_model.continue_learning = False
    cfg.attention_model.ewc_enabled = False
    cfg.attention_model.compute_fisher = False
    cfg.attention_model.max_validation_samples = 0
    cfg.attention_model.validation_subset_seed = SEED
    return cfg


def run_validation(cfg, model, validation_data):
    validation_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    validation_cfg.attention_model.data_dir = str(VALIDATION_PATH)
    validation_cfg.attention_model.freeze_weight = True
    validation_cfg.env.collect.data_type = "uniform"
    result, _, _ = train_api(
        validation_cfg,
        net=model,
        direct_data=validation_data,
    )
    return metrics_from_result(result)


def main() -> None:
    if not TRAIN_PATH.is_file() or not VALIDATION_PATH.is_file():
        raise FileNotFoundError(
            f"Expected existing target-1 datasets at {TRAIN_PATH} and {VALIDATION_PATH}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_data = load_arrays(TRAIN_PATH)
    validation_data = load_arrays(VALIDATION_PATH)

    base_cfg = OmegaConf.load(CONFIG_PATH)
    new_schema = OmegaConf.to_container(
        base_cfg.domains.bipedalwalker.observation_schema, resolve=False
    )
    old_schema = copy.deepcopy(new_schema)
    for spec in old_schema:
        if spec.get("distribution") == "bernoulli_effect":
            spec["distribution"] = "bernoulli"

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(SEED, workers=True)
    init_cfg = make_config(
        arm="next_contact_bce", schema=old_schema, gamma=0.0
    )
    initial_model = AttentionWorldModel(init_cfg.attention_model)
    gamma1_initial_path = (
        ROOT / "trainer" / "results"
        / f"bipedal_contact_focal_target1_seed{SEED}"
        / "shared_initial_state.pt"
    )
    if FOCAL_GAMMA != 1.0 and gamma1_initial_path.is_file():
        initial_state = torch.load(
            gamma1_initial_path, map_location="cpu", weights_only=True
        )
    else:
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in initial_model.state_dict().items()
        }
    torch.save(initial_state, OUTPUT_ROOT / "shared_initial_state.pt")

    results: dict[str, Any] = {
        "experiment": f"bipedal_target1_next_contact_bce_vs_change_focal_gamma{GAMMA_TAG}",
        "seed": SEED,
        "train_data": str(TRAIN_PATH),
        "validation_data": str(VALIDATION_PATH),
        "train_samples": int(len(train_data["a"])),
        "validation_samples": int(len(validation_data["a"])),
        "split_protocol": f"WMRLDataModule first 90% train / final 10% internal validation; shuffled training with seed {SEED}",
        "external_validation_protocol": "all 30000 fixed target-1 uniform transitions",
        "continuous_target": "normalized next_state - current_state, MSE",
        "baseline_contact_target": "absolute next contact state, BCE",
        "focal_contact_target": f"contact KEEP/CHANGE from current and next state, binary focal loss, gamma={FOCAL_GAMMA:g}, no class weight",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": {"name": "Adam", "lr": 5e-4, "weight_decay": 1e-5},
        "arms": {},
    }
    focal_arm = f"contact_change_focal_gamma{GAMMA_TAG}"
    for arm, schema, gamma in (
        ("next_contact_bce", old_schema, 0.0),
        (focal_arm, new_schema, FOCAL_GAMMA),
    ):
        print(f"\n===== Running {arm} =====", flush=True)
        cfg = make_config(arm=arm, schema=schema, gamma=gamma)
        pl.seed_everything(SEED, workers=True)
        model = AttentionWorldModel(cfg.attention_model)
        model.load_state_dict(initial_state, strict=True)
        before = run_validation(cfg, model, validation_data)

        pl.seed_everything(SEED, workers=True)
        train_result, _, model = train_api(
            cfg,
            net=model,
            direct_data=train_data,
        )
        after = run_validation(cfg, model, validation_data)
        results["arms"][arm] = {
            "gamma": gamma,
            "contact_target": "next_state" if arm == "next_contact_bce" else "keep_change",
            "before_validation": before,
            "training_summary": metrics_from_result(train_result),
            "after_validation": after,
            "checkpoint": str(OUTPUT_ROOT / arm / "model.ckpt"),
        }
        (OUTPUT_ROOT / "results.json").write_text(
            json.dumps(serializable(results), indent=2, sort_keys=True) + "\n"
        )

    print(f"\nResults saved to {OUTPUT_ROOT / 'results.json'}", flush=True)
    write_three_seed_summary()


def write_three_seed_summary() -> None:
    seed_results = {}
    for seed in (0, 1, 2):
        seed_name = (
            f"bipedal_contact_focal_target1_seed{seed}"
            if FOCAL_GAMMA == 1.0
            else f"bipedal_contact_focal_gamma{GAMMA_TAG}_target1_seed{seed}"
        )
        result_path = ROOT / "trainer" / "results" / seed_name / "results.json"
        if not result_path.is_file():
            return
        seed_results[seed] = json.loads(result_path.read_text())

    focal_arm = f"contact_change_focal_gamma{GAMMA_TAG}"
    summary: dict[str, Any] = {
        "experiment": f"bipedal_target1_next_contact_bce_vs_change_focal_gamma{GAMMA_TAG}",
        "focal_gamma": FOCAL_GAMMA,
        "seeds": [0, 1, 2],
        "validation_data": str(VALIDATION_PATH),
        "arms": {},
    }
    for arm in ("next_contact_bce", focal_arm):
        metric_names = set.intersection(*[
            {
                key for key, value in result["arms"][arm]["after_validation"].items()
                if isinstance(value, (int, float))
            }
            for result in seed_results.values()
        ])
        arm_summary = {"after_validation": {}}
        for metric in sorted(metric_names):
            values = [
                float(seed_results[seed]["arms"][arm]["after_validation"][metric])
                for seed in (0, 1, 2)
            ]
            arm_summary["after_validation"][metric] = {
                "mean": float(np.mean(values)),
                "std_population": float(np.std(values)),
                "per_seed": dict(zip((0, 1, 2), values)),
            }
        summary["arms"][arm] = arm_summary

    summary_path = (
        ROOT / "trainer" / "results"
        / f"bipedal_contact_focal_target1_gamma{GAMMA_TAG}_3seed_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Three-seed summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
