"""Train and evaluate MiniGrid policies for a baseline WM across seeds/targets.

For every (seed, target) pair this runner:

1. trains PPO with the corresponding baseline world-model checkpoint;
2. evaluates the frozen-WM-selected best policy in the real MiniGrid environment; and
3. appends the dense-reward and success summary to one CSV.

The script only orchestrates existing entry points.  It does not implement a
second reward function: ``PPO.use_main_dense_reward=True`` and
``PPO.test_reward_mode=wm`` select the shared custom dense reward used by
``PPO_world_training.py`` and ``PPO_world_test.py``.

Example (the default is seeds 0..4 and targets 0..2)::

    python wm/modelBased/policy_training/run_policy_target_eval.py --baseline dr

Small smoke run::

    python wm/modelBased/policy_training/run_policy_target_eval.py \
        --train-timesteps 32768 --test-episodes 20 --test-envs 8

Use ``--dry-run`` to inspect all commands without starting training.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[3]
WM_ROOT = REPO_ROOT / "wm"
TRAINING_ENTRY = WM_ROOT / "modelBased" / "policy_training" / "PPO_world_training.py"
EVAL_ENTRY = WM_ROOT / "modelBased" / "policy_training" / "PPO_world_test.py"
DEFAULT_WM_DIR = WM_ROOT / "modelBased" / "models" / "AttentionWM"
DEFAULT_LAYOUT_DIR = REPO_ROOT / "trainer" / "level" / "minigrid" / "target_task"


def _parse_int_list(value: str, name: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated list of integers: {value!r}"
            ) from exc
        if number < 0:
            raise argparse.ArgumentTypeError(f"{name} cannot contain negative values")
        result.append(number)
    if not result:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return result


def _env_for_children() -> dict[str, str]:
    """Use this repository as the canonical trainer/WM workspace."""
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(REPO_ROOT),
            "TRAINER_ROOT": str(REPO_ROOT),
            "WM_ROOT": str(WM_ROOT),
            "TRAINER_PATH": str(REPO_ROOT / "trainer"),
            "ENV_PATH": str(REPO_ROOT / "trainer" / "level"),
            "WORLD_MODEL_PATH": str(WM_ROOT / "modelBased"),
            "MODEL_FPATH": str(WM_ROOT / "modelBased" / "models"),
            "TRAIN_DATASET_PATH": str(WM_ROOT / "modelBased" / "data" / "train_world_model"),
        }
    )
    return env


def _common_overrides(
    *,
    seed: int,
    target: int,
    layout_path: Path,
    wm_checkpoint: Path,
    policy_checkpoint: Path,
    output_root: Path,
    train_timesteps: int | None,
    test_episodes: int,
    test_envs: int,
    use_wandb: bool,
    test_deterministic: bool = False,
) -> list[str]:
    """Build shared task/reward/path overrides with an explicit test action mode."""
    # Explicit paths avoid Hydra interpolation retaining target_task2 from the
    # base config and keep all aggregate trainer artifacts outside wm/outputs.
    evaluation_dir = output_root / "evaluation"
    overrides = [
        "domain=minigrid",
        f"PPO.seed={seed}",
        f"domains.minigrid.task_name=target_task{target}",
        f"domains.minigrid.layout_path={layout_path}",
        f"PPO.env_path={layout_path}",
        f"PPO.checkpoint_path_wm={wm_checkpoint}",
        f"PPO.checkpoint_path={policy_checkpoint}",
        "PPO.train_in_real_env=false",
        "PPO.wm_control_mode=ppo",
        "PPO.use_main_dense_reward=true",
        "PPO.test_reward_mode=wm",
        f"PPO.total_test_episodes={test_episodes}",
        f"PPO.test_num_envs={test_envs}",
        f"PPO.test_deterministic={'true' if test_deterministic else 'false'}",
        "PPO.save_csv=true",
        f"PPO.save_path_csv={evaluation_dir}",
        f"PPO.use_wandb={'true' if use_wandb else 'false'}",
        f"paths.outputs={output_root}",
        f"paths.results={output_root / 'results'}",
        f"paths.evaluation={output_root / 'evaluation'}",
        f"paths.wandb={output_root / 'wandb'}",
    ]
    if train_timesteps is not None:
        overrides.append(f"PPO.max_training_timesteps={train_timesteps}")
    return overrides


def _policy_artifact_paths(canonical_path: Path) -> dict[str, Path]:
    """Return WM PPO selection artifacts, retaining legacy canonical support."""
    suffix = canonical_path.suffix or ".ckpt"
    stem = canonical_path.stem if canonical_path.suffix else canonical_path.name
    return {
        "canonical": canonical_path,
        "best": canonical_path.with_name(f"{stem}_best{suffix}"),
        "last": canonical_path.with_name(f"{stem}_last{suffix}"),
        "manifest": canonical_path.with_name("policy_selection.json"),
    }


def _select_evaluation_policy(artifacts: dict[str, Path]) -> Path:
    """Prefer selected WM-PPO policy, falling back to legacy policy.ckpt."""
    return artifacts["best"] if artifacts["best"].is_file() else artifacts["canonical"]


def _run_command(command: list[str], env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    print("[RUN]", " ".join(str(part) for part in command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            # Keep the full child log while also exposing live progress to the
            # terminal; this is important for multi-hour PPO runs.
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See {log_path}"
        )


def _read_evaluation_summary(eval_csv: Path) -> dict[str, float | str]:
    """Read the evaluator's aggregate ``mean`` row."""
    if not eval_csv.is_file():
        raise FileNotFoundError(f"Evaluation CSV was not created: {eval_csv}")
    with eval_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluation CSV is empty: {eval_csv}")
    row = next((item for item in rows if item.get("episode") == "mean"), rows[-1])

    def number(name: str, default: float = 0.0) -> float:
        value = row.get(name, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return {
        "mean_reward": number("reward"),
        "mean_native_reward": number("native_reward"),
        "success_rate": number("success"),
        "mean_steps": number("steps"),
        "goal_dist_mean": number("goal_dist_mean"),
        "goal_dist_min": number("goal_dist_min"),
        "goal_region_entry_count": number("goal_region_entry_count"),
        "progress_milestone_count": number("progress_milestone_count"),
        "critical_door_crossing_count": number("critical_door_crossing_count"),
        "wm_shaped_reward": number("wm_shaped_reward"),
    }


def _upsert_summary_row(summary_csv: Path, row: dict[str, object]) -> None:
    """Insert a result, replacing an older row for the same baseline/seed/target."""
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing: list[dict[str, str]] = []
    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        # Preserve any columns from an existing summary while using the new
        # schema for newly generated rows.
        for old_name in existing[0].keys() if existing else []:
            if old_name not in fieldnames:
                fieldnames.append(old_name)
    def same_identity(old: dict[str, str]) -> bool:
        if (str(old.get("seed")) != str(row.get("seed"))
                or str(old.get("target")) != str(row.get("target"))):
            return False
        old_baseline = str(old.get("baseline", "")).strip()
        new_baseline = str(row.get("baseline", "")).strip()
        # Legacy summaries predate the baseline column and are DR results.
        return old_baseline == new_baseline or (not old_baseline and new_baseline == "dr")

    existing = [old for old in existing if not same_identity(old)]
    existing.append({name: row.get(name, "") for name in fieldnames})
    # The file is rewritten atomically enough for this single-process runner
    # and remains a normal CSV for W&B/Excel readers.
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)


def _resolve_run_id(output_root: Path, requested: str | None) -> str:
    if requested:
        run_root = output_root / requested
        if not run_root.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_root}")
        return requested
    candidates = []
    if output_root.is_dir():
        candidates = [
            path for path in output_root.iterdir()
            if path.is_dir()
            and (
                any(path.glob("target_task*/seed*/policy_best.ckpt"))
                or any(path.glob("target_task*/seed*/policy.ckpt"))
            )
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No completed policy batch found under {output_root}; pass --run-id explicitly"
        )
    return max(candidates, key=lambda path: path.name).name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("dr", "mac", "p2e", "target"),
        default="dr",
        help="World-model baseline used for policy training/evaluation (default: dr)",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Policy/WM seeds")
    parser.add_argument("--targets", default="0,1,2", help="MiniGrid target indices")
    parser.add_argument(
        "--wm-checkpoint-template",
        default=None,
        help="WM checkpoint template; may contain {baseline} and {seed}",
    )
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: outputs/results/{baseline}_policy_target_eval)",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=None,
        help="Override PPO.max_training_timesteps; default uses config.yaml",
    )
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-envs", type=int, default=128)
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip PPO training and evaluate policies already in --run-id",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing timestamp directory for --eval-only; default is latest completed batch",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    targets = _parse_int_list(args.targets, "--targets")
    if args.train_timesteps is not None and args.train_timesteps < 1:
        raise SystemExit("--train-timesteps must be positive")
    if args.test_episodes < 1 or args.test_envs < 1:
        raise SystemExit("--test-episodes and --test-envs must be positive")

    wm_checkpoint_template = args.wm_checkpoint_template or str(
        DEFAULT_WM_DIR / "{baseline}_attention_world_model_minigrid_none_effect_seed{seed}.ckpt"
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else REPO_ROOT / "outputs" / "results" / f"{args.baseline}_policy_target_eval"
    ).expanduser().resolve()
    run_id = (
        _resolve_run_id(output_root, args.run_id)
        if args.eval_only
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root = output_root / run_id
    summary_csv = output_root / f"{args.baseline}_policy_target_summary.csv"
    env = _env_for_children()
    failures = 0

    print(f"[{args.baseline} policy batch] seeds={seeds} targets={targets}")
    print(f"[{args.baseline} policy batch] output root={run_root}")
    print(f"[{args.baseline} policy batch] reward=main_dense, evaluation reward_mode=wm")

    for seed in seeds:
        wm_checkpoint = Path(
            wm_checkpoint_template.format(baseline=args.baseline, seed=seed)
        ).expanduser().resolve()
        if not wm_checkpoint.is_file():
            message = f"Missing {args.baseline} WM checkpoint for seed {seed}: {wm_checkpoint}"
            if args.continue_on_error:
                print("[ERROR]", message)
                failures += len(targets)
                continue
            raise FileNotFoundError(message)

        for target in targets:
            layout_path = (args.layout_dir / f"target_task{target}.txt").expanduser().resolve()
            if not layout_path.is_file():
                message = f"Missing target layout: {layout_path}"
                if args.continue_on_error:
                    print("[ERROR]", message)
                    failures += 1
                    continue
                raise FileNotFoundError(message)

            case_root = run_root / f"target_task{target}" / f"seed{seed}"
            policy_checkpoint = case_root / "policy.ckpt"
            policy_artifacts = _policy_artifact_paths(policy_checkpoint)
            eval_csv = case_root / "evaluation" / "ppo_real_env_test.csv"
            training_common = _common_overrides(
                seed=seed,
                target=target,
                layout_path=layout_path,
                wm_checkpoint=wm_checkpoint,
                policy_checkpoint=policy_checkpoint,
                output_root=case_root,
                train_timesteps=args.train_timesteps,
                test_episodes=args.test_episodes,
                test_envs=args.test_envs,
                use_wandb=args.use_wandb,
            )
            train_log = case_root / "training.log"
            eval_log = case_root / "evaluation.log"

            try:
                if args.eval_only:
                    selected_policy = _select_evaluation_policy(policy_artifacts)
                    if not selected_policy.is_file():
                        raise FileNotFoundError(
                            f"Policy checkpoint not found: {policy_artifacts['best']} "
                            f"or legacy {policy_checkpoint}"
                        )
                    print(f"[EVAL ONLY] Using policy: {selected_policy}")
                elif args.skip_existing and _select_evaluation_policy(policy_artifacts).is_file():
                    selected_policy = _select_evaluation_policy(policy_artifacts)
                    print(f"[SKIP] Existing policy: {selected_policy}")
                else:
                    _run_command(
                        [args.python, str(TRAINING_ENTRY), *training_common],
                        env,
                        train_log,
                        args.dry_run,
                    )

                selected_policy = (
                    policy_artifacts["best"]
                    if not args.eval_only
                    and not args.skip_existing
                    and args.dry_run
                    else _select_evaluation_policy(policy_artifacts)
                )
                # New WM-PPO runs must have best.ckpt; canonical fallback keeps
                # policy.ckpt-only historical batches evaluable.
                if not args.dry_run and not selected_policy.is_file():
                    raise FileNotFoundError(
                        f"Selected policy was not created: {selected_policy}"
                    )
                evaluation_common = _common_overrides(
                    seed=seed,
                    target=target,
                    layout_path=layout_path,
                    wm_checkpoint=wm_checkpoint,
                    policy_checkpoint=selected_policy,
                    output_root=case_root,
                    train_timesteps=args.train_timesteps,
                    test_episodes=args.test_episodes,
                    test_envs=args.test_envs,
                    use_wandb=args.use_wandb,
                    test_deterministic=True,
                )

                _run_command(
                    [args.python, str(EVAL_ENTRY), *evaluation_common],
                    env,
                    eval_log,
                    args.dry_run,
                )
                if not args.dry_run:
                    metrics = _read_evaluation_summary(eval_csv)
                    _upsert_summary_row(
                        summary_csv,
                        {
                            "baseline": args.baseline,
                            "seed": seed,
                            "target": f"target_task{target}",
                            "wm_checkpoint": str(wm_checkpoint),
                            "policy_checkpoint": str(policy_checkpoint),
                            "policy_selected_path": str(selected_policy),
                            "policy_best_path": str(policy_artifacts["best"]),
                            "policy_last_path": str(policy_artifacts["last"]),
                            "policy_selection_manifest": str(policy_artifacts["manifest"]),
                            "evaluation_mode": "deterministic",
                            "evaluation_csv": str(eval_csv),
                            **metrics,
                        },
                    )
                    print(
                        f"[DONE] seed={seed} target=target_task{target} "
                        f"reward={metrics['mean_reward']:.5f} "
                        f"success_rate={metrics['success_rate']:.3f}"
                    )
            except Exception as exc:
                failures += 1
                print(f"[ERROR] seed={seed} target=target_task{target}: {exc}")
                if not args.continue_on_error:
                    raise

    if args.dry_run:
        print(f"[{args.baseline} policy batch] Dry run complete; no training/evaluation was started.")
    else:
        print(f"[{args.baseline} policy batch] Summary CSV: {summary_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
