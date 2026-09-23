"""Run the same frozen-WM MiniGrid MPC evaluation for every baseline/seed/target.

Defaults: MAC, target, and DR checkpoints at seeds 0..4; target_task0..4;
10 paired real-environment episodes per case. This launches planning only.

From the repository root::

    source .env
    python wm/modelBased/policy_training/runners/run_mpc_target_eval.py --dry-run
    python wm/modelBased/policy_training/runners/run_mpc_target_eval.py

Use --baselines, --seeds, and --targets to run a subset. Completed cases are
skipped on restart; --force reruns them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
WM_ROOT = REPO_ROOT / "wm"
DEFAULT_CHECKPOINT_DIR = WM_ROOT / "modelBased" / "models" / "AttentionWM"
DEFAULT_LAYOUT_DIR = REPO_ROOT / "trainer" / "level" / "minigrid" / "target_task"
DEFAULT_OUTPUT_ROOT = WM_ROOT / "outputs" / "planning" / "mpc_baseline_target_eval"
BASELINES = ("mac", "target", "dr", "p2e")
DEFAULT_BASELINES = ("mac", "target", "dr")


def _parse_list(value: str, name: str, *, integers: bool = False) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} cannot be empty")
    if integers:
        try:
            items = [int(item) for item in items]
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated integers") from exc
        if any(item < 0 for item in items):
            raise ValueError(f"{name} cannot contain negative values")
    return list(dict.fromkeys(items))


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(REPO_ROOT),
        "TRAINER_ROOT": str(REPO_ROOT),
        "WM_ROOT": str(WM_ROOT),
        "TRAINER_PATH": str(REPO_ROOT / "trainer"),
        "ENV_PATH": str(REPO_ROOT / "trainer" / "level"),
        "WORLD_MODEL_PATH": str(WM_ROOT / "modelBased"),
        "MODEL_FPATH": str(WM_ROOT / "modelBased" / "models"),
        "TRAIN_DATASET_PATH": str(WM_ROOT / "modelBased" / "data" / "train_world_model"),
    })
    return env


def _read_results(path: Path, *, episodes: int, eval_seed: int, checkpoint: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"MPC episode results missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} episodes in {path}, found {len(rows)}")
    for index, row in enumerate(rows):
        if (int(row["episode"]) != index or int(row["seed"]) != eval_seed + index
                or Path(row["wm_checkpoint"]).resolve() != checkpoint):
            raise ValueError(f"Episode identity/checkpoint mismatch in {path}, row {index}")
    return rows


def _upsert_csv(path: Path, rows: list[dict], identity: tuple[str, ...]) -> None:
    existing = []
    fieldnames = list(rows[0])
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        for name in existing[0] if existing else ():
            if name not in fieldnames:
                fieldnames.append(name)
    replacement = {tuple(str(row[key]) for key in identity): row for row in rows}
    existing = [
        row for row in existing
        if tuple(str(row[key]) for key in identity) not in replacement
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*existing, *rows])


def _record_case(output_root: Path, case: dict, rows: list[dict[str, str]], case_dir: Path) -> None:
    identity = {key: case[key] for key in ("baseline", "wm_seed", "target")}
    episode_rows = [
        {**identity, "horizon": case["horizon"],
         "execute_steps": case["execute_steps"],
         "max_episode_steps": case["max_episode_steps"], **row,
         "result_dir": str(case_dir)}
        for row in rows
    ]
    _upsert_csv(
        output_root / "mpc_episode_results.csv", episode_rows,
        ("baseline", "wm_seed", "target", "episode"),
    )
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    successful_rows = [row for row in rows if row["success"].lower() == "true"]
    total_planning_seconds = sum(
        float(row["plan_calls"]) * float(row["mean_planning_latency_ms"]) / 1000
        for row in rows
    )
    summary = {
        **identity,
        "eval_seed_start": case["eval_seed_start"],
        "episodes": len(rows),
        "horizon": case["horizon"],
        "execute_steps": case["execute_steps"],
        "max_episode_steps": case["max_episode_steps"],
        "success_rate": sum(row["success"].lower() == "true" for row in rows) / len(rows),
        "mean_environment_steps": mean("environment_steps"),
        "mean_success_environment_steps": (
            sum(float(row["environment_steps"]) for row in successful_rows)
            / len(successful_rows)
            if successful_rows else ""
        ),
        "mean_native_reward": mean("real_reward"),
        "mean_dense_return": mean("dense_return"),
        "mean_early_replans": mean("early_replans"),
        "mean_plan_calls": mean("plan_calls"),
        "total_planning_seconds": total_planning_seconds,
        "mean_planning_seconds_per_episode": total_planning_seconds / len(rows),
        "total_imagined_transitions": sum(int(row["imagined_transitions"]) for row in rows),
        "mean_pose_match_rate": mean("wm_real_pose_match_rate"),
        "mean_inventory_match_rate": mean("wm_real_inventory_match_rate"),
        "wm_checkpoint": case["wm_checkpoint"],
        "result_dir": str(case_dir),
    }
    _upsert_csv(
        output_root / "mpc_case_summary.csv", [summary],
        ("baseline", "wm_seed", "target"),
    )


def _write_comparison_summary(
    output_root: Path,
    baselines: list[str],
    seeds: list[int],
    targets: list[int],
    episodes: int,
    horizon: int,
    execute_steps: int,
) -> Path:
    episode_path = output_root / "mpc_episode_results.csv"
    with episode_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    wanted_targets = {f"target_task{target}" for target in targets}
    rows = [
        row for row in all_rows
        if row["baseline"] in baselines
        and int(row["wm_seed"]) in seeds
        and row["target"] in wanted_targets
    ]
    expected = len(baselines) * len(seeds) * len(targets) * episodes
    if len(rows) != expected:
        raise ValueError(
            f"Cannot build complete comparison: expected {expected} episode rows, "
            f"found {len(rows)} in {episode_path}"
        )

    summary_rows = []
    for baseline in baselines:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        for target_name in [*(f"target_task{target}" for target in targets), "ALL"]:
            group = (
                baseline_rows if target_name == "ALL"
                else [row for row in baseline_rows if row["target"] == target_name]
            )
            successful = [row for row in group if row["success"].lower() == "true"]
            mean = lambda key: sum(float(row[key]) for row in group) / len(group)
            summary_rows.append({
                "baseline": baseline,
                "target": target_name,
                "wm_seeds": len(seeds),
                "episodes": len(group),
                "horizon": horizon,
                "execute_steps": execute_steps,
                "success_rate": len(successful) / len(group),
                "mean_dense_reward": mean("dense_return"),
                "mean_native_reward": mean("real_reward"),
                "mean_environment_steps": mean("environment_steps"),
                "mean_success_environment_steps": (
                    sum(float(row["environment_steps"]) for row in successful)
                    / len(successful)
                    if successful else ""
                ),
            })
    summary_path = output_root / "mpc_baseline_target_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--seeds", default="0,1,2,3,4", help="World-model checkpoint seeds")
    parser.add_argument("--targets", default=",".join(map(str, range(5))))
    parser.add_argument("--episodes", type=int, default=10, help="Real episodes per case")
    parser.add_argument("--max-episode-steps", type=int, default=512, help="MiniGrid episode limit (default: 512)")
    parser.add_argument("--eval-seed", type=int, default=1, help="First environment seed, shared by all cases")
    parser.add_argument("--horizon", type=int, default=16, help="MPC planning horizon (default: 16)")
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed cases")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        baselines = _parse_list(args.baselines, "--baselines")
        seeds = _parse_list(args.seeds, "--seeds", integers=True)
        targets = _parse_list(args.targets, "--targets", integers=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = set(baselines) - set(BASELINES)
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(sorted(unknown))}; choose from {', '.join(BASELINES)}")
    if (args.episodes < 1 or args.eval_seed < 0 or args.horizon < 1
            or args.execute_steps < 1
            or args.max_episode_steps < 1):
        raise SystemExit("--episodes, --horizon, --execute-steps and --max-episode-steps must be positive; --eval-seed must be nonnegative")
    if args.execute_steps > args.horizon:
        raise SystemExit("--execute-steps must be no greater than --horizon")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases = []
    missing = []
    for baseline in baselines:
        for wm_seed in seeds:
            checkpoint = checkpoint_dir / f"{baseline}_attention_world_model_minigrid_none_effect_seed{wm_seed}.ckpt"
            if not checkpoint.is_file():
                missing.append(str(checkpoint))
            for target in targets:
                layout = layout_dir / f"target_task{target}.txt"
                if not layout.is_file():
                    missing.append(str(layout))
                cases.append((baseline, wm_seed, target, checkpoint, layout))
    if missing:
        raise SystemExit("Missing checkpoint(s) or layout(s):\n" + "\n".join(sorted(set(missing))))

    print(f"[MPC batch] {len(cases)} cases; {args.episodes} paired episodes each; output={output_root}", flush=True)
    failures = 0
    for case_number, (baseline, wm_seed, target, checkpoint, layout) in enumerate(cases, 1):
        case_dir = output_root / baseline / f"wm_seed{wm_seed}" / f"target_task{target}"
        case_config = {
            "baseline": baseline,
            "wm_seed": wm_seed,
            "target": f"target_task{target}",
            "wm_checkpoint": str(checkpoint),
            "layout": str(layout),
            "episodes": args.episodes,
            "max_episode_steps": args.max_episode_steps,
            "eval_seed_start": args.eval_seed,
            "horizon": args.horizon,
            "execute_steps": args.execute_steps,
        }
        results_path = case_dir / "wm_mpc_results.csv"
        manifest_path = case_dir / "case_config.json"
        command = [
            args.python, "-u", "-m", "modelBased.policy_training.planners.mpc_planner",
            "domain=minigrid",
            "PPO.wm_control_mode=mpc",
            "PPO.train_in_real_env=false",
            "PPO.use_main_dense_reward=true",
            f"PPO.seed={args.eval_seed}",
            f"domains.minigrid.task_name=target_task{target}",
            f"domains.minigrid.layout_path={layout}",
            f"PPO.env_path={layout}",
            f"PPO.checkpoint_path_wm={checkpoint}",
            f"PPO.mpc.episodes={args.episodes}",
            f"PPO.mpc.horizon={args.horizon}",
            f"PPO.mpc.execute_steps={args.execute_steps}",
            f"PPO.max_ep_len={args.max_episode_steps}",
            "PPO.mpc.print_every_steps=512",
            f"PPO.mpc.output_dir={case_dir}",
            f"hydra.run.dir={case_dir / 'hydra'}",
        ]
        print(f"[{case_number}/{len(cases)}] {baseline} WM seed {wm_seed}, target {target}", flush=True)
        try:
            previous_config = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
            if previous_config is not None:
                previous_config.setdefault("horizon", 16)
            if previous_config is not None and previous_config != case_config and not args.force:
                raise ValueError(f"Settings differ from completed run in {manifest_path}; use another --output-root or --force")
            complete = (
                previous_config == case_config
                and results_path.is_file()
                and (case_dir / "wm_mpc_steps.csv").is_file()
                and (case_dir / "wm_mpc_traces.json").is_file()
            )
            if args.dry_run:
                print("[DRY RUN]", shlex.join(command), flush=True)
                continue
            if complete and not args.force:
                rows = _read_results(results_path, episodes=args.episodes, eval_seed=args.eval_seed, checkpoint=checkpoint)
                print("[SKIP] completed case", flush=True)
            else:
                case_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.unlink(missing_ok=True)
                with (case_dir / "run.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run(command, cwd=WM_ROOT, env=_child_env(), stdout=log, stderr=subprocess.STDOUT, check=False)
                if process.returncode:
                    raise RuntimeError(f"MPC exited {process.returncode}; see {case_dir / 'run.log'}")
                rows = _read_results(results_path, episodes=args.episodes, eval_seed=args.eval_seed, checkpoint=checkpoint)
                manifest_path.write_text(json.dumps(case_config, indent=2) + "\n", encoding="utf-8")
            _record_case(output_root, case_config, rows, case_dir)
            print(f"[DONE] success={sum(row['success'] == 'True' for row in rows)}/{len(rows)}; results={results_path}", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures += 1
            print(f"[ERROR] {baseline} seed {wm_seed} target {target}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return 1
    if not args.dry_run:
        comparison_path = _write_comparison_summary(
            output_root, baselines, seeds, targets, args.episodes,
            args.horizon, args.execute_steps,
        )
        print(f"[MPC batch] case summary: {output_root / 'mpc_case_summary.csv'}", flush=True)
        print(f"[MPC batch] episode results: {output_root / 'mpc_episode_results.csv'}", flush=True)
        print(f"[MPC batch] baseline comparison: {comparison_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
