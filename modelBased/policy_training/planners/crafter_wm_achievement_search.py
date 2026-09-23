"""Training-free achievement search through a frozen Crafter world model.

Crafter has no single geometric goal for an admissible A* heuristic.  This
module therefore uses a batched, diversity-preserving beam search over all 17
native actions.  It reports a lower bound on the achievements reachable in the
learned model and can optionally replay the discovered open-loop plan in the
real custom environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import ACHIEVEMENT_NAMES, native_reward_batch
from modelBased.world_model.crafter_dynamics import (
    crafter_player_counts,
    imagined_crafter_step_batch,
)
from domain.crafter.crafter_support import load_crafter_planning_model


WM_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = WM_ROOT.parent
ACTION_NAMES = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)


@dataclass(frozen=True)
class _Candidate:
    index: int
    actions: tuple[int, ...]
    score: tuple[float, ...]
    mask: int
    digest: bytes
    alive: bool


def _entity_hp(states: torch.Tensor) -> torch.Tensor:
    obj = states[:, 0]
    hp = torch.full_like(obj, -1.0)
    hp = torch.where(obj == 14, torch.full_like(hp, 3.0), hp)
    hp = torch.where(obj == 15, torch.full_like(hp, 5.0), hp)
    hp = torch.where(obj == 16, torch.full_like(hp, 3.0), hp)
    return hp


def _new_life_state(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "hunger": torch.zeros(batch, device=device),
        "thirst": torch.zeros(batch, device=device),
        "fatigue": torch.zeros(batch, device=device),
        "recover": torch.zeros(batch, device=device),
        "sleeping": torch.zeros(batch, device=device, dtype=torch.bool),
    }


def _achievement_mask(bits: np.ndarray) -> int:
    result = 0
    for index in np.flatnonzero(bits):
        result |= 1 << int(index)
    return result


def _digest_state(
    state: np.ndarray,
    inventory: np.ndarray,
    unlocked: np.ndarray,
    life: np.ndarray,
    entity_hp: np.ndarray,
) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.rint(state).astype(np.uint8, copy=False).tobytes())
    digest.update(np.rint(inventory).astype(np.int8, copy=False).tobytes())
    digest.update(np.packbits(unlocked.astype(np.uint8, copy=False)).tobytes())
    digest.update(np.rint(life * 2.0).astype(np.int16, copy=False).tobytes())
    digest.update(np.rint(entity_hp).astype(np.int8, copy=False).tobytes())
    return digest.digest()


def _diverse_beam(candidates: list[_Candidate], beam_width: int) -> list[_Candidate]:
    """Round-robin achievement masks so one early event cannot fill the beam."""
    groups: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        if candidate.alive:
            groups.setdefault(candidate.mask, []).append(candidate)
    for values in groups.values():
        values.sort(key=lambda item: item.score, reverse=True)
    ordered_groups = sorted(
        groups.values(), key=lambda values: values[0].score, reverse=True
    )
    selected: list[_Candidate] = []
    rank = 0
    while len(selected) < beam_width:
        added = False
        for values in ordered_groups:
            if rank < len(values):
                selected.append(values[rank])
                added = True
                if len(selected) == beam_width:
                    break
        if not added:
            break
        rank += 1
    return selected


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def search_achievements(
    model,
    spec,
    initial_state: torch.Tensor,
    initial_inventory: torch.Tensor,
    *,
    beam_width: int,
    max_depth: int,
    progress_every: int,
) -> dict:
    device = initial_state.device
    states = initial_state.unsqueeze(0)
    inventories = initial_inventory.unsqueeze(0)
    unlocked = torch.zeros((1, len(ACHIEVEMENT_NAMES)), device=device, dtype=torch.bool)
    life_state = _new_life_state(1, device)
    entity_hp = _entity_hp(states)
    cumulative = np.zeros(1, dtype=np.float64)
    action_paths: list[tuple[int, ...]] = [tuple()]
    initial_objects = initial_state[0].detach().cpu().numpy().astype(np.uint8)

    initial_life = np.zeros(5, dtype=np.float32)
    initial_digest = _digest_state(
        initial_state.detach().cpu().numpy(),
        initial_inventory.detach().cpu().numpy(),
        np.zeros(len(ACHIEVEMENT_NAMES), dtype=bool),
        initial_life,
        entity_hp[0].detach().cpu().numpy(),
    )
    visited: dict[bytes, tuple[float, ...]] = {initial_digest: (0.0, 0.0, 0.0, 0.0, 0.0)}
    best_actions: tuple[int, ...] = tuple()
    best_score: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_achievements: list[str] = []
    expanded_transitions = 0
    depth_reached = 0
    started = time.perf_counter()

    for depth in range(1, max_depth + 1):
        batch = states.shape[0]
        parent = torch.arange(batch, device=device).repeat_interleave(len(ACTION_NAMES))
        actions = torch.arange(len(ACTION_NAMES), device=device).repeat(batch)
        current_states = states[parent]
        current_inventories = inventories[parent]
        current_unlocked = unlocked[parent]
        current_life = {name: value[parent] for name, value in life_state.items()}
        current_hp = entity_hp[parent]

        next_states, next_inventories = imagined_crafter_step_batch(
            model,
            current_states,
            actions,
            current_inventories,
            spec.attention_mask_size,
            spec.inventory_output_mode,
            predict_survival=spec.predict_survival,
            inventory_value_mode=spec.inventory_value_mode,
        )
        rewards, _, next_unlocked, next_life, next_hp, next_inventories = native_reward_batch(
            current_states,
            current_inventories,
            actions,
            next_states,
            next_inventories,
            current_unlocked,
            current_life["sleeping"],
            current_hp,
            current_life,
        )
        expanded_transitions += int(actions.numel())

        finite = torch.isfinite(next_states).flatten(1).all(dim=1)
        finite &= torch.isfinite(next_inventories).all(dim=1)
        valid_player = crafter_player_counts(next_states).eq(1)
        structurally_valid = finite & valid_player

        states_np = next_states.detach().cpu().numpy()
        inventories_np = next_inventories.detach().cpu().numpy()
        unlocked_np = next_unlocked.detach().cpu().numpy()
        hp_np = next_hp.detach().cpu().numpy()
        life_np = np.stack(
            [
                next_life["hunger"].detach().cpu().numpy(),
                next_life["thirst"].detach().cpu().numpy(),
                next_life["fatigue"].detach().cpu().numpy(),
                next_life["recover"].detach().cpu().numpy(),
                next_life["sleeping"].detach().cpu().numpy().astype(np.float32),
            ],
            axis=1,
        )
        rewards_np = rewards.detach().cpu().numpy()
        parent_np = parent.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        valid_np = structurally_valid.detach().cpu().numpy()

        unique: dict[bytes, _Candidate] = {}
        for index in np.flatnonzero(valid_np):
            parent_index = int(parent_np[index])
            action = int(actions_np[index])
            total_reward = float(cumulative[parent_index] + rewards_np[index])
            achievement_count = int(unlocked_np[index].sum())
            core = inventories_np[index, 4:]
            inventory_types = int((core > 0.5).sum())
            inventory_total = float(core.sum())
            objects = np.rint(states_np[index, 0]).astype(np.uint8, copy=False)
            comparable = (objects != 13) & (initial_objects != 13)
            map_edits = int(np.count_nonzero(objects[comparable] != initial_objects[comparable]))
            digest = _digest_state(
                states_np[index], inventories_np[index], unlocked_np[index],
                life_np[index], hp_np[index],
            )
            # The final digest component only breaks otherwise exact ties and
            # avoids a systematic preference for low-numbered actions.
            tie_break = int.from_bytes(digest[:8], "little") / float(2**64)
            score = (
                float(achievement_count), total_reward, float(inventory_types),
                inventory_total, float(map_edits), tie_break,
            )
            candidate = _Candidate(
                index=index,
                actions=action_paths[parent_index] + (action,),
                score=score,
                mask=_achievement_mask(unlocked_np[index]),
                digest=digest,
                alive=bool(inventories_np[index, 0] > 0.5),
            )
            previous = unique.get(digest)
            if previous is None or candidate.score > previous.score:
                unique[digest] = candidate

            if candidate.score > best_score:
                best_score = candidate.score
                best_actions = candidate.actions
                best_achievements = [
                    name for name, present in zip(ACHIEVEMENT_NAMES, unlocked_np[index])
                    if bool(present)
                ]

        novel: list[_Candidate] = []
        for candidate in unique.values():
            state_quality = candidate.score[:5]
            previous_quality = visited.get(candidate.digest)
            if previous_quality is not None and previous_quality >= state_quality:
                continue
            visited[candidate.digest] = state_quality
            novel.append(candidate)

        selected = _diverse_beam(novel, beam_width)
        depth_reached = depth
        if progress_every > 0 and (
            depth == 1 or depth % progress_every == 0 or not selected
        ):
            print(
                f"[Crafter WM search] depth={depth}/{max_depth} "
                f"frontier={len(selected)} visited={len(visited)} "
                f"predicted_achievements={len(best_achievements)} "
                f"transitions={expanded_transitions}",
                flush=True,
            )
        if not selected:
            break

        indices = torch.as_tensor(
            [candidate.index for candidate in selected], device=device, dtype=torch.long
        )
        states = next_states[indices]
        inventories = next_inventories[indices]
        unlocked = next_unlocked[indices]
        entity_hp = next_hp[indices]
        life_state = {name: value[indices] for name, value in next_life.items()}
        cumulative = np.asarray([candidate.score[1] for candidate in selected], dtype=np.float64)
        action_paths = [candidate.actions for candidate in selected]

    return {
        "actions": list(best_actions),
        "predicted_achievements": best_achievements,
        "predicted_unique_achievement_count": len(best_achievements),
        "search_score": list(best_score[:5]),
        "depth_reached": depth_reached,
        "expanded_transitions": expanded_transitions,
        "visited_states": len(visited),
        "search_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def replay_in_world_model(model, spec, state, inventory, actions: list[int]) -> dict:
    states = state.unsqueeze(0)
    inventories = inventory.unsqueeze(0)
    unlocked = torch.zeros((1, len(ACHIEVEMENT_NAMES)), device=state.device, dtype=torch.bool)
    life_state = _new_life_state(1, state.device)
    entity_hp = _entity_hp(states)
    total_reward = 0.0
    events = []
    valid = True
    for step, action_value in enumerate(actions, start=1):
        action = torch.tensor([action_value], device=state.device)
        next_states, next_inventories = imagined_crafter_step_batch(
            model, states, action, inventories, spec.attention_mask_size,
            spec.inventory_output_mode, predict_survival=spec.predict_survival,
            inventory_value_mode=spec.inventory_value_mode,
        )
        reward, newly, unlocked, life_state, entity_hp, next_inventories = native_reward_batch(
            states, inventories, action, next_states, next_inventories, unlocked,
            life_state["sleeping"], entity_hp, life_state,
        )
        names = [
            name for name, active in zip(ACHIEVEMENT_NAMES, newly[0].tolist()) if active
        ]
        if names:
            events.append({"step": step, "action": ACTION_NAMES[action_value], "new": names})
        total_reward += float(reward.item())
        states, inventories = next_states, next_inventories
        if int(crafter_player_counts(states)[0]) != 1 or float(inventories[0, 0]) <= 0:
            valid = False
            break
    names = [name for name, active in zip(ACHIEVEMENT_NAMES, unlocked[0].tolist()) if active]
    return {
        "valid": valid,
        "steps": step if actions else 0,
        "native_return": total_reward,
        "unique_achievement_count": len(names),
        "achievements": names,
        "events": events,
        "final_inventory": inventories[0].detach().cpu().tolist(),
    }


def replay_in_real_environment(layout: Path, seed: int, actions: list[int]) -> dict:
    env = CustomCrafterEnv(txt_file_path=str(layout), max_steps=max(len(actions), 1), seed=seed)
    env.reset()
    total_reward = 0.0
    unlocked: set[str] = set()
    events = []
    steps = 0
    try:
        for steps, action in enumerate(actions, start=1):
            _, reward, terminated, truncated, info = env.step(action)
            names = [str(name) for name in info.get("newly_unlocked", [])]
            if names:
                events.append({"step": steps, "action": ACTION_NAMES[action], "new": names})
                unlocked.update(names)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "steps": steps,
        "native_return": total_reward,
        "unique_achievement_count": len(unlocked),
        "achievements": sorted(unlocked),
        "events": events,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=WM_ROOT / "modelBased/models/AttentionWM/dr_attention_world_model_crafter_none_seed0_best.ckpt",
    )
    parser.add_argument(
        "--layout", type=Path,
        default=REPOSITORY_ROOT / "trainer/level/crafter/target_tasks/crafter_target_task_5.txt",
    )
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--real-replay", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    layout = args.layout.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Crafter WM checkpoint not found: {checkpoint}")
    if not layout.is_file():
        raise FileNotFoundError(f"Crafter layout not found: {layout}")
    if args.beam_width < 1 or args.max_depth < 1:
        raise ValueError("--beam-width and --max-depth must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, spec = load_crafter_planning_model(checkpoint)
    model.to(device).eval()
    env = CustomCrafterEnv(txt_file_path=str(layout), max_steps=args.max_depth, seed=args.seed)
    observation, _ = env.reset()
    env.close()
    initial_state = torch.as_tensor(
        np.transpose(observation["image"], (2, 0, 1)), device=device, dtype=torch.float32
    )
    initial_inventory = torch.as_tensor(
        observation["inventory"], device=device, dtype=torch.float32
    )

    search = search_achievements(
        model, spec, initial_state, initial_inventory,
        beam_width=args.beam_width, max_depth=args.max_depth,
        progress_every=args.progress_every,
    )
    wm_replay = replay_in_world_model(
        model, spec, initial_state, initial_inventory, search["actions"]
    )
    report = {
        "method": "batched_diverse_beam_search",
        "interpretation": "lower_bound_not_optimality_proof",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "layout": str(layout),
        "seed": args.seed,
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "action_names": [ACTION_NAMES[action] for action in search["actions"]],
        **search,
        "wm_replay": wm_replay,
    }
    if args.real_replay:
        report["real_open_loop_replay"] = replay_in_real_environment(
            layout, args.seed, search["actions"]
        )
    output = args.output
    if output is None:
        output = (
            WM_ROOT / "outputs/planning/crafter" /
            f"{checkpoint.stem}_{layout.stem}_achievement_search.json"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[Crafter WM search] predicted achievements: "
        f"{wm_replay['unique_achievement_count']}/{len(ACHIEVEMENT_NAMES)}"
    )
    print(f"[Crafter WM search] names: {wm_replay['achievements']}")
    print(f"[Crafter WM search] predicted native return: {wm_replay['native_return']:.3f}")
    if args.real_replay:
        real = report["real_open_loop_replay"]
        print(
            f"[Crafter WM search] real open-loop replay: "
            f"{real['unique_achievement_count']}/{len(ACHIEVEMENT_NAMES)} achievements, "
            f"return={real['native_return']:.3f}"
        )
    print(f"[Crafter WM search] report: {output}")


if __name__ == "__main__":
    main()
