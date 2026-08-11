"""Crafter reward and deterministic belief bookkeeping for imagined rollouts.

The learned model predicts grid and non-survival inventory transitions. In
the no-AI domain, planning exactly tracks the hidden player life counters,
static cow HP, sleeping, and achievement history instead of asking a one-step
WM to infer unobservable simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ACHIEVEMENT_NAMES = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)

INVENTORY_SLOTS = (
    "health", "food", "drink", "energy", "wood", "stone", "coal", "iron",
    "diamond", "sapling", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)

_RESOURCE_FRONT_IDS = {
    1: "collect_drink",
    3: "collect_stone",
    6: "collect_wood",
    8: "collect_coal",
    9: "collect_iron",
    10: "collect_diamond",
}
_RESOURCE_LEAVES = {3: 4, 6: 2, 8: 4, 9: 4, 10: 4}
_PLACE_IDS = {"place_stone": 3, "place_table": 11, "place_furnace": 12, "place_plant": 18}


def _chw(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3-D Crafter grid, got {grid.shape}")
    if grid.shape[0] == 2:
        return grid
    if grid.shape[-1] == 2:
        return np.moveaxis(grid, -1, 0)
    raise ValueError(f"Expected Crafter grid with two channels, got {grid.shape}")


def _front_cell(grid: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    chw = _chw(grid)
    hits = np.argwhere(chw[0] == 13)
    if len(hits) == 0:
        return None
    y, x = hits[0]
    # Direction IDs are the inverse of the custom adapter's DIR_TO_ID map.
    direction = int(chw[1, y, x])
    offsets = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    dy, dx = offsets.get(direction, (0, 0))
    fy, fx = int(y + dy), int(x + dx)
    if 0 <= fy < chw.shape[1] and 0 <= fx < chw.shape[2]:
        return int(chw[0, fy, fx]), (fy, fx)
    return None


def _front_id(grid: np.ndarray) -> int | None:
    front = _front_cell(grid)
    return None if front is None else front[0]


def _action_name(action: int) -> str:
    from crafter import constants
    return str(constants.actions[int(action)])


@dataclass
class CrafterAchievementTracker:
    """Track Crafter achievement history during WM imagined planning."""

    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in ACHIEVEMENT_NAMES})
    unlocked: set[str] = field(default_factory=set)
    sleeping: bool = False
    entity_hp: dict[tuple[int, int], float] = field(default_factory=dict)

    def reset(self) -> None:
        self.counts = {name: 0 for name in ACHIEVEMENT_NAMES}
        self.unlocked.clear()
        self.sleeping = False
        self.entity_hp.clear()

    def update(
        self,
        grid: np.ndarray,
        inventory: np.ndarray,
        action: int,
        next_grid: np.ndarray,
        next_inventory: np.ndarray,
    ) -> dict:
        """Update history from one transition and return reward diagnostics."""
        current = np.asarray(inventory, dtype=np.float32).reshape(-1)
        following = np.asarray(next_inventory, dtype=np.float32).reshape(-1)
        if len(current) < len(INVENTORY_SLOTS) or len(following) < len(INVENTORY_SLOTS):
            raise ValueError("Crafter inventory must contain all 16 slots")
        before = dict(zip(INVENTORY_SLOTS, current[:16]))
        after = dict(zip(INVENTORY_SLOTS, following[:16]))
        name = _action_name(action)
        events: list[str] = []

        # Crafter overrides the selected action while sleeping. A sleep step
        # must not be credited as a resource interaction merely because the
        # player happens to face water/tree/stone.
        effective_name = name
        if self.sleeping:
            if before["energy"] < 9:
                effective_name = "sleep"
            else:
                events.append("wake_up")
                self.sleeping = False

        if effective_name == "do":
            front_cell = _front_cell(grid)
            front = None if front_cell is None else front_cell[0]
            next_front = _front_id(next_grid)
            if front in _RESOURCE_FRONT_IDS:
                candidate = _RESOURCE_FRONT_IDS[front]
                if front == 1:
                    if after["drink"] >= before["drink"]:
                        events.append(candidate)
                elif next_front == _RESOURCE_LEAVES[front] or after[candidate.removeprefix("collect_")] > before[candidate.removeprefix("collect_")]:
                    events.append(candidate)
            elif front == 2 and after["sapling"] > before["sapling"]:
                events.append("collect_sapling")
            elif front == 18 and after["food"] > before["food"]:
                events.append("eat_plant")
            elif front == 14 and after["food"] > before["food"]:
                events.append("eat_cow")
            elif front in (15, 16) and front_cell is not None:
                position = front_cell[1]
                hp = self.entity_hp.setdefault(position, 5.0 if front == 15 else 3.0)
                damage = max(
                    1.0,
                    2.0 if before["wood_sword"] > 0.5 else 1.0,
                    3.0 if before["stone_sword"] > 0.5 else 1.0,
                    5.0 if before["iron_sword"] > 0.5 else 1.0,
                )
                hp -= damage
                self.entity_hp[position] = max(0.0, hp)
                if hp <= 0:
                    events.append("defeat_zombie" if front == 15 else "defeat_skeleton")

        elif effective_name.startswith("make_") and effective_name in self.counts:
            item = effective_name.removeprefix("make_")
            if after.get(item, 0.0) > before.get(item, 0.0):
                events.append(effective_name)

        elif effective_name.startswith("place_") and effective_name in _PLACE_IDS and effective_name in self.counts:
            required_slot = {
                "place_stone": "stone",
                "place_table": "wood",
                "place_furnace": "stone",
                "place_plant": "sapling",
            }[effective_name]
            if after[required_slot] < before[required_slot]:
                events.append(effective_name)

        # Sleeping is not visible in the symbolic grid, so keep this one bit
        # of simulator history in the tracker itself.
        if effective_name == "sleep" and before["energy"] < 9:
            self.sleeping = True
        # Native Crafter wakes the player immediately whenever health drops.
        if after["health"] < before["health"]:
            self.sleeping = False

        newly_unlocked: list[str] = []
        for event in events:
            self.counts[event] += 1
            if event not in self.unlocked:
                self.unlocked.add(event)
                newly_unlocked.append(event)

        health_reward = float((after["health"] - before["health"]) / 10.0)
        achievement_reward = float(bool(newly_unlocked))
        return {
            # Keep a scalar field for legacy callers and expose all events
            # when multiple achievements occur on the same simulator step.
            "event": events[0] if len(events) == 1 else None,
            "events": events,
            "newly_unlocked": newly_unlocked,
            "achievement_reward": achievement_reward,
            "health_reward": health_reward,
            "reward": achievement_reward + health_reward,
            "counts": dict(self.counts),
        }


def native_reward_from_achievements(
    previous_health: float,
    next_health: float,
    previous_achievements: dict[str, int],
    next_achievements: dict[str, int],
    unlocked: set[str],
) -> tuple[float, list[str]]:
    """Compute Crafter's native reward from simulator achievement snapshots."""
    newly_unlocked = [
        name for name, count in next_achievements.items()
        if int(count) > 0 and name not in unlocked
    ]
    unlocked.update(newly_unlocked)
    reward = float((float(next_health) - float(previous_health)) / 10.0)
    reward += float(bool(newly_unlocked))
    return reward, newly_unlocked


def native_reward_batch(
    states,
    inventories,
    actions,
    next_states,
    next_inventories,
    unlocked,
    sleeping,
    cow_hp=None,
    life_state=None,
):
    """Vectorized native reward/event approximation for WM imagined rollouts.

    ``unlocked``, ``sleeping``, ``cow_hp`` and ``life_state`` are
    planner-owned tensors. The WM itself only supplies the two predicted
    observations. ``cow_hp`` is retained as a compatibility argument name;
    when supplied it has shape ``(B, H, W)`` and stores hidden HP for cows,
    zombies and skeletons, with ``-1`` meaning an uninitialized entity cell.
    """
    import torch

    obj = states[:, 0].long()
    next_obj = next_states[:, 0].long()
    batch, height, width = obj.shape
    flat = (obj == 13).reshape(batch, -1).float()
    pos = torch.argmax(flat, dim=1)
    ys, xs = pos // width, pos % width
    dirs = states[:, 1][torch.arange(batch, device=obj.device), ys, xs].long()
    dy = torch.zeros_like(ys)
    dx = torch.zeros_like(xs)
    dy = torch.where(dirs == 1, -torch.ones_like(dy), dy)
    dy = torch.where(dirs == 2, torch.ones_like(dy), dy)
    dx = torch.where(dirs == 3, -torch.ones_like(dx), dx)
    dx = torch.where(dirs == 4, torch.ones_like(dx), dx)
    fy, fx = ys + dy, xs + dx
    valid = (fy >= 0) & (fy < height) & (fx >= 0) & (fx < width)
    fy_safe, fx_safe = fy.clamp(0, height - 1), fx.clamp(0, width - 1)
    front = next_front = obj[torch.arange(batch, device=obj.device), fy_safe, fx_safe]
    next_front = next_obj[torch.arange(batch, device=obj.device), fy_safe, fx_safe]
    front = torch.where(valid, front, torch.zeros_like(front))
    next_front = torch.where(valid, next_front, torch.zeros_like(next_front))

    inv = inventories.float()
    next_inv = next_inventories.float()
    next_cow_hp = cow_hp
    event = torch.zeros((batch, len(ACHIEVEMENT_NAMES)), dtype=torch.bool, device=obj.device)

    if life_state is not None:
        required = {"hunger", "thirst", "fatigue", "recover", "sleeping"}
        missing = required.difference(life_state)
        if missing:
            raise ValueError(f"life_state is missing fields: {sorted(missing)}")
        previous_sleeping = life_state["sleeping"].to(device=obj.device, dtype=torch.bool)
        forced_sleep = previous_sleeping & (inv[:, 3] < 9)
        wake_event = previous_sleeping & (inv[:, 3] >= 9)
        # While sleeping below maximum energy, Crafter ignores the selected
        # action and executes sleep instead.
        effective_actions = torch.where(
            forced_sleep,
            torch.full_like(actions, 6),
            actions,
        )
        sleeping_after_action = forced_sleep | (
            (~previous_sleeping)
            & (effective_actions == 6)
            & (inv[:, 3] < 9)
        )
    else:
        effective_actions = actions
        wake_event = None
        sleeping_after_action = sleeping

    do = effective_actions == 5
    # Material collection. A successful collection changes the tile, except
    # water; inventory deltas cover the probabilistic grass->sapling action.
    resource_conditions = {
        0: do & (front == 8) & ((next_front == 4) | (next_inv[:, 6] > inv[:, 6] + 0.5)),
        1: do & (front == 10) & ((next_front == 4) | (next_inv[:, 8] > inv[:, 8] + 0.5)),
        2: do & (front == 1),
        3: do & (front == 9) & ((next_front == 4) | (next_inv[:, 7] > inv[:, 7] + 0.5)),
        4: do & (front == 2) & (next_inv[:, 9] > inv[:, 9] + 0.5),
        5: do & (front == 3) & ((next_front == 4) | (next_inv[:, 5] > inv[:, 5] + 0.5)),
        6: do & (front == 6) & ((next_front == 2) | (next_inv[:, 4] > inv[:, 4] + 0.5)),
    }
    for index, condition in resource_conditions.items():
        event[:, index] = condition
    # Entity defeat is tracked below from hidden HP. In no-AI mode the dead
    # entity remains in the symbolic grid, so next_front alone is insufficient.
    if cow_hp is None:
        # Legacy fallback. This is less reliable because a WM can miss the
        # food increment even when the underlying cow was killed.
        event[:, 9] = do & (front == 14) & (next_inv[:, 1] > inv[:, 1] + 0.5)
    else:
        if cow_hp.shape != (batch, height, width):
            raise ValueError(
                "entity_hp must have shape (batch, height, width), "
                f"got {tuple(cow_hp.shape)}"
            )
        cow_hp = cow_hp.to(device=obj.device, dtype=torch.float32)
        batch_index = torch.arange(batch, device=obj.device)
        front_hp = cow_hp[batch_index, fy_safe, fx_safe]
        # Initialize native HP lazily for all attackable entities. A tracked
        # dead entity remains at <= 0 even when no-AI leaves its texture.
        entity = (front == 14) | (front == 15) | (front == 16)
        default_hp = torch.where(
            front == 14,
            torch.full_like(front_hp, 3.0),
            torch.where(front == 15, torch.full_like(front_hp, 5.0), torch.full_like(front_hp, 3.0)),
        )
        front_hp = torch.where(entity & (front_hp < 0), default_hp, front_hp)
        damage = torch.ones((batch,), device=obj.device, dtype=torch.float32)
        damage = torch.where(inv[:, 13] > 0.5, torch.full_like(damage, 2.0), damage)
        damage = torch.where(inv[:, 14] > 0.5, torch.full_like(damage, 3.0), damage)
        damage = torch.where(inv[:, 15] > 0.5, torch.full_like(damage, 5.0), damage)
        hit_cow = do & (front == 14)
        hit_zombie = do & (front == 15)
        hit_skeleton = do & (front == 16)
        hit_entity = hit_cow | hit_zombie | hit_skeleton
        killed_cow = hit_cow & (front_hp > 0) & ((front_hp - damage) <= 0)
        updated_hp = torch.where(hit_entity, front_hp - damage, front_hp)
        next_cow_hp = cow_hp.clone()
        next_cow_hp[batch_index, fy_safe, fx_safe] = torch.where(
            entity,
            updated_hp.clamp_min(0.0),
            next_cow_hp[batch_index, fy_safe, fx_safe],
        )
        event[:, 9] = killed_cow
        event[:, 7] = hit_skeleton & (updated_hp <= 0)
        event[:, 8] = hit_zombie & (updated_hp <= 0)
        # With ai_enabled=False, entity.update() never removes a dead object.
        # Every further hit on a dead cow adds food; unlocked gates reward.
        cow_food_gain = hit_cow & ((front_hp - damage) <= 0)
    if cow_hp is None:
        cow_food_gain = event[:, 9]

    next_life_state = None
    if life_state is not None:
        corrected = next_inv.clone()
        survival = inv[:, :4].clone()

        # Known no-AI action effects on survival inventory.
        drink_water = do & (front == 1)
        survival[:, 1] += cow_food_gain.float() * 6.0
        survival[:, 2] += drink_water.float()

        hunger = life_state["hunger"].to(obj.device, torch.float32).clone()
        thirst = life_state["thirst"].to(obj.device, torch.float32).clone()
        fatigue = life_state["fatigue"].to(obj.device, torch.float32).clone()
        recover = life_state["recover"].to(obj.device, torch.float32).clone()
        hunger = torch.where(cow_food_gain, torch.zeros_like(hunger), hunger)
        thirst = torch.where(drink_water, torch.zeros_like(thirst), thirst)

        half = torch.full_like(hunger, 0.5)
        one = torch.ones_like(hunger)
        hunger += torch.where(sleeping_after_action, half, one)
        consume_food = hunger > 25
        hunger = torch.where(consume_food, torch.zeros_like(hunger), hunger)
        survival[:, 1] -= consume_food.float()

        thirst += torch.where(sleeping_after_action, half, one)
        consume_drink = thirst > 20
        thirst = torch.where(consume_drink, torch.zeros_like(thirst), thirst)
        survival[:, 2] -= consume_drink.float()

        fatigue = torch.where(
            sleeping_after_action,
            torch.minimum(fatigue - 1.0, torch.zeros_like(fatigue)),
            fatigue + 1.0,
        )
        restore_energy = fatigue < -10
        consume_energy = fatigue > 30
        fatigue = torch.where(
            restore_energy | consume_energy,
            torch.zeros_like(fatigue),
            fatigue,
        )
        survival[:, 3] += restore_energy.float() - consume_energy.float()

        necessities = (
            (survival[:, 1] > 0)
            & (survival[:, 2] > 0)
            & ((survival[:, 3] > 0) | sleeping_after_action)
        )
        recover += torch.where(
            necessities,
            torch.where(sleeping_after_action, torch.full_like(recover, 2.0), one),
            torch.where(sleeping_after_action, -half, -one),
        )
        gain_health = recover > 25
        lose_health = recover < -15
        recover = torch.where(
            gain_health | lose_health,
            torch.zeros_like(recover),
            recover,
        )
        survival[:, 0] += gain_health.float() - lose_health.float()

        # Moving onto lava immediately sets health to zero. Movement actions
        # use their own direction rather than the player's previous facing.
        move_dy = torch.zeros_like(ys)
        move_dx = torch.zeros_like(xs)
        move_dx = torch.where(effective_actions == 1, -torch.ones_like(move_dx), move_dx)
        move_dx = torch.where(effective_actions == 2, torch.ones_like(move_dx), move_dx)
        move_dy = torch.where(effective_actions == 3, -torch.ones_like(move_dy), move_dy)
        move_dy = torch.where(effective_actions == 4, torch.ones_like(move_dy), move_dy)
        moving = (effective_actions >= 1) & (effective_actions <= 4)
        ty = (ys + move_dy).clamp(0, height - 1)
        tx = (xs + move_dx).clamp(0, width - 1)
        move_target = obj[torch.arange(batch, device=obj.device), ty, tx]
        survival[:, 0] = torch.where(
            moving & (move_target == 7),
            torch.zeros_like(survival[:, 0]),
            survival[:, 0],
        )

        survival = survival.clamp(0.0, 9.0)
        hurt = survival[:, 0] < inv[:, 0]
        next_sleeping_exact = sleeping_after_action & ~hurt
        corrected[:, :4] = survival
        next_inv = corrected
        next_life_state = {
            "hunger": hunger,
            "thirst": thirst,
            "fatigue": fatigue,
            "recover": recover,
            "sleeping": next_sleeping_exact,
        }
    event[:, 10] = do & (front == 18) & (next_inv[:, 1] > inv[:, 1] + 0.5)

    make_map = {
        11: (15, 10), 12: (13, 11), 13: (11, 12),
        14: (16, 13), 15: (14, 14), 16: (12, 15),
    }
    for action_id, (achievement_index, inventory_index) in make_map.items():
        event[:, achievement_index] = (
            (effective_actions == action_id)
            & (next_inv[:, inventory_index] > inv[:, inventory_index] + 0.5)
        )

    place_map = {7: 19, 8: 20, 9: 17, 10: 18}
    place_inv = {7: 5, 8: 4, 9: 5, 10: 9}
    place_obj = {7: 3, 8: 11, 9: 12, 10: 18}
    for action_id, index in place_map.items():
        item_index = place_inv[action_id]
        event[:, index] = (
            (effective_actions == action_id)
            & (next_inv[:, item_index] < inv[:, item_index] - 0.5)
            & (next_front == place_obj[action_id])
        )

    if life_state is not None:
        event[:, 21] = wake_event
        next_sleeping = next_life_state["sleeping"]
    else:
        wake = sleeping & (inv[:, 3] >= 9)
        event[:, 21] = wake
        next_sleeping = (sleeping & ~wake) | (
            (effective_actions == 6) & (inv[:, 3] < 9) & ~wake
        )

    newly_unlocked = event & ~unlocked
    next_unlocked = unlocked | event
    health_reward = (next_inv[:, 0] - inv[:, 0]) / 10.0
    reward = newly_unlocked.any(dim=1).float() + health_reward
    if life_state is not None:
        return (
            reward,
            newly_unlocked,
            next_unlocked,
            next_life_state,
            next_cow_hp,
            next_inv,
        )
    if cow_hp is not None:
        return reward, newly_unlocked, next_unlocked, next_sleeping, next_cow_hp
    return reward, newly_unlocked, next_unlocked, next_sleeping
