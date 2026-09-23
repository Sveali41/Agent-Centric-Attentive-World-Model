"""Shared MiniGrid dense reward used by imagined and real PPO rollouts."""

from __future__ import annotations

from collections import deque
import heapq
import itertools

import numpy as np
import torch


class DoorTopology:
    """Static room graph whose doors remain boundaries between components."""

    def __init__(
        self,
        component_map: torch.Tensor,
        door_id_map: torch.Tensor,
        door_components: torch.Tensor,
        component_goal_distances: torch.Tensor,
    ) -> None:
        self.component_map = component_map
        self.door_id_map = door_id_map
        self.door_components = door_components
        self.component_goal_distances = component_goal_distances

    @property
    def num_doors(self) -> int:
        return int(self.door_components.shape[0])


def reward_settings(ppo_cfg) -> dict[str, float | bool]:
    """Read the isolated main-dense-reward configuration block."""
    cfg = getattr(ppo_cfg, "main_dense_reward", {})

    def value(name: str, default):
        return getattr(cfg, name, default)

    return {
        "step_penalty": float(value("step_penalty", -0.001)),
        "progress_weight": float(value("progress_weight", 0.05)),
        "progress_clip": float(value("progress_clip", 1.0)),
        "best_progress_weight": float(value("best_progress_weight", 0.15)),
        "best_progress_clip": float(value("best_progress_clip", 1.0)),
        "best_progress_start_fraction": float(
            value("best_progress_start_fraction", 0.25)
        ),
        "progress_milestone_fraction": float(
            value("progress_milestone_fraction", 0.50)
        ),
        "progress_milestone_reward": float(
            value("progress_milestone_reward", 0.5)
        ),
        "goal_region_action_progress_enabled": bool(
            value("goal_region_action_progress_enabled", True)
        ),
        "goal_region_action_progress_weight": float(
            value("goal_region_action_progress_weight", 0.2)
        ),
        "action_distance_max_expansions": int(
            value("action_distance_max_expansions", 5000)
        ),
        "goal_region_reward": float(value("goal_region_reward", 0.5)),
        "goal_region_reward_once": bool(value("goal_region_reward_once", True)),
        "goal_region_key_pickup_reward": float(
            value("goal_region_key_pickup_reward", 0.0)
        ),
        "goal_region_door_open_reward": float(
            value("goal_region_door_open_reward", 0.0)
        ),
        "critical_door_crossing_reward": float(
            value("critical_door_crossing_reward", 0.5)
        ),
        "critical_door_crossing_once": bool(
            value("critical_door_crossing_once", True)
        ),
        "success_reward": float(value("success_reward", 20.0)),
        "lava_reward": float(value("lava_reward", -1.0)),
    }


def build_goal_distance_map(state: torch.Tensor, goal_yx: tuple[int, int]) -> torch.Tensor:
    """Build the static BFS potential used by the main-workspace reward."""
    frame = torch.as_tensor(state).detach().cpu().numpy()
    objects = frame[0]
    height, width = objects.shape
    goal_y, goal_x = map(int, goal_yx)
    traversable = np.isin(objects, (1, 3, 4, 5, 8, 10))
    distances = np.full((height, width), np.inf, dtype=np.float32)
    if not traversable[goal_y, goal_x]:
        raise ValueError("BFS goal cell is not traversable")
    distances[goal_y, goal_x] = 0.0
    frontier = deque([(goal_y, goal_x)])
    while frontier:
        y, x = frontier.popleft()
        next_distance = distances[y, x] + 1.0
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and traversable[next_y, next_x]
                and not np.isfinite(distances[next_y, next_x])
            ):
                distances[next_y, next_x] = next_distance
                frontier.append((next_y, next_x))
    return torch.as_tensor(distances, dtype=torch.float32, device=state.device)


def build_goal_region_mask(state: torch.Tensor, goal_yx: tuple[int, int]) -> torch.Tensor:
    """Flood-fill the room containing goal, using walls and doors as borders."""
    objects = torch.as_tensor(state)[0].detach().cpu().numpy()
    height, width = objects.shape
    goal_y, goal_x = map(int, goal_yx)
    blocked = np.isin(objects, (2, 4))
    if blocked[goal_y, goal_x]:
        raise ValueError("Goal cell cannot be a wall or door")
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[goal_y, goal_x] = True
    frontier = deque([(goal_y, goal_x)])
    while frontier:
        y, x = frontier.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and not blocked[next_y, next_x]
                and not mask[next_y, next_x]
            ):
                mask[next_y, next_x] = True
                frontier.append((next_y, next_x))
    return torch.as_tensor(mask, dtype=torch.bool, device=state.device)


def build_door_topology(state: torch.Tensor, goal_yx: tuple[int, int]) -> DoorTopology:
    """Build static rooms and their door graph from the episode-initial layout."""
    objects = torch.as_tensor(state)[0].detach().cpu().numpy()
    height, width = objects.shape
    goal_y, goal_x = map(int, goal_yx)
    # Doors are deliberately excluded: they connect rooms, but are not rooms.
    room_cells = ~np.isin(objects, (2, 4))
    if not room_cells[goal_y, goal_x]:
        raise ValueError("Goal cell cannot be a wall or door")
    components = np.full((height, width), -1, dtype=np.int64)
    component_count = 0
    for y in range(height):
        for x in range(width):
            if not room_cells[y, x] or components[y, x] >= 0:
                continue
            components[y, x] = component_count
            frontier = deque([(y, x)])
            while frontier:
                cy, cx = frontier.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height and 0 <= nx < width
                        and room_cells[ny, nx] and components[ny, nx] < 0
                    ):
                        components[ny, nx] = component_count
                        frontier.append((ny, nx))
            component_count += 1
    door_id_map = np.full((height, width), -1, dtype=np.int64)
    door_components: list[tuple[int, int]] = []
    for y, x in np.argwhere(objects == 4):
        adjacent = sorted({
            int(components[ny, nx])
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            if 0 <= ny < height and 0 <= nx < width and components[ny, nx] >= 0
        })
        # A physical door must join exactly two rooms to be a crossing reward.
        if len(adjacent) == 2:
            door_id_map[y, x] = len(door_components)
            door_components.append((adjacent[0], adjacent[1]))
    graph = [[] for _ in range(component_count)]
    for left, right in door_components:
        graph[left].append(right)
        graph[right].append(left)
    goal_component = int(components[goal_y, goal_x])
    distances = np.full(component_count, np.inf, dtype=np.float32)
    distances[goal_component] = 0.0
    frontier = deque([goal_component])
    while frontier:
        component = frontier.popleft()
        for neighbor in graph[component]:
            if not np.isfinite(distances[neighbor]):
                distances[neighbor] = distances[component] + 1.0
                frontier.append(neighbor)
    return DoorTopology(
        torch.as_tensor(components, dtype=torch.long, device=state.device),
        torch.as_tensor(door_id_map, dtype=torch.long, device=state.device),
        torch.as_tensor(
            np.asarray(door_components, dtype=np.int64).reshape((len(door_components), 2)),
            dtype=torch.long,
            device=state.device,
        ),
        torch.as_tensor(distances, dtype=torch.float32, device=state.device),
    )


def _agent_positions(states: torch.Tensor) -> torch.Tensor:
    batch, _, _, width = states.shape
    agent = states[:, 0].eq(10).reshape(batch, -1)
    if not bool(agent.any(dim=1).all()):
        raise ValueError("Every MiniGrid state must contain an agent")
    flat = agent.float().argmax(dim=1)
    return torch.stack((flat // width, flat % width), dim=1)


_ACTION_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _goal_from_distance_map(goal_distance_map: torch.Tensor) -> tuple[int, int]:
    goals = torch.nonzero(goal_distance_map.eq(0), as_tuple=False)
    if goals.shape[0] != 1:
        raise ValueError("Goal distance map must contain exactly one zero-distance goal")
    return int(goals[0, 0]), int(goals[0, 1])


def _action_goal_distance(
    state: torch.Tensor,
    goal: tuple[int, int],
    carrying_token: int,
    max_expansions: int,
) -> float:
    """Exact legal action distance, including turn/pickup/toggle/drop."""
    if max_expansions < 1:
        raise ValueError("action_distance_max_expansions must be positive")
    frame = np.rint(state.detach().cpu().numpy()).astype(np.uint8, copy=True)
    _, height, width = frame.shape
    agent = np.argwhere(frame[0] == 10)
    if len(agent) != 1:
        return float("inf")
    row, col = (int(value) for value in agent[0])
    direction = int(frame[2, row, col])
    if not 0 <= direction < 4 or not 0 <= int(carrying_token) <= 6:
        return float("inf")
    frame[0, row, col] = 1
    frame[1, row, col] = 0
    frame[2, row, col] = 0

    goal_row, goal_col = goal
    traversable = np.isin(frame[0], (1, 3, 4, 5, 8))
    if not (
        0 <= goal_row < height
        and 0 <= goal_col < width
        and traversable[goal_row, goal_col]
    ):
        return float("inf")
    heuristic = np.full((height, width), np.inf, dtype=np.float32)
    heuristic[goal_row, goal_col] = 0.0
    static_frontier = deque([(goal_row, goal_col)])
    while static_frontier:
        cell_row, cell_col = static_frontier.popleft()
        next_distance = heuristic[cell_row, cell_col] + 1.0
        for next_row, next_col in (
            (cell_row - 1, cell_col), (cell_row + 1, cell_col),
            (cell_row, cell_col - 1), (cell_row, cell_col + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and traversable[next_row, next_col]
                and not np.isfinite(heuristic[next_row, next_col])
            ):
                heuristic[next_row, next_col] = next_distance
                static_frontier.append((next_row, next_col))
    if not np.isfinite(heuristic[row, col]):
        return float("inf")

    initial = (row, col, direction, int(carrying_token), frame.tobytes())
    counter = itertools.count()
    frontier = [(float(heuristic[row, col]), 0, next(counter), initial)]
    best_cost = {initial: 0}
    expansions = 0
    while frontier:
        _, distance, _, node = heapq.heappop(frontier)
        row, col, direction, carrying, encoded = node
        if distance != best_cost.get(node):
            continue
        if (row, col) == goal:
            return float(distance)
        expansions += 1
        if expansions > max_expansions:
            return float("inf")
        grid = np.frombuffer(encoded, dtype=np.uint8).reshape(3, height, width)

        def add(next_node, action_cost):
            next_row, next_col = next_node[:2]
            if not np.isfinite(heuristic[next_row, next_col]):
                return
            if action_cost < best_cost.get(next_node, float("inf")):
                best_cost[next_node] = action_cost
                heapq.heappush(
                    frontier,
                    (
                        action_cost + float(heuristic[next_row, next_col]),
                        action_cost,
                        next(counter),
                        next_node,
                    ),
                )

        add((row, col, (direction - 1) % 4, carrying, encoded), distance + 1)
        add((row, col, (direction + 1) % 4, carrying, encoded), distance + 1)
        front_row = row + _ACTION_DELTAS[direction][0]
        front_col = col + _ACTION_DELTAS[direction][1]
        if not (0 <= front_row < height and 0 <= front_col < width):
            continue
        front_object = int(grid[0, front_row, front_col])
        front_color = int(grid[1, front_row, front_col])
        front_state = int(grid[2, front_row, front_col])
        if front_object in (1, 3, 8) or (front_object == 4 and front_state == 0):
            add((front_row, front_col, direction, carrying, encoded), distance + 1)
        if front_object == 5 and carrying == 0 and 0 <= front_color < 6:
            updated = bytearray(encoded)
            cell = front_row * width + front_col
            updated[cell] = 1
            updated[height * width + cell] = 0
            updated[2 * height * width + cell] = 0
            add((row, col, direction, front_color + 1, bytes(updated)), distance + 1)
        if front_object == 4:
            can_open = front_state == 1 or (
                front_state == 2 and carrying == front_color + 1
            )
            if can_open:
                updated = bytearray(encoded)
                cell = front_row * width + front_col
                updated[2 * height * width + cell] = 0
                add((row, col, direction, carrying, bytes(updated)), distance + 1)
        if carrying and front_object == 1:
            updated = bytearray(encoded)
            cell = front_row * width + front_col
            updated[cell] = 5
            updated[height * width + cell] = carrying - 1
            updated[2 * height * width + cell] = 0
            add((row, col, direction, 0, bytes(updated)), distance + 1)
    return float("inf")


def _action_goal_distances(
    states: torch.Tensor,
    carrying_tokens: torch.Tensor,
    goal: tuple[int, int],
    max_expansions: int,
) -> torch.Tensor:
    values = [
        _action_goal_distance(states[index], goal, int(carrying_tokens[index]), max_expansions)
        for index in range(states.shape[0])
    ]
    return torch.as_tensor(values, dtype=torch.float32, device=states.device)


def main_dense_rewards(
    previous: torch.Tensor,
    current: torch.Tensor,
    actions: torch.Tensor,
    goal_distance_map: torch.Tensor,
    goal_region_mask: torch.Tensor,
    goal_region_seen: torch.Tensor,
    success: torch.Tensor,
    lava_terminated: torch.Tensor,
    *,
    step_penalty: float = -0.001,
    progress_weight: float = 0.05,
    progress_clip: float = 1.0,
    best_goal_distances: torch.Tensor | None = None,
    initial_goal_distances: torch.Tensor | None = None,
    progress_milestone_seen: torch.Tensor | None = None,
    previous_carrying_tokens: torch.Tensor | None = None,
    current_carrying_tokens: torch.Tensor | None = None,
    best_progress_weight: float = 0.15,
    best_progress_clip: float = 1.0,
    best_progress_start_fraction: float = 0.25,
    progress_milestone_fraction: float = 0.50,
    progress_milestone_reward: float = 0.5,
    goal_region_action_progress_enabled: bool = True,
    goal_region_action_progress_weight: float = 0.2,
    action_distance_max_expansions: int = 5000,
    goal_region_reward: float = 0.5,
    goal_region_reward_once: bool = True,
    goal_region_key_pickup_reward: float = 0.0,
    goal_region_door_open_reward: float = 0.0,
    rewarded_goal_region_key_positions: torch.Tensor | None = None,
    rewarded_goal_region_door_positions: torch.Tensor | None = None,
    door_topology: DoorTopology | None = None,
    pending_door_ids: torch.Tensor | None = None,
    pending_door_origins: torch.Tensor | None = None,
    rewarded_door_crossings: torch.Tensor | None = None,
    critical_door_crossing_reward: float = 0.5,
    critical_door_crossing_once: bool = True,
    success_reward: float = 20.0,
    lava_reward: float = -1.0,
    transition_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute shared staged MiniGrid shaping for imagined and real PPO."""
    if progress_clip <= 0:
        raise ValueError("progress_clip must be positive")
    if best_progress_clip <= 0:
        raise ValueError("best_progress_clip must be positive")
    if not 0.0 <= best_progress_start_fraction < 1.0:
        raise ValueError("best_progress_start_fraction must be in [0, 1)")
    if not 0.0 <= progress_milestone_fraction <= 1.0:
        raise ValueError("progress_milestone_fraction must be in [0, 1]")
    previous_positions = _agent_positions(previous)
    current_positions = _agent_positions(current)
    previous_distance = goal_distance_map[
        previous_positions[:, 0], previous_positions[:, 1]
    ]
    current_distance = goal_distance_map[
        current_positions[:, 0], current_positions[:, 1]
    ]
    finite = torch.isfinite(previous_distance) & torch.isfinite(current_distance)
    progress = torch.where(
        finite,
        previous_distance - current_distance,
        torch.zeros_like(previous_distance),
    ).clamp(-float(progress_clip), float(progress_clip))
    if best_goal_distances is None:
        best_goal_distances = previous_distance.clone()
    if initial_goal_distances is None:
        initial_goal_distances = previous_distance.clone()
    if progress_milestone_seen is None:
        progress_milestone_seen = torch.zeros_like(previous_distance, dtype=torch.bool)
    if transition_valid is None:
        transition_valid = torch.ones_like(previous_distance, dtype=torch.bool)
    expected_position_shape = (previous.shape[0], previous.shape[-2], previous.shape[-1])
    if rewarded_goal_region_key_positions is None:
        rewarded_goal_region_key_positions = torch.zeros(
            expected_position_shape, dtype=torch.bool, device=previous.device
        )
    if rewarded_goal_region_door_positions is None:
        rewarded_goal_region_door_positions = torch.zeros(
            expected_position_shape, dtype=torch.bool, device=previous.device
        )
    if rewarded_goal_region_key_positions.shape != expected_position_shape:
        raise ValueError("rewarded_goal_region_key_positions has incompatible shape")
    if rewarded_goal_region_door_positions.shape != expected_position_shape:
        raise ValueError("rewarded_goal_region_door_positions has incompatible shape")
    if previous_carrying_tokens is None or current_carrying_tokens is None:
        raise ValueError("Dense MiniGrid interaction rewards require inventory tokens")
    if door_topology is not None:
        if pending_door_ids is None or pending_door_origins is None or rewarded_door_crossings is None:
            raise ValueError("Door-crossing reward requires per-environment door state")
        if rewarded_door_crossings.shape != (previous.shape[0], door_topology.num_doors):
            raise ValueError("rewarded_door_crossings has incompatible shape")
    updated_best_goal_distances = best_goal_distances.clone()

    previous_inside = goal_region_mask[
        previous_positions[:, 0], previous_positions[:, 1]
    ]
    goal_region_action_transition = previous_inside.clone()
    if not goal_region_action_progress_enabled:
        goal_region_action_transition = torch.zeros_like(goal_region_action_transition)
    if bool(goal_region_action_transition.any()):
        if previous_carrying_tokens is None or current_carrying_tokens is None:
            raise ValueError("Goal-region action-distance shaping requires inventory tokens")
        indices = goal_region_action_transition.nonzero(as_tuple=False).flatten()
        goal = _goal_from_distance_map(goal_distance_map)
        previous_action_distance = _action_goal_distances(
            previous[indices], previous_carrying_tokens[indices], goal,
            action_distance_max_expansions,
        )
        current_action_distance = _action_goal_distances(
            current[indices], current_carrying_tokens[indices], goal,
            action_distance_max_expansions,
        )
        finite_action = torch.isfinite(previous_action_distance) & torch.isfinite(current_action_distance)
        action_progress = torch.where(
            finite_action, previous_action_distance - current_action_distance,
            torch.zeros_like(previous_action_distance),
        ).clamp(-float(progress_clip), float(progress_clip))
        progress = progress.clone()
        progress[indices] = action_progress

    progress_weight_by_transition = torch.where(
        goal_region_action_transition,
        torch.full_like(progress, float(goal_region_action_progress_weight)),
        torch.full_like(progress, float(progress_weight)),
    )
    progress_reward = progress * progress_weight_by_transition
    rewards = torch.full_like(progress_reward, float(step_penalty)) + progress_reward
    rewards = torch.where(transition_valid.bool(), rewards, torch.zeros_like(rewards))

    current_inside = goal_region_mask[
        current_positions[:, 0], current_positions[:, 1]
    ]

    finite_best = torch.isfinite(best_goal_distances) & torch.isfinite(current_distance)
    new_best_progress = torch.where(
        finite_best,
        (best_goal_distances - current_distance).clamp(0.0, float(best_progress_clip)),
        torch.zeros_like(current_distance),
    )
    finite_initial = (
        torch.isfinite(initial_goal_distances)
        & torch.isfinite(current_distance)
        & (initial_goal_distances > 0.0)
    )
    completed_fraction = torch.where(
        finite_initial,
        ((initial_goal_distances - current_distance) / initial_goal_distances).clamp(0.0, 1.0),
        torch.zeros_like(current_distance),
    )
    best_progress_eligible = (
        finite_initial
        & (completed_fraction >= float(best_progress_start_fraction))
        & ~goal_region_action_transition
        & transition_valid.bool()
        & ~success.bool()
        & ~lava_terminated.bool()
    )
    new_best_reward = torch.where(
        best_progress_eligible,
        new_best_progress * float(best_progress_weight),
        torch.zeros_like(new_best_progress),
    )
    rewards += new_best_reward
    updated_best_goal_distances = torch.where(
        torch.isfinite(current_distance) & transition_valid.bool(),
        torch.minimum(best_goal_distances, current_distance),
        best_goal_distances,
    )
    milestone_crossed = (
        finite_initial
        & (completed_fraction >= float(progress_milestone_fraction))
        & ~progress_milestone_seen
        & transition_valid.bool()
        & ~success.bool()
        & ~lava_terminated.bool()
    )
    updated_milestone_seen = progress_milestone_seen | milestone_crossed
    milestone_reward = milestone_crossed.float() * float(progress_milestone_reward)
    rewards += milestone_reward
    adjacent_move = (previous_positions - current_positions).abs().sum(dim=1).eq(1)
    entered = (
        ~previous_inside
        & current_inside
        & actions.long().eq(2)
        & adjacent_move
        & transition_valid.bool()
        & ~success.bool()
        & ~lava_terminated.bool()
    )
    updated_seen = goal_region_seen.clone()
    if goal_region_reward_once:
        entered &= ~goal_region_seen
        updated_seen |= entered
    region_reward = entered.float() * float(goal_region_reward)
    rewards += region_reward

    # Reward only successful, goal-directed interactions while already inside
    # the goal room. The front cell must lie on a strictly lower optimistic-BFS
    # contour, and its position is rewarded at most once per episode.
    batch_ids = torch.arange(previous.shape[0], device=previous.device)
    directions = previous[
        batch_ids, 2, previous_positions[:, 0], previous_positions[:, 1]
    ].long()
    valid_direction = directions.ge(0) & directions.lt(len(_ACTION_DELTAS))
    deltas = torch.as_tensor(_ACTION_DELTAS, device=previous.device, dtype=torch.long)
    safe_directions = directions.clamp(0, len(_ACTION_DELTAS) - 1)
    front_positions = previous_positions + deltas[safe_directions]
    height, width = previous.shape[-2:]
    valid_front = (
        valid_direction
        & front_positions[:, 0].ge(0) & front_positions[:, 0].lt(height)
        & front_positions[:, 1].ge(0) & front_positions[:, 1].lt(width)
    )
    safe_front_y = front_positions[:, 0].clamp(0, height - 1)
    safe_front_x = front_positions[:, 1].clamp(0, width - 1)
    front_distance = goal_distance_map[safe_front_y, safe_front_x]
    forward_progress = (
        valid_front & torch.isfinite(previous_distance) & torch.isfinite(front_distance)
        & (front_distance < previous_distance)
    )
    usable_interaction = (
        previous_inside & forward_progress & transition_valid.bool()
        & ~success.bool() & ~lava_terminated.bool()
    )
    previous_front_object = previous[batch_ids, 0, safe_front_y, safe_front_x].long()
    current_front_object = current[batch_ids, 0, safe_front_y, safe_front_x].long()
    previous_front_state = previous[batch_ids, 2, safe_front_y, safe_front_x].long()
    current_front_state = current[batch_ids, 2, safe_front_y, safe_front_x].long()
    pickup_succeeded = (
        usable_interaction & actions.long().eq(3) & previous_front_object.eq(5)
        & ~current_front_object.eq(5)
        & previous_carrying_tokens.eq(0) & current_carrying_tokens.ne(previous_carrying_tokens)
    )
    door_opened = (
        usable_interaction & actions.long().eq(4) & previous_front_object.eq(4)
        & previous_front_state.ne(0) & current_front_object.eq(4)
        & current_front_state.eq(0)
    )
    previously_rewarded_key = rewarded_goal_region_key_positions[
        batch_ids, safe_front_y, safe_front_x
    ]
    previously_rewarded_door = rewarded_goal_region_door_positions[
        batch_ids, safe_front_y, safe_front_x
    ]
    goal_region_key_picked_up = pickup_succeeded & ~previously_rewarded_key
    goal_region_door_opened = door_opened & ~previously_rewarded_door
    updated_rewarded_goal_region_key_positions = rewarded_goal_region_key_positions.clone()
    updated_rewarded_goal_region_door_positions = rewarded_goal_region_door_positions.clone()
    updated_rewarded_goal_region_key_positions[
        batch_ids, safe_front_y, safe_front_x
    ] |= goal_region_key_picked_up
    updated_rewarded_goal_region_door_positions[
        batch_ids, safe_front_y, safe_front_x
    ] |= goal_region_door_opened
    goal_region_key_pickup_reward_value = (
        goal_region_key_picked_up.float() * float(goal_region_key_pickup_reward)
    )
    goal_region_door_open_reward_value = (
        goal_region_door_opened.float() * float(goal_region_door_open_reward)
    )
    rewards += goal_region_key_pickup_reward_value + goal_region_door_open_reward_value

    critical_door_crossed = torch.zeros_like(transition_valid, dtype=torch.bool)
    door_crossing_reward = torch.zeros_like(rewards)
    updated_pending_door_ids = pending_door_ids
    updated_pending_door_origins = pending_door_origins
    updated_rewarded_door_crossings = rewarded_door_crossings
    if door_topology is not None:
        previous_components = door_topology.component_map[
            previous_positions[:, 0], previous_positions[:, 1]
        ]
        current_components = door_topology.component_map[
            current_positions[:, 0], current_positions[:, 1]
        ]
        current_door_ids = door_topology.door_id_map[
            current_positions[:, 0], current_positions[:, 1]
        ]
        previous_door_ids = door_topology.door_id_map[
            previous_positions[:, 0], previous_positions[:, 1]
        ]
        adjacent = (previous_positions - current_positions).abs().sum(dim=1).eq(1)
        usable = transition_valid.bool() & ~success.bool() & ~lava_terminated.bool()
        # Completing must be the immediate forward move from the entered door.
        pending = pending_door_ids.ge(0)
        completion = (
            pending & usable & actions.long().eq(2) & adjacent
            & previous_door_ids.eq(pending_door_ids) & current_components.ge(0)
        )
        if bool(completion.any()):
            indices = completion.nonzero(as_tuple=False).flatten()
            door_ids = pending_door_ids[indices]
            origins = pending_door_origins[indices]
            destinations = current_components[indices]
            pairs = door_topology.door_components[door_ids]
            joins_sides = (
                ((pairs[:, 0] == origins) & (pairs[:, 1] == destinations))
                | ((pairs[:, 1] == origins) & (pairs[:, 0] == destinations))
            )
            origin_distance = door_topology.component_goal_distances[origins]
            destination_distance = door_topology.component_goal_distances[destinations]
            progressing = joins_sides & torch.isfinite(origin_distance) & torch.isfinite(destination_distance) & (destination_distance < origin_distance)
            if critical_door_crossing_once:
                progressing &= ~rewarded_door_crossings[indices, door_ids]
            critical_door_crossed[indices] = progressing
        updated_pending_door_ids = torch.full_like(pending_door_ids, -1)
        updated_pending_door_origins = torch.full_like(pending_door_origins, -1)
        entered_door = (
            usable & actions.long().eq(2) & adjacent
            & previous_components.ge(0) & current_door_ids.ge(0)
        )
        updated_pending_door_ids[entered_door] = current_door_ids[entered_door]
        updated_pending_door_origins[entered_door] = previous_components[entered_door]
        updated_rewarded_door_crossings = rewarded_door_crossings.clone()
        if bool(critical_door_crossed.any()):
            indices = critical_door_crossed.nonzero(as_tuple=False).flatten()
            updated_rewarded_door_crossings[indices, pending_door_ids[indices]] = True
        door_crossing_reward = critical_door_crossed.float() * float(critical_door_crossing_reward)
        rewards += door_crossing_reward

    rewards = torch.where(
        success.bool(), torch.full_like(rewards, float(success_reward)), rewards
    )
    rewards = torch.where(
        lava_terminated.bool(), torch.full_like(rewards, float(lava_reward)), rewards
    )
    return rewards, updated_seen, {
        "progress": progress,
        "progress_reward": progress_reward,
        "new_best_progress": new_best_progress,
        "new_best_reward": new_best_reward,
        "best_progress_eligible": best_progress_eligible,
        "completed_fraction": completed_fraction,
        "progress_milestone_crossed": milestone_crossed,
        "progress_milestone_reward": milestone_reward,
        "progress_milestone_seen": updated_milestone_seen,
        "new_best_goal_distance": updated_best_goal_distances,
        "goal_region_action_transition": goal_region_action_transition,
        "previous_distance": previous_distance,
        "current_distance": current_distance,
        "goal_region_entered": entered,
        "goal_region_reward": region_reward,
        "goal_region_key_picked_up": goal_region_key_picked_up,
        "goal_region_key_pickup_reward": goal_region_key_pickup_reward_value,
        "goal_region_door_opened": goal_region_door_opened,
        "goal_region_door_open_reward": goal_region_door_open_reward_value,
        "rewarded_goal_region_key_positions": updated_rewarded_goal_region_key_positions,
        "rewarded_goal_region_door_positions": updated_rewarded_goal_region_door_positions,
        "critical_door_crossed": critical_door_crossed,
        "critical_door_crossing_reward": door_crossing_reward,
        "pending_door_ids": updated_pending_door_ids,
        "pending_door_origins": updated_pending_door_origins,
        "rewarded_door_crossings": updated_rewarded_door_crossings,
    }
