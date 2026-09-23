import os
import sys
import csv
import json
import math
import time
from pathlib import Path

WM_ROOT = Path(__file__).resolve().parents[3]
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))
# Direct module execution starts below the repository root, so load the shared
# project environment before modelBased.common.utils reads its required paths.
try:
    from dotenv import load_dotenv
    load_dotenv(WM_ROOT.parent / ".env", override=False)
except ImportError:
    pass
os.environ.setdefault("PROJECT_ROOT", str(WM_ROOT.parent))
os.environ.setdefault("WM_ROOT", str(WM_ROOT))
os.environ.setdefault("ENV_PATH", str(WM_ROOT.parent / "level"))
os.environ.setdefault("WORLD_MODEL_PATH", str(WM_ROOT / "modelBased"))
os.environ.setdefault(
    "TRAIN_DATASET_PATH", str(WM_ROOT / "modelBased" / "data" / "train_world_model")
)
os.environ.setdefault("MODEL_FPATH", str(WM_ROOT / "modelBased" / "models"))
os.environ.setdefault("GENERATOR_PATH", str(WM_ROOT.parent / "generator"))
os.environ.setdefault("TRAINER_PATH", str(WM_ROOT.parent / "trainer"))

from modelBased.common.utils import PROJECT_ROOT, WM_OUTPUTS_PATH
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
import heapq
import itertools
import imageio.v2 as imageio
from collections import deque
from omegaconf import DictConfig
import hydra
from datetime import datetime
from modelBased.common import utils
from domain.minigrid import minigrid_support as minigrid_utils
from domain.minigrid.action_codec import COMPACT_ACTION_NAMES, MODEL_ACTION_COUNT
from domain.minigrid.action_codec import carrying_token_from_env, compact_to_native
from domain.minigrid.transition_codec import decode_minigrid_transition
from modelBased.common.artifacts import world_model_checkpoint_path
import modelBased.world_model.AttentionWM_support as AttentionWM_support
import modelBased.world_model.Embedding_support as Embedding_support
import modelBased.world_model.MLP_support as MLP_support
import wandb


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GraphPlanner:
    def __init__(self, model, num_actions, mask_size, valid_values_obj, valid_values_color, valid_values_state):
        self.model = model
        self.num_actions = num_actions
        self.mask_size = mask_size
        self.valid_values_obj = valid_values_obj
        self.valid_values_color = valid_values_color
        self.valid_values_state = valid_values_state
        self.node_states = {}
        self.transitions = {}
        self.last_plan_diagnostics = {}
        self._layout_signature = None
        self._dynamic_positions = None
        self._terminal_failure_positions = frozenset()

    def _set_reference_layout(self, full_obs: np.ndarray):
        """Select planning-relevant cells for a stable semantic state key.

        A learned decoder can make tiny color/static-cell errors during open
        loop rollout. Keying graph nodes by every float in the full map turns
        those irrelevant differences into an exponential number of states.
        Agent pose, inventory, and the initial interactive objects are the
        sufficient planning state for MiniGrid navigation.
        """
        obs = np.asarray(full_obs)
        obj = obs[0].astype(np.int16)
        # Walls/lava define the fixed layout. Interactive objects can change
        # and therefore remain in the semantic key.
        signature = np.stack(
            (obj == 2, obj == 9), axis=0
        ).astype(np.uint8).tobytes()
        if signature == self._layout_signature:
            return
        self._layout_signature = signature
        self._terminal_failure_positions = frozenset(
            tuple(map(int, position)) for position in np.argwhere(obj == 9)
        )
        self._dynamic_positions = np.argwhere(
            np.isin(obj, (4, 5, 6, 7))
        ).astype(np.int64)
        self.node_states.clear()
        self.transitions.clear()

    def _is_terminal_failure_state(self, full_obs: np.ndarray) -> bool:
        """Return whether the predicted agent pose is on native lava."""
        position = minigrid_utils.get_agent_position(np.asarray(full_obs))
        return tuple(map(int, position)) in self._terminal_failure_positions

    def _state_key(self, full_obs: np.ndarray, inventory_token: int = 0) -> bytes:
        obs = np.asarray(full_obs)
        agent_y, agent_x = minigrid_utils.get_agent_position(obs)
        direction = int(obs[2, agent_y, agent_x])
        header = np.asarray(
            [int(inventory_token), agent_y, agent_x, direction],
            dtype=np.int16,
        )
        if self._dynamic_positions is None or not len(self._dynamic_positions):
            return header.tobytes()
        ys = self._dynamic_positions[:, 0]
        xs = self._dynamic_positions[:, 1]
        dynamic_values = obs[:, ys, xs].T.astype(np.int16, copy=False)
        return header.tobytes() + dynamic_values.tobytes()

    def expand_node(self, full_obs: np.ndarray, inventory_token: int = 0):
        key = self._state_key(full_obs, inventory_token)
        if key in self.transitions:
            return self.transitions[key]
        if self.num_actions != MODEL_ACTION_COUNT:
            raise ValueError(
                f"MiniGrid WM planner requires {MODEL_ACTION_COUNT} compact "
                f"actions, got {self.num_actions}"
            )

        model_device = next(self.model.parameters()).device
        full = torch.as_tensor(
            full_obs, device=model_device, dtype=torch.float32
        ).unsqueeze(0)
        full_batch = full.expand(self.num_actions, -1, -1, -1).clone()
        positions = utils.get_agent_position_torch(full_batch)
        masked = utils.extract_masked_state_torch(
            full_batch, self.mask_size, positions
        )
        actions = torch.arange(
            self.num_actions, device=model_device, dtype=torch.long
        )
        inventories = torch.full(
            (self.num_actions,),
            int(inventory_token),
            device=model_device,
            dtype=torch.long,
        )
        with torch.no_grad():
            prediction, _, inventory_logits = self.model(
                masked, actions, None, inv=inventories
            )
            next_masked, next_inventory, _ = decode_minigrid_transition(
                prediction,
                masked,
                inventory_logits,
                inventories,
                mode=getattr(
                    self.model, "minigrid_transition_mode", "effect"
                ),
                constrain_agent=True,
            )
            next_full_batch = utils.put_back_masked_state_torch(
                next_masked, full_batch, self.mask_size, positions
            )

        self.node_states[key] = (full_obs.copy(), int(inventory_token))
        successors = []
        seen_successors = set()
        for action in range(self.num_actions):
            next_full = next_full_batch[action].detach().cpu().numpy()
            next_inventory_token = (
                int(next_inventory[action].item())
                if next_inventory is not None
                else int(inventory_token)
            )
            # MiniGrid permits the transition onto lava and then terminates.
            # The decoded observation contains the agent token at that cell,
            # so use the immutable initial layout to reject terminal failures.
            if self._is_terminal_failure_state(next_full):
                continue
            next_key = self._state_key(next_full, next_inventory_token)
            # Useless interactions and blocked forward actions are self-loops;
            # retaining them only multiplies the search space.
            if next_key == key or next_key in seen_successors:
                continue
            seen_successors.add(next_key)
            self.node_states.setdefault(
                next_key, (next_full.copy(), next_inventory_token)
            )
            successors.append((next_key, int(action)))
        self.transitions[key] = successors
        return successors

    def predict_transition(
        self,
        full_obs: np.ndarray,
        inventory_token: int,
        action: int,
    ) -> tuple[np.ndarray, int]:
        """Return the decoded one-step successor used by the search graph."""
        key = self._state_key(full_obs, inventory_token)
        successors = self.expand_node(full_obs, inventory_token)
        for next_key, successor_action in successors:
            if int(successor_action) == int(action):
                next_state, next_inventory = self.node_states[next_key]
                return next_state.copy(), int(next_inventory)
        # Self-loop actions are intentionally omitted from the search graph.
        return np.asarray(full_obs).copy(), int(inventory_token)

    @staticmethod
    def _goal_distance_map(start_full: np.ndarray, goal_position):
        """Optimistic static distance used as an admissible A* heuristic."""
        obj = np.asarray(start_full)[0]
        rows, cols = obj.shape
        gy, gx = map(int, goal_position)
        distances = np.full((rows, cols), np.inf, dtype=np.float32)
        distances[gy, gx] = 0.0
        queue = deque([(gy, gx)])
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                # Doors, keys and movable objects are optimistically
                # traversable. Only immutable walls and lava are blocked.
                if int(obj[ny, nx]) in (2, 9):
                    continue
                if np.isfinite(distances[ny, nx]):
                    continue
                distances[ny, nx] = distances[y, x] + 1.0
                queue.append((ny, nx))
        return distances

    def plan(
        self,
        start_full: np.ndarray,
        goal_full: np.ndarray,
        k: int = 1,
        inventory_token: int = 0,
        max_expansions: int | None = None,
        progress_every_expansions: int = 0,
    ):
        """A* search for a predicted trajectory whose agent reaches the goal.

        A full-map equality goal is too strict for a learned categorical WM:
        an otherwise correct rollout can differ in an irrelevant wall/color
        cell.  Planning therefore uses the goal agent coordinate as the task
        predicate while retaining the full decoded map and inventory in the
        visited-state key.  This also makes the planner's state contract match
        PPO's decoded MiniGrid state contract.
        """
        if k < 1:
            return []
        self._set_reference_layout(start_full)
        start_key = self._state_key(start_full, inventory_token)
        goal_position = minigrid_utils.get_agent_position(goal_full)
        if goal_position is None:
            return []
        self.node_states[start_key] = (start_full.copy(), int(inventory_token))
        if self._is_terminal_failure_state(start_full):
            self.last_plan_diagnostics = {
                "found": False,
                "path_length": 0,
                "expansions": 0,
                "visited_states": 1,
                "source": "world_model_astar",
                "failure_reason": "terminal_lava",
            }
            return []
        distance_map = self._goal_distance_map(start_full, goal_position)

        def heuristic(node_key):
            state, _ = self.node_states[node_key]
            y, x = minigrid_utils.get_agent_position(state)
            value = float(distance_map[int(y), int(x)])
            if not np.isfinite(value):
                return float(
                    abs(int(y) - int(goal_position[0]))
                    + abs(int(x) - int(goal_position[1]))
                )
            return value

        counter = itertools.count()
        queue = [(heuristic(start_key), 0, next(counter), start_key)]
        best_cost = {start_key: 0}
        came_from = {}
        limit = int(max_expansions or max(5000, 100 * k))
        expansions = 0

        while queue and expansions < limit:
            _, path_cost, _, node_key = heapq.heappop(queue)
            if path_cost != best_cost.get(node_key):
                continue
            node_state, node_inventory = self.node_states[node_key]
            if minigrid_utils.get_agent_position(node_state) == goal_position:
                actions = []
                current = node_key
                while current != start_key:
                    previous, action = came_from[current]
                    actions.append(int(action))
                    current = previous
                actions.reverse()
                self.last_plan_diagnostics = {
                    "found": True,
                    "path_length": len(actions),
                    "expansions": expansions,
                    "visited_states": len(best_cost),
                    "source": "world_model_astar",
                }
                return actions
            if path_cost >= k:
                continue
            expansions += 1
            if (
                progress_every_expansions > 0
                and expansions % int(progress_every_expansions) == 0
            ):
                print(
                    f"[WM A* Search] expansions={expansions}/{limit} "
                    f"visited={len(best_cost)} depth={path_cost}"
                )
            for next_key, next_action in self.expand_node(
                node_state, node_inventory
            ):
                next_cost = path_cost + 1
                if next_cost >= best_cost.get(next_key, float("inf")):
                    continue
                best_cost[next_key] = next_cost
                came_from[next_key] = (node_key, int(next_action))
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic(next_key),
                        next_cost,
                        next(counter),
                        next_key,
                    ),
                )
        self.last_plan_diagnostics = {
            "found": False,
            "path_length": 0,
            "expansions": expansions,
            "visited_states": len(best_cost),
            "source": "world_model_astar",
        }
        return []


# MiniGrid uses x/y coordinates internally while this project stores policy
# observations as C/H/W (therefore y/x).  Keeping the exact symbolic planner
# here gives learned-model plans an executable oracle: a WM shortcut is never
# allowed to become a behavior-cloning target merely because the decoder
# hallucinated an open wall or an impossible agent move.
_DIRECTION_XY = ((1, 0), (0, 1), (-1, 0), (0, -1))
_EMPTY_CELL = np.asarray((1, 0, 0), dtype=np.uint8)


def _symbolic_state_key(grid, pose_inventory):
    return grid.tobytes() + np.asarray(pose_inventory, dtype=np.int16).tobytes()


def _symbolic_transition(grid, pose_inventory, action):
    """Apply one standard MiniGrid transition to an encoded full grid.

    The represented state is ``(x, y, direction, carried_obj, color, state)``.
    It covers all transitions required by the project's key/door navigation
    tasks.  A box with hidden contents is intentionally rejected because its
    contents are not represented in a three-channel observation.
    """
    x, y, direction, carry_obj, carry_color, carry_state = map(
        int, pose_inventory
    )
    dx, dy = _DIRECTION_XY[direction]
    front_x, front_y = x + dx, y + dy
    if not (
        0 <= front_x < grid.shape[0]
        and 0 <= front_y < grid.shape[1]
    ):
        return grid, pose_inventory
    obj, color, state = map(int, grid[front_x, front_y])
    next_grid = grid

    if action == 0:  # left
        direction = (direction - 1) % 4
    elif action == 1:  # right
        direction = (direction + 1) % 4
    elif action == 2:  # forward
        if obj in (1, 3, 8) or (obj == 4 and state == 0):
            x, y = front_x, front_y
    elif action == 3:  # pickup
        if carry_obj < 0 and obj in (5, 6, 7):
            next_grid = grid.copy()
            next_grid[front_x, front_y] = _EMPTY_CELL
            carry_obj, carry_color, carry_state = obj, color, state
    elif action == 4:  # toggle
        if obj == 7:
            raise ValueError(
                "Symbolic planning cannot infer a box's hidden contents from "
                "a fully observed three-channel grid"
            )
        if obj == 4:
            next_state = None
            if state == 2 and carry_obj == 5 and carry_color == color:
                next_state = 0
            elif state == 1:
                next_state = 0
            elif state == 0:
                next_state = 1
            if next_state is not None:
                next_grid = grid.copy()
                next_grid[front_x, front_y, 2] = next_state
    elif action == 5:  # drop
        if carry_obj >= 0 and obj == 1:
            next_grid = grid.copy()
            next_grid[front_x, front_y] = (
                carry_obj,
                carry_color,
                carry_state,
            )
            carry_obj = carry_color = carry_state = -1
    else:
        raise ValueError(
            f"Expected compact MiniGrid action 0..{MODEL_ACTION_COUNT - 1}, "
            f"got {action}"
        )

    next_pose_inventory = (
        x,
        y,
        direction,
        carry_obj,
        carry_color,
        carry_state,
    )
    return next_grid, next_pose_inventory


def _symbolic_distance_map(grid, goal_xy):
    """Optimistic static grid distance for exact-state A*."""
    width, height = grid.shape[:2]
    distances = np.full((width, height), width * height + 1, dtype=np.int32)
    distances[goal_xy] = 0
    queue = deque([goal_xy])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _DIRECTION_XY:
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if int(grid[next_x, next_y, 0]) in (2, 9):
                continue
            if distances[next_x, next_y] <= distances[x, y] + 1:
                continue
            distances[next_x, next_y] = distances[x, y] + 1
            queue.append((next_x, next_y))
    return distances


def plan_exact_minigrid(env, max_expansions=100000):
    """Return a shortest executable compact-action path for ``env``.

    This is the safety oracle/fallback for learned WM A*.  It uses the same
    six-action compact codec as data collection, the WM and PPO.  Drop is not
    expanded: dropping a carried object is never required by the standard
    reach-goal/key-door objective, and including arbitrary drop locations
    makes the exact state space needlessly unbounded.
    """
    unwrapped = env.unwrapped
    grid = unwrapped.grid.encode()
    goals = np.argwhere(grid[:, :, 0] == 8)
    if not len(goals):
        raise ValueError("MiniGrid layout does not contain a goal")
    goal_xy = tuple(map(int, goals[0]))
    carrying = getattr(unwrapped, "carrying", None)
    if carrying is None:
        carrying_values = (-1, -1, -1)
    else:
        carrying_values = tuple(map(int, carrying.encode()))
    start_pose = (
        int(unwrapped.agent_pos[0]),
        int(unwrapped.agent_pos[1]),
        int(unwrapped.agent_dir),
        *carrying_values,
    )
    start_key = _symbolic_state_key(grid, start_pose)
    distance_map = _symbolic_distance_map(grid, goal_xy)
    states = {start_key: (grid, start_pose)}
    best_cost = {start_key: 0}
    came_from = {}
    counter = itertools.count()
    queue = [
        (
            int(distance_map[start_pose[0], start_pose[1]]),
            0,
            next(counter),
            start_key,
        )
    ]
    expansions = 0

    while queue and expansions < int(max_expansions):
        _, cost, _, state_key = heapq.heappop(queue)
        if cost != best_cost.get(state_key):
            continue
        state_grid, pose = states[state_key]
        if tuple(pose[:2]) == goal_xy:
            actions = []
            current = state_key
            while current != start_key:
                previous, action = came_from[current]
                actions.append(int(action))
                current = previous
            actions.reverse()
            return actions, {
                "found": True,
                "path_length": len(actions),
                "expansions": expansions,
                "visited_states": len(best_cost),
                "source": "exact_minigrid",
            }

        expansions += 1
        x, y, direction, carry_obj, _, _ = pose
        dx, dy = _DIRECTION_XY[direction]
        front_x, front_y = x + dx, y + dy
        front_obj = (
            int(state_grid[front_x, front_y, 0])
            if 0 <= front_x < state_grid.shape[0]
            and 0 <= front_y < state_grid.shape[1]
            else 2
        )
        # Rotations and forward are always considered. Interactions that
        # cannot change the state are excluded before expansion.
        candidate_actions = [0, 1, 2]
        if carry_obj < 0 and front_obj in (5, 6, 7):
            candidate_actions.append(3)
        if front_obj == 4:
            candidate_actions.append(4)

        for action in candidate_actions:
            next_grid, next_pose = _symbolic_transition(
                state_grid, pose, action
            )
            next_key = _symbolic_state_key(next_grid, next_pose)
            if next_key == state_key:
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_key, float("inf")):
                continue
            states[next_key] = (next_grid, next_pose)
            best_cost[next_key] = next_cost
            came_from[next_key] = (state_key, int(action))
            heuristic = int(distance_map[next_pose[0], next_pose[1]])
            heapq.heappush(
                queue,
                (
                    next_cost + heuristic,
                    next_cost,
                    next(counter),
                    next_key,
                ),
            )

    return [], {
        "found": False,
        "path_length": 0,
        "expansions": expansions,
        "visited_states": len(best_cost),
        "source": "exact_minigrid",
    }


def replay_actions_in_real_env(env, actions):
    """Execute a compact-action path and retain BC observations."""
    states = []
    inventories = []
    next_states = []
    next_inventories = []
    rewards = []
    terminated = truncated = False
    for action in actions:
        observation = env.observation(env.unwrapped.gen_obs())["image"]
        states.append(minigrid_utils.ColRowCanl_to_CanlRowCol(observation))
        inventories.append(carrying_token_from_env(env))
        next_observation, reward, terminated, truncated, _ = env.step(
            compact_to_native(action)
        )
        next_states.append(
            minigrid_utils.ColRowCanl_to_CanlRowCol(
                next_observation["image"]
            )
        )
        next_inventories.append(carrying_token_from_env(env))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    reached_goal = bool(terminated and rewards and rewards[-1] > 0.0)
    return {
        "states": states,
        "inventory_tokens": inventories,
        "next_states": next_states,
        "next_inventory_tokens": next_inventories,
        "actions": list(map(int, actions[: len(states)])),
        "rewards": rewards,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_goal": reached_goal,
    }


def replay_actions_in_world_model(
    model,
    initial_state,
    actions,
    attention_mask_size,
    initial_inventory_token=0,
    real_reference=None,
):
    """Open-loop replay used to gate and construct planner demonstrations."""
    model_device = next(model.parameters()).device
    state = torch.as_tensor(
        initial_state, dtype=torch.float32, device=model_device
    ).unsqueeze(0)
    inventory = torch.as_tensor(
        [initial_inventory_token], dtype=torch.long, device=model_device
    )
    goal_position = find_position(np.asarray(initial_state), (8, 1, 0))
    states = []
    inventories = []
    next_states = []
    next_inventories = []
    pose_matches = []
    inventory_matches = []

    for index, action in enumerate(actions):
        states.append(state[0].detach().cpu().numpy())
        inventories.append(int(inventory[0].item()))
        positions = utils.get_agent_position_torch(state)
        masked = utils.extract_masked_state_torch(
            state, attention_mask_size, positions
        )
        action_tensor = torch.as_tensor(
            [action], dtype=torch.long, device=model_device
        )
        with torch.no_grad():
            prediction, _, inventory_logits = model(
                masked, action_tensor, None, inv=inventory
            )
            decoded, next_inventory, _ = decode_minigrid_transition(
                prediction,
                masked,
                inventory_logits,
                inventory,
                mode=getattr(
                    model, "minigrid_transition_mode", "effect"
                ),
                constrain_agent=True,
            )
            state = utils.put_back_masked_state_torch(
                decoded,
                state,
                attention_mask_size,
                positions,
            )
        inventory = next_inventory.long()
        state_numpy = state[0].detach().cpu().numpy()
        next_states.append(state_numpy)
        next_inventories.append(int(inventory[0].item()))

        if real_reference is not None:
            real_next = real_reference["next_states"][index]
            pose_matches.append(
                minigrid_utils.get_agent_position(state_numpy)
                == minigrid_utils.get_agent_position(real_next)
            )
            inventory_matches.append(
                int(inventory[0].item())
                == int(real_reference["next_inventory_tokens"][index])
            )

    final_position = minigrid_utils.get_agent_position(
        next_states[-1] if next_states else initial_state
    )
    reached_goal = goal_position is not None and final_position == goal_position
    return {
        "states": states,
        "inventory_tokens": inventories,
        "next_states": next_states,
        "next_inventory_tokens": next_inventories,
        "actions": list(map(int, actions)),
        "reached_goal": bool(reached_goal),
        "pose_match_rate": (
            float(np.mean(pose_matches)) if pose_matches else None
        ),
        "inventory_match_rate": (
            float(np.mean(inventory_matches)) if inventory_matches else None
        ),
        "all_poses_match": bool(all(pose_matches)) if pose_matches else None,
        "all_inventories_match": (
            bool(all(inventory_matches)) if inventory_matches else None
        ),
    }


def find_position(array, target):
    target = np.asarray(target).reshape(-1, 1, 1)
    matches = np.argwhere((np.asarray(array) == target).all(axis=0))
    return tuple(matches[0]) if matches.size else None


def _checkpoint_attention_mask_size(checkpoint_path: Path) -> int | None:
    """Read the MiniGrid attention window encoded in a WM checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    metadata_size = None
    if isinstance(checkpoint, dict):
        hparams = checkpoint.get("hyper_parameters")
        if hparams is not None:
            value = (
                hparams.get("attention_mask_size")
                if hasattr(hparams, "get")
                else None
            )
            if value is not None:
                metadata_size = int(value)

    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    positional_sizes = set()
    if hasattr(state_dict, "items"):
        for key, value in state_dict.items():
            if (
                str(key).split(".")[-1] != "pos_embedding"
                or not hasattr(value, "shape")
            ):
                continue
            shape = tuple(value.shape)
            if len(shape) != 3:
                continue
            token_count = int(shape[1])
            mask_size = math.isqrt(token_count)
            if mask_size * mask_size != token_count:
                raise RuntimeError(
                    "Checkpoint pos_embedding token count is not a square: "
                    f"key={key}, shape={shape}"
                )
            positional_sizes.add(mask_size)

    if len(positional_sizes) > 1:
        raise RuntimeError(
            "Checkpoint contains conflicting positional embedding sizes: "
            f"{sorted(positional_sizes)}"
        )
    positional_size = next(iter(positional_sizes), None)
    if (
        metadata_size is not None
        and positional_size is not None
        and metadata_size != positional_size
    ):
        raise RuntimeError(
            "Checkpoint attention-mask metadata disagrees with pos_embedding: "
            f"metadata={metadata_size}, pos_embedding={positional_size}"
        )

    inferred = metadata_size if metadata_size is not None else positional_size
    if inferred is not None and (inferred <= 0 or inferred % 2 == 0):
        raise RuntimeError(
            "Checkpoint attention_mask_size must be a positive odd integer, "
            f"got {inferred}"
        )
    return inferred


def _planning_output_dir(checkpoint: Path, configured_output) -> Path:
    if (
        configured_output is not None
        and str(configured_output).strip()
        and str(configured_output).lower() != "null"
    ):
        return Path(str(configured_output)).expanduser().resolve()

    model_prefix = checkpoint.stem.split("_", 1)[0].lower()
    if not model_prefix:
        raise ValueError(
            f"Cannot derive planning result prefix from checkpoint: {checkpoint}"
        )
    return (
        WM_OUTPUTS_PATH / "planning" / f"{model_prefix}_astar_results"
    ).resolve()


def _load_world_model_for_planning(cfg: DictConfig):
    """Load the aligned MiniGrid WM artifact used by policy training."""
    hparams_wm = cfg.attention_model
    if str(hparams_wm.model_type).lower() != "attention":
        raise ValueError(
            "MiniGrid categorical planning requires model_type=Attention; "
            "the shared decoder also consumes the auxiliary inventory head."
        )
    configured = getattr(cfg.PPO, "checkpoint_path_wm", None)
    checkpoint = (
        Path(str(configured)).expanduser().resolve()
        if configured is not None
        and str(configured).strip()
        and str(configured).lower() != "null"
        else world_model_checkpoint_path(cfg, str(cfg.domain))
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"World-model checkpoint not found: {checkpoint}")

    configured_mask_size = int(hparams_wm.attention_mask_size)
    checkpoint_mask_size = _checkpoint_attention_mask_size(checkpoint)
    mask_size = (
        checkpoint_mask_size
        if checkpoint_mask_size is not None
        else configured_mask_size
    )
    source = "checkpoint" if checkpoint_mask_size is not None else "config fallback"
    print(
        f"[WM Planner] attention_mask_size={mask_size} "
        f"(source={source}, configured={configured_mask_size})"
    )

    model = AttentionWM_support.AttentionModule(
        hparams_wm.data_type,
        hparams_wm.grid_shape,
        mask_size,
        hparams_wm.embed_dim,
        hparams_wm.num_heads,
        env_type=hparams_wm.env_type,
        frame_stack=hparams_wm.frame_stack,
        minigrid_transition_mode=getattr(
            hparams_wm, "minigrid_transition_mode", "effect"
        ),
    ).to(device)
    utils.load_model_weight(model, str(checkpoint))
    model.eval()
    return model, checkpoint, mask_size


def _goal_state(initial_state: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Build the coordinate-only goal representation consumed by GraphPlanner."""
    goal_yx = find_position(initial_state, (8, 1, 0))
    if goal_yx is None:
        raise ValueError("Initial MiniGrid observation does not contain a goal")
    goal_state = np.asarray(initial_state).copy()
    agent_yx = minigrid_utils.get_agent_position(goal_state)
    goal_state[:, agent_yx[0], agent_yx[1]] = np.asarray(
        [1, 0, 0], dtype=goal_state.dtype
    )
    goal_state[:, goal_yx[0], goal_yx[1]] = np.asarray(
        [10, 0, 0], dtype=goal_state.dtype
    )
    return goal_state, goal_yx


def _execute_plan_in_real_env(
    *,
    layout_path: str,
    actions: list[int],
    direction: int,
    seed: int,
    max_steps: int,
    gif_path: Path | None,
    gif_fps: int,
) -> dict:
    """Evaluate a completed plan; this is never called during A* search."""
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=layout_path,
            custom_mission="Reach the goal.",
            agent_start_dir=int(direction),
            max_steps=int(max_steps),
            render_mode="rgb_array" if gif_path is not None else None,
        )
    )
    env.reset(seed=int(seed))
    frames = [env.render()] if gif_path is not None else []
    reward_sum = 0.0
    terminated = truncated = False
    executed = 0
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(
            compact_to_native(int(action))
        )
        reward_sum += float(reward)
        executed += 1
        if gif_path is not None:
            frames.append(env.render())
        if terminated or truncated:
            break
    final_position = tuple(map(int, env.unwrapped.agent_pos[::-1]))
    env.close()
    if gif_path is not None and frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=max(1, int(gif_fps)))
    return {
        "real_reached_goal": bool(terminated and reward_sum > 0.0),
        "real_reward": reward_sum,
        "real_executed_steps": executed,
        "real_terminated": bool(terminated),
        "real_truncated": bool(truncated),
        "real_final_y": final_position[0],
        "real_final_x": final_position[1],
    }


def _json_default(value):
    """Convert NumPy and Path values used by planner diagnostics to JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _agent_pose(full_obs: np.ndarray) -> tuple[int, int, int]:
    y, x = minigrid_utils.get_agent_position(full_obs)
    return int(y), int(x), int(np.asarray(full_obs)[2, y, x])


def run_receding_horizon_episode(
    *,
    planner: GraphPlanner,
    layout_path: str,
    direction: int,
    seed: int,
    max_steps: int,
    max_depth: int,
    max_expansions: int,
    progress_every_expansions: int,
    gif_path: Path | None,
    gif_fps: int,
    print_every_steps: int = 1,
) -> tuple[dict, list[dict]]:
    """Execute one real action from each complete WM A* plan, then replan."""
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=layout_path,
            custom_mission="Reach the goal.",
            agent_start_dir=int(direction),
            max_steps=int(max_steps),
            render_mode="rgb_array" if gif_path is not None else None,
        )
    )
    observation, _ = env.reset(seed=int(seed))
    current_state = minigrid_utils.ColRowCanl_to_CanlRowCol(
        observation["image"]
    )
    current_inventory = carrying_token_from_env(env)
    goal_state, goal_yx = _goal_state(current_state)
    frames = [env.render()] if gif_path is not None else []
    trace = []
    reward_sum = 0.0
    total_expansions = 0
    total_visited_states = 0
    total_search_seconds = 0.0
    transition_checks = 0
    transition_mismatches = 0
    pose_mismatches = 0
    inventory_mismatches = 0
    terminated = truncated = False
    no_plan = False

    for step in range(int(max_steps)):
        search_started = time.perf_counter()
        actions = planner.plan(
            current_state,
            goal_state,
            k=int(max_depth),
            inventory_token=int(current_inventory),
            max_expansions=int(max_expansions),
            progress_every_expansions=int(progress_every_expansions),
        )
        search_seconds = time.perf_counter() - search_started
        diagnostics = dict(planner.last_plan_diagnostics)
        total_expansions += int(diagnostics.get("expansions", 0))
        total_visited_states += int(diagnostics.get("visited_states", 0))
        total_search_seconds += search_seconds
        if not actions:
            no_plan = True
            print(
                f"[WM A*][dir={direction} seed={seed} step={step}] "
                f"no_plan expansions={diagnostics.get('expansions', 0)} "
                f"search={search_seconds:.3f}s"
            )
            break

        action = int(actions[0])
        predicted_state, predicted_inventory = planner.predict_transition(
            current_state,
            int(current_inventory),
            action,
        )
        next_observation, reward, terminated, truncated, _ = env.step(
            compact_to_native(action)
        )
        next_state = minigrid_utils.ColRowCanl_to_CanlRowCol(
            next_observation["image"]
        )
        next_inventory = carrying_token_from_env(env)
        predicted_pose = _agent_pose(predicted_state)
        real_pose = _agent_pose(next_state)
        pose_match = predicted_pose == real_pose
        inventory_match = int(predicted_inventory) == int(next_inventory)
        predicted_key = planner._state_key(
            predicted_state, int(predicted_inventory)
        )
        real_key = planner._state_key(next_state, int(next_inventory))
        transition_match = predicted_key == real_key
        transition_checks += 1
        transition_mismatches += int(not transition_match)
        pose_mismatches += int(not pose_match)
        inventory_mismatches += int(not inventory_match)
        reward_sum += float(reward)

        trace.append(
            {
                "step": step,
                "action": action,
                "action_name": COMPACT_ACTION_NAMES[action],
                "planned_path_length": len(actions),
                "expansions": int(diagnostics.get("expansions", 0)),
                "visited_states": int(diagnostics.get("visited_states", 0)),
                "search_seconds": search_seconds,
                "predicted_pose_yxd": predicted_pose,
                "real_pose_yxd": real_pose,
                "predicted_inventory": int(predicted_inventory),
                "real_inventory": int(next_inventory),
                "pose_match": pose_match,
                "inventory_match": inventory_match,
                "transition_match": transition_match,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if gif_path is not None:
            frames.append(env.render())
        if print_every_steps > 0 and (
            step % int(print_every_steps) == 0 or terminated or truncated
        ):
            print(
                f"[WM A*][dir={direction} seed={seed} step={step}] "
                f"action={COMPACT_ACTION_NAMES[action]} "
                f"plan_len={len(actions)} expansions={diagnostics.get('expansions', 0)} "
                f"transition_match={transition_match} reward={float(reward):.3f} "
                f"search={search_seconds:.3f}s"
            )

        current_state = next_state
        current_inventory = next_inventory
        if terminated or truncated:
            break

    final_y, final_x, _ = _agent_pose(current_state)
    env.close()
    if gif_path is not None and frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=max(1, int(gif_fps)))
    executed_steps = len(trace)
    reached_goal = bool(terminated and reward_sum > 0.0)
    result = {
        "direction": int(direction),
        "seed": int(seed),
        "execution_mode": "receding_horizon",
        "plan_found": bool(trace),
        "all_replans_found": bool(trace) and not no_plan,
        "replans": len(trace) + int(no_plan),
        "real_reached_goal": reached_goal,
        "real_reward": reward_sum,
        "real_executed_steps": executed_steps,
        "real_terminated": bool(terminated),
        "real_truncated": bool(truncated),
        "real_final_y": final_y,
        "real_final_x": final_x,
        "goal_y": int(goal_yx[0]),
        "goal_x": int(goal_yx[1]),
        "total_expansions": total_expansions,
        "total_visited_states": total_visited_states,
        "total_search_seconds": total_search_seconds,
        "search_real_env_steps": 0,
        "transition_checks": transition_checks,
        "transition_mismatches": transition_mismatches,
        "transition_match_rate": (
            1.0 - transition_mismatches / transition_checks
            if transition_checks
            else 0.0
        ),
        "pose_mismatches": pose_mismatches,
        "inventory_mismatches": inventory_mismatches,
    }
    return result, trace


def plan_and_validate_world_model(cfg: DictConfig) -> list[dict]:
    """Run open-loop or receding-horizon A* with a frozen world model."""
    if str(cfg.domain) != "minigrid":
        raise ValueError("WM GraphPlanner currently supports MiniGrid only")
    hparams_wm = cfg.attention_model
    planner_cfg = cfg.PPO
    astar_cfg = getattr(planner_cfg, "astar", None)

    def setting(name, legacy_name, default):
        if astar_cfg is not None and hasattr(astar_cfg, name):
            return getattr(astar_cfg, name)
        return getattr(planner_cfg, legacy_name, default)

    model, checkpoint, attention_mask_size = _load_world_model_for_planning(cfg)
    configured_output = setting(
        "output_dir", "wm_planner_output_dir", None
    )
    output_dir = _planning_output_dir(checkpoint, configured_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = [
        int(value)
        for value in setting(
            "initial_directions", "wm_planner_initial_directions", [0]
        )
    ]
    if not directions or any(value not in range(4) for value in directions):
        raise ValueError("PPO.astar.initial_directions must contain values 0..3")
    seeds = [
        int(value)
        for value in setting("seeds", "wm_planner_seeds", [planner_cfg.seed])
    ]
    if not seeds:
        raise ValueError("PPO.astar.seeds must not be empty")
    max_depth = int(setting("max_depth", "wm_planner_max_depth", 128))
    max_expansions = int(
        setting("max_expansions", "wm_planner_max_expansions", 100000)
    )
    execution_mode = str(
        setting("execution_mode", "wm_planner_execution_mode", "receding")
    ).lower()
    if execution_mode not in {"receding", "open_loop"}:
        raise ValueError(
            "PPO.astar.execution_mode must be 'receding' or 'open_loop'"
        )
    validate_real = bool(
        setting("validate_real", "wm_planner_validate_real", True)
    )
    rollout_plans_in_real_env = bool(
        setting("rollout_plans_in_real_env", "wm_planner_rollout_plans_in_real_env", False)
    )
    save_gif = bool(setting("save_gif", "wm_planner_save_gif", False))
    gif_fps = int(setting("gif_fps", "wm_planner_gif_fps", 10))
    print_every_steps = int(
        setting("print_every_steps", "wm_planner_print_every_steps", 1)
    )
    progress_every_expansions = int(
        setting(
            "progress_every_expansions",
            "wm_planner_progress_every_expansions",
            5000,
        )
    )
    max_real_steps = int(
        setting("max_real_steps", "wm_planner_max_real_steps", planner_cfg.max_ep_len)
    )
    layout_path = str(planner_cfg.env_path)
    if bool(setting("evaluate_all_targets", "wm_planner_evaluate_all_targets", False)):
        target_dir = Path(str(setting("target_dir", "wm_planner_target_dir", ""))).expanduser()
        layout_paths = sorted(
            target_dir.glob("target_task*.txt"),
            key=lambda path: int(path.stem.removeprefix("target_task")),
        )
        if not layout_paths:
            raise FileNotFoundError(f"No target_task*.txt files found in {target_dir}")
    else:
        layout_paths = [Path(layout_path)]

    planner = GraphPlanner(
        model,
        MODEL_ACTION_COUNT,
        attention_mask_size,
        hparams_wm.valid_values_obj,
        hparams_wm.valid_values_color,
        hparams_wm.valid_values_state,
    )
    results = []
    print(f"[WM Planner] checkpoint={checkpoint}")
    print(f"[WM Planner] layout={layout_path}")
    print(
        f"[WM Planner] target_layouts={len(layout_paths)} "
        f"batch={len(layout_paths) > 1}"
    )
    print(
        f"[WM Planner] mode={execution_mode} validate_real={validate_real} "
        f"max_depth={max_depth} max_expansions={max_expansions}"
    )
    for target_layout in layout_paths:
      layout_path = str(target_layout.resolve())
      print(f"[WM Planner] target={Path(layout_path).stem}")
      for seed, direction in itertools.product(seeds, directions):
        if execution_mode == "receding" and validate_real and not bool(
            setting("evaluate_all_targets", "wm_planner_evaluate_all_targets", False)
        ):
            gif_path = (
                output_dir / f"seed_{seed}_direction_{direction}_receding.gif"
                if save_gif
                else None
            )
            row, trace = run_receding_horizon_episode(
                planner=planner,
                layout_path=layout_path,
                direction=direction,
                seed=seed,
                max_steps=max_real_steps,
                max_depth=max_depth,
                max_expansions=max_expansions,
                progress_every_expansions=progress_every_expansions,
                gif_path=gif_path,
                gif_fps=gif_fps,
                print_every_steps=print_every_steps,
            )
            trace_path = (
                output_dir
                / f"seed_{seed}_direction_{direction}_receding_trace.json"
            )
            with trace_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "planning_source": "world_model_astar",
                        "wm_checkpoint": str(checkpoint),
                        "layout_path": layout_path,
                        "direction": direction,
                        "seed": seed,
                        "trace": trace,
                    },
                    handle,
                    indent=2,
                    default=_json_default,
                )
            row.update(
                {
                    "plan_length": (
                        int(trace[0]["planned_path_length"]) if trace else 0
                    ),
                    "expansions": int(row["total_expansions"]),
                    "visited_states": int(row["total_visited_states"]),
                    "search_seconds": float(row["total_search_seconds"]),
                    "wm_reached_goal": bool(trace),
                    "actions_path": str(trace_path),
                }
            )
            results.append(row)
            print(
                f"[WM Planner][dir={direction} seed={seed}] "
                f"real_goal={row['real_reached_goal']} "
                f"steps={row['real_executed_steps']} "
                f"replans={row['replans']} "
                f"transition_match={row['transition_match_rate']:.1%}"
            )
            continue

        # Exactly one reset supplies the seed observation. The environment is
        # closed before search, so no real transition enters GraphPlanner.
        seed_env = FullyObsWrapper(
            CustomMiniGridEnv(
                txt_file_path=layout_path,
                custom_mission="Reach the goal.",
                agent_start_dir=direction,
                max_steps=int(planner_cfg.max_ep_len),
                render_mode=None,
            )
        )
        observation, _ = seed_env.reset(seed=seed)
        initial_state = minigrid_utils.ColRowCanl_to_CanlRowCol(
            observation["image"]
        )
        initial_inventory = carrying_token_from_env(seed_env)
        seed_env.close()
        goal_state, goal_yx = _goal_state(initial_state)

        search_started = time.perf_counter()
        actions = planner.plan(
            initial_state,
            goal_state,
            k=max_depth,
            inventory_token=initial_inventory,
            max_expansions=max_expansions,
            progress_every_expansions=progress_every_expansions,
        )
        search_seconds = time.perf_counter() - search_started
        diagnostics = dict(planner.last_plan_diagnostics)
        wm_replay = (
            replay_actions_in_world_model(
                model,
                initial_state,
                actions,
                attention_mask_size,
                initial_inventory_token=initial_inventory,
            )
            if actions
            else {"reached_goal": False}
        )
        target_name = Path(layout_path).stem
        actions_path = (
            output_dir / f"{target_name}_seed_{seed}_direction_{direction}_actions.json"
        )
        with actions_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "planning_source": "world_model_astar",
                    "wm_checkpoint": str(checkpoint),
                    "layout_path": layout_path,
                    "direction": direction,
                    "seed": seed,
                    "goal_yx": [int(goal_yx[0]), int(goal_yx[1])],
                    "actions": actions,
                    "action_names": [COMPACT_ACTION_NAMES[a] for a in actions],
                    "diagnostics": diagnostics,
                },
                handle,
                indent=2,
                default=_json_default,
            )

        row = {
            "target": target_name,
            "direction": direction,
            "seed": seed,
            "execution_mode": "open_loop",
            "plan_found": bool(actions),
            "plan_length": len(actions),
            "expansions": int(diagnostics.get("expansions", 0)),
            "visited_states": int(diagnostics.get("visited_states", 0)),
            "search_seconds": search_seconds,
            "search_real_env_steps": 0,
            "wm_reached_goal": bool(wm_replay.get("reached_goal", False)),
            "real_reached_goal": False,
            "real_reward": 0.0,
            "real_executed_steps": 0,
            "real_terminated": False,
            "real_truncated": False,
            "real_final_y": "",
            "real_final_x": "",
            "actions_path": str(actions_path),
        }
        if actions and rollout_plans_in_real_env:
            gif_path = (
                output_dir / f"{target_name}_seed_{seed}_direction_{direction}_real_rollout.gif"
                if save_gif
                else None
            )
            row.update(
                _execute_plan_in_real_env(
                    layout_path=layout_path,
                    actions=actions,
                    direction=direction,
                    seed=seed,
                    max_steps=int(planner_cfg.max_ep_len),
                    gif_path=gif_path,
                    gif_fps=gif_fps,
                )
            )
        results.append(row)
        print(
            f"[WM Planner][dir={direction} seed={seed}] found={row['plan_found']} "
            f"length={row['plan_length']} expansions={row['expansions']} "
            f"wm_goal={row['wm_reached_goal']} "
            f"real_goal={row['real_reached_goal']} "
            f"search={search_seconds:.3f}s"
        )

    summary_path = output_dir / "wm_planner_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    success_count = sum(bool(row["real_reached_goal"]) for row in results)
    print(f"[WM Planner] results={summary_path}")
    if validate_real:
        print(
            f"[WM Planner] real success={success_count}/{len(results)} "
            f"({success_count / len(results):.1%})"
        )
    return results


@hydra.main(
    version_base=None,
    config_path=str(WM_ROOT / "modelBased/config"),
    config_name="config",
)
def run_planner_rollout(cfg: DictConfig):
    return plan_and_validate_world_model(cfg)


if __name__ == "__main__":
    run_planner_rollout()
