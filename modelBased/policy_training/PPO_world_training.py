import sys
import random
import json
from collections import Counter, deque
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from modelBased.common.utils import PROJECT_ROOT as REPO_ROOT
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from domain.minigrid.minigrid_support import stochastic_env_kwargs
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
from modelBased.policy_training.PPO import PPO
from modelBased.policy_training.experiment_naming import (
    policy_checkpoint_path,
    policy_wandb_identity,
)
from modelBased.common.artifacts import world_model_checkpoint_path
import hydra
from datetime import datetime
from modelBased.common import utils
from domain.minigrid.action_codec import (
    COMPACT_ACTION_NAMES,
    INVENTORY_TOKEN_COUNT,
    MODEL_ACTION_COUNT,
    carrying_token_from_env,
    compact_to_native,
)
from modelBased.policy_training.dijkstra_planner import (
    plan_exact_minigrid,
    replay_actions_in_real_env,
    replay_actions_in_world_model,
)
from modelBased.policy_training.minigrid_wm_rollout import rollout_minigrid_wm
from modelBased.policy_training.minigrid_dense_reward import (
    build_goal_distance_map as build_main_goal_distance_map,
    build_door_topology,
    build_goal_region_mask,
    main_dense_rewards,
    reward_settings as main_dense_reward_settings,
)

from omegaconf import DictConfig, OmegaConf 
from modelBased.world_model import AttentionWM_support
from modelBased.world_model import Embedding_support
from modelBased.world_model import MLP_support
import wandb
from modelBased.policy_training.PPO import preprocess_observation 
import time



# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def seed_policy_training(seed):
    """Seed policy initialization, action sampling, and environment resets."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def append_inventory_to_policy_state(state_batch, carrying_key):
    """Append a colour-aware one-hot inventory to flat PPO observations."""
    tokens = carrying_key.long() + 1
    inventory = torch.nn.functional.one_hot(
        tokens, num_classes=INVENTORY_TOKEN_COUNT
    ).to(dtype=state_batch.dtype, device=state_batch.device)
    return torch.cat((state_batch, inventory), dim=-1)

def get_destination(obs, episode, maxstep, destination):
    """
    from the obs state, check if the agent has reached the destination
    and return done and reward

    1.object:("unseen": 0,  "empty": 1, "wall": 2, "door": 4, "key": 5, "goal": 8, "agent": 10)
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10

    2. color:
    "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5

    3. status
    State, 0: open, 1: closed, 2: locked

    check from wrappers.py full_obs-->encode
    """
    if obs[0, destination[0], destination[1]] == 10:
        # agent has reached the destination
        if episode >= maxstep:
            done = True
            reward = 0
        else:
            reward = 1 - 0.9 * (episode / maxstep)
            done = True
    else:
        done = False
        reward = 0
    return done, reward


def find_position(array, target):
    """
    Find the position of a target value in a 3D numpy array.
    
    Args:
        array (np.ndarray): The 3D array to search.
        target (tuple): The target value to locate (e.g., (8, 1, 0)).

    Returns:
        tuple: The position (x, y) of the target in the array if found, otherwise None.
    """
    # Find all indices where the value matches the target
    target = np.array(target).reshape(-1, 1, 1)
    result = np.argwhere((array == target).all(axis=0))

    # Check if any matches were found
    if result.size > 0:
        return tuple(result[0])  # Return the first match as a tuple (x, y)
    else:
        return None


def build_optimistic_goal_distance_map(state, goal_position_yx):
    """Return grid distances to goal for inexpensive rollout diagnostics.

    Walls and lava are treated as blocked. Keys and doors are treated as
    traversable because the agent can interact with them, so this is an
    optimistic navigation distance rather than a task-completion oracle.
    """
    state_np = torch.as_tensor(state).detach().cpu().numpy()
    if state_np.ndim != 3 or state_np.shape[0] != 3:
        raise ValueError(
            "Expected a MiniGrid state with shape (3,H,W), got "
            f"{tuple(state_np.shape)}"
        )
    obj = state_np[0]
    rows, cols = obj.shape
    goal_y, goal_x = map(int, goal_position_yx)
    distances = np.full((rows, cols), -1, dtype=np.int64)
    distances[goal_y, goal_x] = 0
    queue = deque([(goal_y, goal_x)])
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            if distances[ny, nx] >= 0:
                continue
            # MiniGrid object IDs: wall=2, lava=9.
            if int(obj[ny, nx]) in (2, 9):
                continue
            distances[ny, nx] = distances[y, x] + 1
            queue.append((ny, nx))
    return torch.as_tensor(distances, device=state.device, dtype=torch.long)


def minigrid_front_cells(states, agent_positions):
    """Return object/state values and coordinates directly in front of agents."""
    if states.ndim != 4 or states.shape[1] != 3:
        raise ValueError(f"Expected MiniGrid states [B,3,H,W], got {tuple(states.shape)}")
    bsz, _, rows, cols = states.shape
    batch_ids = torch.arange(bsz, device=states.device)
    positions = agent_positions.long()
    directions = states[
        batch_ids, 2, positions[:, 0], positions[:, 1]
    ].long()
    valid_direction = (directions >= 0) & (directions < 4)
    # MiniGrid directions 0..3 are right, down, left, up. Coordinates are y,x.
    direction_yx = torch.tensor(
        ((0, 1), (1, 0), (0, -1), (-1, 0)),
        device=states.device,
        dtype=torch.long,
    )
    offsets = direction_yx[directions.clamp(0, 3)]
    front_y = positions[:, 0] + offsets[:, 0]
    front_x = positions[:, 1] + offsets[:, 1]
    valid = (
        valid_direction
        & (front_y >= 0)
        & (front_y < rows)
        & (front_x >= 0)
        & (front_x < cols)
    )
    safe_y = front_y.clamp(0, rows - 1)
    safe_x = front_x.clamp(0, cols - 1)
    front_object = states[batch_ids, 0, safe_y, safe_x].long()
    front_state = states[batch_ids, 2, safe_y, safe_x].long()
    front_object = torch.where(valid, front_object, torch.full_like(front_object, -1))
    front_state = torch.where(valid, front_state, torch.full_like(front_state, -1))
    return front_object, front_state, safe_y, safe_x, valid

def process_data(state, maks_size):
    return utils.extract_masked_state_torch(state, maks_size)

def imagined_minigrid_step_batch(
    model,
    states,
    actions,
    carrying_key,
    attention_mask_size,
    valid_values_obj=None,
    valid_values_color=None,
    valid_values_state=None,
    *,
    agent_positions=None,
):
    """Advance B imagined states and inventories with one learned WM call."""
    next_states, inventory_tokens, _ = rollout_minigrid_wm(
        model,
        states,
        actions,
        carrying_key.long() + 1,
        attention_mask_size,
        agent_positions=agent_positions,
        collect_diagnostics=False,
    )
    return next_states, inventory_tokens - 1


def policy_selection_paths(checkpoint_path):
    """Return the canonical, best, last, and manifest paths for WM PPO."""
    canonical = Path(checkpoint_path)
    suffix = canonical.suffix or ".ckpt"
    stem = canonical.stem if canonical.suffix else canonical.name
    return {
        "canonical": canonical,
        "best": canonical.with_name(f"{stem}_best{suffix}"),
        "last": canonical.with_name(f"{stem}_last{suffix}"),
        "manifest": canonical.with_name("policy_selection.json"),
    }


def policy_selection_is_better(candidate, incumbent):
    """Lexicographic WM-only policy ranking: success, distance, then steps."""
    if incumbent is None:
        return True
    candidate_key = (
        int(candidate["success_count"]),
        -float(candidate["mean_min_goal_distance"]),
        -float(candidate["mean_steps"]),
    )
    incumbent_key = (
        int(incumbent["success_count"]),
        -float(incumbent["mean_min_goal_distance"]),
        -float(incumbent["mean_steps"]),
    )
    return candidate_key > incumbent_key


def evaluate_deterministic_minigrid_wm_policy(
    ppo_agent,
    model,
    initial_state_templates,
    goal_position_yx,
    goal_distance_map,
    attention_mask_size,
    obs_norm_values,
    directions,
    max_steps,
):
    """Score ``policy_old`` in fixed WM rollouts without sampling or real env use.

    Each requested initial direction gets one argmax rollout.  Scores are
    lexicographically comparable by successful directions, then optimistic
    minimum goal distance, then steps (failed rollouts count as ``max_steps``).
    """
    directions = [int(direction) for direction in directions]
    states = initial_state_templates[directions].clone()
    batch_size = len(directions)
    carrying_key = torch.full((batch_size,), -1, device=states.device, dtype=torch.long)
    goal_positions = torch.as_tensor(goal_position_yx, device=states.device, dtype=torch.long)
    goal_positions = goal_positions.expand(batch_size, -1)
    agent_positions = utils.get_agent_position_torch(states)
    initial_distances = goal_distance_map[
        agent_positions[:, 0].long(), agent_positions[:, 1].long()
    ].float()
    # A disconnected optimistic map should rank below any finite distance but
    # remain a finite JSON metric.
    initial_distances = torch.where(
        torch.isfinite(initial_distances) & (initial_distances >= 0),
        initial_distances,
        torch.full_like(initial_distances, float(max_steps)),
    )
    min_distances = initial_distances.clone()
    steps = torch.full((batch_size,), int(max_steps), device=states.device, dtype=torch.long)
    successes = torch.zeros(batch_size, device=states.device, dtype=torch.bool)
    active = torch.ones(batch_size, device=states.device, dtype=torch.bool)
    old_training = ppo_agent.policy_old.training
    ppo_agent.policy_old.eval()
    try:
        with torch.no_grad():
            for step in range(1, int(max_steps) + 1):
                if not bool(active.any()):
                    break
                previous_states = states
                previous_positions = utils.get_agent_position_torch(states)
                front_objects, _, _, _, valid_front = minigrid_front_cells(
                    states, previous_positions
                )
                policy_states = utils.normalize_obs(
                    states.clone(), obs_norm_values
                ).reshape(batch_size, -1)
                policy_states = append_inventory_to_policy_state(policy_states, carrying_key)
                actions = torch.argmax(ppo_agent.policy_old.actor(policy_states), dim=-1)
                next_states, next_carrying = imagined_minigrid_step_batch(
                    model,
                    states,
                    actions,
                    carrying_key,
                    attention_mask_size,
                    agent_positions=previous_positions,
                )
                lava = active & (actions == 2) & valid_front & (front_objects == 9)
                agent_present = (next_states[:, 0] == 10).flatten(1).any(dim=1)
                agent_lost = active & ~agent_present
                # Match training's transition semantics: a missing agent is
                # always restored before positions/distances are measured;
                # lava remains a separate terminal outcome.
                states = torch.where(
                    agent_lost[:, None, None, None], previous_states, next_states
                )
                carrying_key = torch.where(agent_lost, carrying_key, next_carrying)
                current_positions = utils.get_agent_position_torch(states)
                distances = goal_distance_map[
                    current_positions[:, 0].long(), current_positions[:, 1].long()
                ].float()
                distances = torch.where(
                    torch.isfinite(distances) & (distances >= 0),
                    distances,
                    torch.full_like(distances, float(max_steps)),
                )
                min_distances = torch.where(active, torch.minimum(min_distances, distances), min_distances)
                reached_goal = active & ~lava & ~agent_lost & torch.all(
                    current_positions == goal_positions, dim=1
                )
                just_finished = active & (reached_goal | lava | agent_lost)
                # Failures intentionally retain max_steps, as stated in the
                # selection metric; only successful directions get their
                # actual trajectory length.
                steps[reached_goal] = step
                successes |= reached_goal
                active &= ~just_finished
    finally:
        ppo_agent.policy_old.train(old_training)
    per_direction = [
        {
            "direction": direction,
            "success": bool(successes[index].item()),
            "steps": int(steps[index].item()),
            "min_goal_distance": float(min_distances[index].item()),
        }
        for index, direction in enumerate(directions)
    ]
    return {
        "success_count": int(successes.sum().item()),
        "mean_min_goal_distance": float(min_distances.mean().item()),
        "mean_steps": float(steps.float().mean().item()),
        "per_direction": per_direction,
    }


def planner_behavior_cloning_warm_start(cfg, model, ppo_agent):
    """Build verified demonstrations and warm-start PPO's actor.

    Paths are generated with exact MiniGrid dynamics, replayed in the real
    environment, and then replayed open-loop through the learned WM.  BC uses
    the WM states because those are the observations the policy will receive
    during imagined PPO training.  This preserves the existing sparse reward,
    six-action space and world-model transition implementation.
    """
    ppo_cfg = cfg.PPO
    if not bool(getattr(ppo_cfg, "planner_warm_start", False)):
        return None
    if str(getattr(ppo_cfg, "env_type", "")) != "minigrid":
        raise ValueError("PPO planner_warm_start currently supports MiniGrid only")

    directions = list(
        getattr(ppo_cfg, "planner_initial_directions", [0, 1, 2, 3])
    )
    directions = [int(direction) for direction in directions]
    if not directions or any(direction not in range(4) for direction in directions):
        raise ValueError("PPO.planner_initial_directions must contain values 0..3")
    max_expansions = int(
        getattr(ppo_cfg, "planner_max_expansions", 100000)
    )
    min_pose_match = float(
        getattr(ppo_cfg, "planner_min_wm_pose_match", 1.0)
    )
    require_wm_goal = bool(
        getattr(ppo_cfg, "planner_require_wm_goal", True)
    )
    seed = int(getattr(ppo_cfg, "seed", 0))
    all_states = []
    all_actions = []
    path_metrics = []
    validation_starts = []

    for direction in directions:
        demo_env = FullyObsWrapper(
            CustomMiniGridEnv(
                txt_file_path=str(ppo_cfg.env_path),
                custom_mission="Find the key and open the door.",
                agent_start_dir=direction,
                max_steps=int(ppo_cfg.max_ep_len),
                render_mode=None,
            )
        )
        initial_observation = demo_env.reset(seed=seed)[0]["image"]
        initial_state = utils.ColRowCanl_to_CanlRowCol(initial_observation)
        validation_starts.append((direction, initial_state.copy()))
        actions, search_metrics = plan_exact_minigrid(
            demo_env, max_expansions=max_expansions
        )
        if not actions:
            raise RuntimeError(
                "Planner warm-start could not find a real MiniGrid path for "
                f"initial direction {direction}: {search_metrics}"
            )

        demo_env.reset(seed=seed)
        real_replay = replay_actions_in_real_env(demo_env, actions)
        if not real_replay["reached_goal"]:
            raise RuntimeError(
                "Planner produced a path that did not reach the goal in the "
                f"real environment for direction {direction}"
            )
        wm_replay = replay_actions_in_world_model(
            model,
            initial_state,
            actions,
            cfg.attention_model.attention_mask_size,
            initial_inventory_token=0,
            real_reference=real_replay,
        )
        pose_match = float(wm_replay["pose_match_rate"] or 0.0)
        if require_wm_goal and not wm_replay["reached_goal"]:
            raise RuntimeError(
                "Verified real path does not reach the goal in WM open-loop "
                f"for direction {direction}; pose_match={pose_match:.2%}. "
                "The current WM is not safe for planner BC on this layout."
            )
        if pose_match + 1e-12 < min_pose_match:
            raise RuntimeError(
                f"WM/real planner replay pose match {pose_match:.2%} is below "
                f"PPO.planner_min_wm_pose_match={min_pose_match:.2%} for "
                f"direction {direction}"
            )

        for state, inventory_token in zip(
            wm_replay["states"], wm_replay["inventory_tokens"]
        ):
            all_states.append(
                preprocess_observation(
                    state,
                    cfg.attention_model.obs_norm_values,
                    inventory_token=inventory_token,
                    inventory_classes=INVENTORY_TOKEN_COUNT,
                )
            )
        all_actions.extend(wm_replay["actions"])
        direction_metrics = {
            **search_metrics,
            "direction": direction,
            "real_goal": real_replay["reached_goal"],
            "wm_goal": wm_replay["reached_goal"],
            "wm_pose_match": pose_match,
            "wm_inventory_match": float(
                wm_replay["inventory_match_rate"] or 0.0
            ),
        }
        path_metrics.append(direction_metrics)
        print(
            f"[Planner BC][dir={direction}] path={len(actions)} "
            f"real_goal={real_replay['reached_goal']} "
            f"wm_goal={wm_replay['reached_goal']} "
            f"pose_match={pose_match:.2%} "
            f"inventory_match={direction_metrics['wm_inventory_match']:.2%}"
        )

    # Different shortest paths can merge at the same state and then choose
    # different but equally valid routes. A categorical policy cannot assign
    # near-one probability to conflicting labels, so collapse exact duplicate
    # WM observations to one deterministic majority action.
    unique_examples = {}
    for state, action in zip(all_states, all_actions):
        key = state.detach().cpu().numpy().tobytes()
        if key not in unique_examples:
            unique_examples[key] = [state, Counter(), int(action)]
        unique_examples[key][1][int(action)] += 1
    conflict_states = sum(
        len(action_counts) > 1
        for _, action_counts, _ in unique_examples.values()
    )
    unique_states = []
    unique_actions = []
    for state, action_counts, first_action in unique_examples.values():
        highest_count = max(action_counts.values())
        candidates = {
            action
            for action, count in action_counts.items()
            if count == highest_count
        }
        chosen_action = (
            first_action if first_action in candidates else min(candidates)
        )
        unique_states.append(state)
        unique_actions.append(chosen_action)
    print(
        f"[Planner BC] raw_examples={len(all_states)} "
        f"unique_examples={len(unique_states)} "
        f"conflicting_states={conflict_states}"
    )
    state_tensor = torch.stack(unique_states, dim=0)
    action_tensor = torch.as_tensor(unique_actions, dtype=torch.long)
    bc_metrics = ppo_agent.behavior_clone(
        state_tensor,
        action_tensor,
        epochs=int(getattr(ppo_cfg, "planner_bc_epochs", 1000)),
        batch_size=int(getattr(ppo_cfg, "planner_bc_batch_size", 256)),
        learning_rate=float(getattr(ppo_cfg, "planner_bc_lr", 3e-4)),
        target_accuracy=float(
            getattr(ppo_cfg, "planner_bc_target_accuracy", 0.995)
        ),
        target_probability=float(
            getattr(ppo_cfg, "planner_bc_target_probability", 0.995)
        ),
    )
    minimum_accuracy = float(
        getattr(ppo_cfg, "planner_bc_min_accuracy", 0.99)
    )
    if bc_metrics["accuracy"] + 1e-12 < minimum_accuracy:
        raise RuntimeError(
            f"Planner BC accuracy {bc_metrics['accuracy']:.2%} is below "
            f"PPO.planner_bc_min_accuracy={minimum_accuracy:.2%}"
        )
    minimum_probability = float(
        getattr(ppo_cfg, "planner_bc_min_target_probability", 0.99)
    )
    if (
        bc_metrics["mean_target_probability"] + 1e-12
        < minimum_probability
    ):
        raise RuntimeError(
            "Planner BC mean target-action probability "
            f"{bc_metrics['mean_target_probability']:.4f} is below "
            "PPO.planner_bc_min_target_probability="
            f"{minimum_probability:.4f}"
        )

    # End-to-end acceptance: execute the cloned actor (argmax) through the WM,
    # rather than treating supervised accuracy as proof of planning ability.
    cloned_rollouts = []
    validation_horizon = int(
        getattr(ppo_cfg, "planner_bc_validation_horizon", ppo_cfg.max_ep_len)
    )
    for direction, initial_state in validation_starts:
        state = torch.as_tensor(
            initial_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        carrying_key = torch.full(
            (1,), -1, dtype=torch.long, device=device
        )
        goal_yx = find_position(initial_state, (8, 1, 0))
        reached_goal = False
        chosen_probabilities = []
        for step in range(validation_horizon):
            policy_state = utils.normalize_obs(
                state.clone(), cfg.attention_model.obs_norm_values
            ).reshape(1, -1)
            policy_state = append_inventory_to_policy_state(
                policy_state, carrying_key
            )
            with torch.no_grad():
                probabilities = ppo_agent.policy_old.actor(policy_state)
                actions = probabilities.argmax(dim=1)
                chosen_probabilities.append(
                    float(probabilities[0, actions[0]].item())
                )
                state, carrying_key = imagined_minigrid_step_batch(
                    model,
                    state,
                    actions,
                    carrying_key,
                    cfg.attention_model.attention_mask_size,
                )
            position = utils.get_agent_position_torch(state)[0]
            if tuple(map(int, position.detach().cpu().tolist())) == goal_yx:
                reached_goal = True
                break
        rollout_metrics = {
            "direction": direction,
            "reached_goal": reached_goal,
            "steps": step + 1,
            "mean_chosen_probability": float(np.mean(chosen_probabilities)),
            "min_chosen_probability": float(np.min(chosen_probabilities)),
        }
        cloned_rollouts.append(rollout_metrics)
        print(
            f"[Planner BC validation][dir={direction}] "
            f"goal={reached_goal} steps={step + 1} "
            f"mean_prob={rollout_metrics['mean_chosen_probability']:.4f} "
            f"min_prob={rollout_metrics['min_chosen_probability']:.4f}"
        )
        if not reached_goal:
            raise RuntimeError(
                "Behavior-cloned actor failed its deterministic WM rollout "
                f"for initial direction {direction}"
            )
    return {
        "paths": path_metrics,
        "bc": bc_metrics,
        "cloned_rollouts": cloned_rollouts,
        "raw_examples": len(all_states),
        "unique_examples": len(unique_states),
        "conflicting_states": conflict_states,
    }




@hydra.main(version_base=None, config_path=str(SCRIPT_ROOT / "modelBased/config"), config_name="config")
def training_agent_wm(cfg: DictConfig):
    regret = run_ppo_wm(cfg)
    return regret

def run_ppo_wm(cfg):
    hparams = cfg
    
    # 1. World Model
    hparams_world_model = hparams.attention_model

    MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
    # Initialize the world model.
    module_class = MODEL_MAPPING.get(hparams_world_model.model_type.lower())
    if (
        hparams_world_model.env_type == "minigrid"
        and str(getattr(hparams_world_model, "minigrid_transition_mode", "effect")) == "effect"
        and module_class is not AttentionWM_support.AttentionModule
    ):
        raise ValueError(
            "MiniGrid effect policy planning requires model_type=Attention"
        )
    if module_class is not None:
        model_kwargs = {}
        if module_class is AttentionWM_support.AttentionModule:
            model_kwargs.update(
                env_type=hparams_world_model.env_type,
                frame_stack=hparams_world_model.frame_stack,
                minigrid_transition_mode=getattr(
                    hparams_world_model, "minigrid_transition_mode", "effect"
                ),
            )
        model = module_class(
            hparams_world_model.data_type,  
            hparams_world_model.grid_shape, 
            hparams_world_model.attention_mask_size, 
            hparams_world_model.embed_dim, 
            hparams_world_model.num_heads,
            **model_kwargs,
        )
    else:
        print(f"Model type: {hparams_world_model.model_type} not supported")
        exit()
    configured_wm_ckpt = getattr(hparams.PPO, "checkpoint_path_wm", None)
    wm_ckpt = (
        Path(str(configured_wm_ckpt)).expanduser().resolve()
        if configured_wm_ckpt is not None and str(configured_wm_ckpt).strip() and str(configured_wm_ckpt).lower() != "null"
        else world_model_checkpoint_path(hparams, str(hparams.domain))
    )
    utils.load_model_weight(model, str(wm_ckpt))
    model.eval() 
    


    # 2. PPO
    # hyperparameters
    # compute regret
    hparams_PPO = hparams.PPO
    seed = int(getattr(hparams_PPO, "seed", 0))
    compute_regret = hparams_PPO.compute_regret
    if compute_regret:
        regret_eval_freq = hparams_PPO.get("regret_eval_freq", 5000)
        regret_eval_episodes = hparams_PPO.get("regret_eval_episodes", 5)
        real_policy_path = hparams_PPO.get("real_policy_path")


    start_time = datetime.now().replace(microsecond=0)
    lr_actor = hparams_PPO.lr_actor
    lr_critic = hparams_PPO.lr_critic
    gamma = hparams_PPO.gamma
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    action_std = hparams_PPO.action_std
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    max_training_timesteps = int(hparams_PPO.max_training_timesteps)
    save_model_freq = int(hparams_PPO.save_model_freq)
    max_ep_len = int(hparams_PPO.max_ep_len)
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    checkpoint_path = str(policy_checkpoint_path(cfg))
    env_path = hparams_PPO.env_path
    visualize_flag = hparams_PPO.visualize
    env_type =  hparams_PPO.env_type
    use_wandb = hparams_PPO.use_wandb
    update_timestep = int(getattr(hparams_PPO, "rollout_steps", 1024))
    if update_timestep < 2:
        raise ValueError("PPO.rollout_steps must be at least 2")
    num_imagined_envs = int(getattr(hparams_PPO, "num_imagined_envs", 1))
    if num_imagined_envs < 1:
        raise ValueError("PPO.num_imagined_envs must be at least 1")
    if update_timestep % num_imagined_envs != 0:
        raise ValueError(
            "PPO.rollout_steps must be divisible by PPO.num_imagined_envs"
        )
    if max_training_timesteps % num_imagined_envs != 0:
        raise ValueError(
            "PPO.max_training_timesteps must be divisible by "
            "PPO.num_imagined_envs"
        )
    console_log_every_steps = int(
        getattr(hparams_PPO, "console_log_every_steps", update_timestep)
    )
    episode_log_every_steps = int(
        getattr(hparams_PPO, "episode_log_every_steps", update_timestep)
    )
    wandb_log_every_updates = int(
        getattr(hparams_PPO, "wandb_log_every_updates", 1)
    )
    if console_log_every_steps < 1 or episode_log_every_steps < 1:
        raise ValueError("PPO console/episode log intervals must be at least 1")
    if wandb_log_every_updates < 1:
        raise ValueError("PPO.wandb_log_every_updates must be at least 1")
    entropy_coef = float(getattr(hparams_PPO, "entropy_coef", 0.01))
    normalize_advantages = bool(getattr(hparams_PPO, "normalize_advantages", True))
    normalize_returns = bool(getattr(hparams_PPO, "normalize_returns", False))
    max_grad_norm = float(getattr(hparams_PPO, "max_grad_norm", 0.5))
    freeze_actor_until_first_reward = bool(
        getattr(hparams_PPO, "freeze_actor_until_first_reward", True)
    )
    rolling_window_episodes = int(
        getattr(hparams_PPO, "rolling_window_episodes", 50)
    )
    if rolling_window_episodes < 1:
        raise ValueError("PPO.rolling_window_episodes must be at least 1")
    policy_diagnostics = bool(getattr(hparams_PPO, "policy_diagnostics", True))
    diagnostics_every_updates = int(
        getattr(hparams_PPO, "diagnostics_every_updates", 1)
    )
    if diagnostics_every_updates < 1:
        raise ValueError("PPO.diagnostics_every_updates must be at least 1")
    best_policy_selection_enabled = bool(
        getattr(hparams_PPO, "best_policy_selection_enabled", True)
    )
    selection_eval_freq_value = getattr(
        hparams_PPO, "best_policy_eval_freq", None
    )
    selection_eval_freq = (
        save_model_freq
        if selection_eval_freq_value is None
        else int(selection_eval_freq_value)
    )
    selection_directions = list(
        getattr(hparams_PPO, "best_policy_directions", [0, 1, 2, 3])
    )
    selection_max_steps_value = getattr(hparams_PPO, "best_policy_max_steps", None)
    selection_max_steps = (
        max_ep_len
        if selection_max_steps_value is None
        else int(selection_max_steps_value)
    )
    if best_policy_selection_enabled:
        if selection_eval_freq < 1:
            raise ValueError("PPO.best_policy_eval_freq must be positive or null")
        if selection_max_steps < 1:
            raise ValueError("PPO.best_policy_max_steps must be positive or null")
        if not selection_directions or any(
            int(direction) not in (0, 1, 2, 3) for direction in selection_directions
        ):
            raise ValueError(
                "PPO.best_policy_directions must be a non-empty subset of [0, 1, 2, 3]"
            )
        if len(set(map(int, selection_directions))) != len(selection_directions):
            raise ValueError("PPO.best_policy_directions must not contain duplicates")
        selection_paths = policy_selection_paths(checkpoint_path)
    else:
        selection_paths = None
    

    if use_wandb:
        sub_run = _init_policy_wandb_run(cfg, default_project="minigrid_policy_training")
    else:
        sub_run = None

    # training_agent()

    if visualize_flag and num_imagined_envs > 1:
        print(
            "[WM PPO] Disabling per-step visualization for parallel imagined "
            "rollouts. Set PPO.num_imagined_envs=1 to visualize every step."
        )
        visualize_flag = False
    if visualize_flag:
        visualize = utils.Visualization(hparams_world_model)
    seed_policy_training(seed)
    print(f"[PPO] Seed: {seed}")
    # 3. Real environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=max_ep_len, render_mode=None,
                        **stochastic_env_kwargs(cfg)))
    # 4. Initialize training
    i_episode = 0
    print_freq = console_log_every_steps
    print_running_reward = 0
    print_running_extrinsic_reward = 0
    print_running_shaping_reward = 0
    print_running_episodes = 0
    print_running_steps = 0
    print_running_successes = 0
    next_print_timestep = print_freq
    next_episode_log_timestep = episode_log_every_steps
    recent_rewards = deque(maxlen=rolling_window_episodes)
    recent_extrinsic_rewards = deque(maxlen=rolling_window_episodes)
    recent_shaping_rewards = deque(maxlen=rolling_window_episodes)
    recent_steps = deque(maxlen=rolling_window_episodes)
    recent_successes = deque(maxlen=rolling_window_episodes)
    time_step = 0
    next_save_timestep = save_model_freq
    step_penalty = float(getattr(hparams_PPO, "step_penalty", 0.0))
    use_main_dense_reward = bool(
        getattr(hparams_PPO, "use_main_dense_reward", False)
    )
    main_reward = main_dense_reward_settings(hparams_PPO)
    reward_shaping_enabled = bool(
        getattr(hparams_PPO, "reward_shaping_enabled", False)
    )
    position_novelty_reward = float(
        getattr(hparams_PPO, "position_novelty_reward", 0.0)
    )
    global_position_count_beta = float(
        getattr(hparams_PPO, "global_position_count_beta", 0.0)
    )
    manhattan_progress_reward_scale = float(
        getattr(hparams_PPO, "manhattan_progress_reward_scale", 0.0)
    )
    key_pickup_reward = float(getattr(hparams_PPO, "key_pickup_reward", 0.0))
    door_open_reward = float(getattr(hparams_PPO, "door_open_reward", 0.0))
    if not reward_shaping_enabled:
        position_novelty_reward = 0.0
        global_position_count_beta = 0.0
        manhattan_progress_reward_scale = 0.0
        key_pickup_reward = 0.0
        door_open_reward = 0.0
    shaping_values = {
        "position_novelty_reward": position_novelty_reward,
        "global_position_count_beta": global_position_count_beta,
        "manhattan_progress_reward_scale": manhattan_progress_reward_scale,
        "key_pickup_reward": key_pickup_reward,
        "door_open_reward": door_open_reward,
    }
    negative_shaping = {
        name: value for name, value in shaping_values.items() if value < 0.0
    }
    if negative_shaping:
        raise ValueError(
            "PPO reward-shaping bonuses must be non-negative: "
            f"{negative_shaping}"
        )
    final_norm_regret = None
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = int(np.prod(env.action_space.shape))
    else:
        # Keep PPO's action IDs identical to the MiniGrid environment and WM
        # dataset (the full MiniGrid action space is normally 0..6).
        # PPO/WM use six compact actions. Historical IDs 0-4 are unchanged;
        # drop is appended as 5 and only native done is excluded.
        action_dim = MODEL_ACTION_COUNT
    state_dim = np.prod(env.observation_space['image'].shape) + INVENTORY_TOKEN_COUNT
    # Constructing/loading the WM consumes PyTorch RNG. Re-seed immediately
    # before PPO construction so matched real/WM runs start from identical
    # actor and critic parameters.
    seed_policy_training(seed)
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        max_grad_norm=max_grad_norm,
        freeze_actor_until_first_reward=freeze_actor_until_first_reward,
        minibatch_size=int(getattr(hparams_PPO, "minibatch_size", 0)),
    )
    planner_warm_start_metrics = planner_behavior_cloning_warm_start(
        cfg, model, ppo_agent
    )
    if sub_run is not None and planner_warm_start_metrics is not None:
        bc_metrics = planner_warm_start_metrics["bc"]
        sub_run.log(
            {
                "planner_bc/examples": bc_metrics["examples"],
                "planner_bc/epochs": bc_metrics["epochs"],
                "planner_bc/loss": bc_metrics["loss"],
                "planner_bc/accuracy": bc_metrics["accuracy"],
                "planner_bc/mean_target_probability": bc_metrics[
                    "mean_target_probability"
                ],
                "planner_bc/min_target_probability": bc_metrics[
                    "min_target_probability"
                ],
            },
            step=0,
        )
    if compute_regret:
        real_policy_agent = PPO(
            state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
            has_continuous_action_space, action_std,
            minibatch_size=int(getattr(hparams_PPO, "minibatch_size", 0)),
        )
        real_policy_agent.load(real_policy_path)

    

    # The target layout and spawn position are fixed; only MiniGrid's initial
    # direction is random.  Materialize its four possible observations once
    # and sample the same NumPy direction draw that CustomMiniGridEnv used on
    # every reset.  This avoids rebuilding/parsing the map for every imagined
    # episode without changing the initial-state distribution.
    base_env = env.unwrapped
    original_random_dir = base_env.rand_agent_start_dir
    original_start_dir = base_env.agent_start_dir
    initial_state_templates = []
    try:
        for direction in range(4):
            base_env.rand_agent_start_dir = False
            base_env.agent_start_dir = direction
            observation = env.reset()[0]["image"]
            initial_state_templates.append(
                torch.as_tensor(
                    utils.ColRowCanl_to_CanlRowCol(observation), device=device
                )
            )
    finally:
        base_env.rand_agent_start_dir = original_random_dir
        base_env.agent_start_dir = original_start_dir
    initial_state_templates = torch.stack(initial_state_templates, dim=0)

    def reset_imagined_state():
        direction = int(np.random.randint(0, 4))
        return initial_state_templates[direction].clone()

    states = torch.stack(
        [reset_imagined_state() for _ in range(num_imagined_envs)], dim=0
    )
    first_state_numpy = states[0].detach().cpu().numpy()
    goal_position_yx = find_position(first_state_numpy, (8, 1, 0))
    if goal_position_yx is None:
        raise ValueError("The imagined MiniGrid layout does not contain a goal")
    goal_positions = torch.as_tensor(
        goal_position_yx, device=device, dtype=torch.long
    ).expand(num_imagined_envs, -1)
    goal_distance_map = build_optimistic_goal_distance_map(
        states[0], goal_position_yx
    )
    if use_main_dense_reward:
        goal_distance_map = build_main_goal_distance_map(states[0], goal_position_yx)

    selection_history = []
    best_selection_score = None
    last_selection_timestep = None
    next_selection_timestep = selection_eval_freq

    def write_policy_selection_manifest():
        if not best_policy_selection_enabled:
            return
        manifest = {
            "selection_source": "frozen_world_model_deterministic_argmax",
            "score_definition": (
                "Lexicographic: maximize success_count; then minimize "
                "mean_min_goal_distance; then minimize mean_steps. "
                "One fixed argmax rollout per configured initial direction; "
                "failures use max_steps. No real-environment evaluation is used."
            ),
            "enabled": True,
            "directions": [int(direction) for direction in selection_directions],
            "max_steps": selection_max_steps,
            "best_timestep": None if best_selection_score is None else best_selection_score["timestep"],
            "best_metrics": best_selection_score,
            "best_path": str(selection_paths["best"]),
            "last_path": str(selection_paths["last"]),
            "canonical_path": str(selection_paths["canonical"]),
            "evaluation_history": selection_history,
        }
        selection_paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        with selection_paths["manifest"].open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def evaluate_and_select_policy(timestep):
        nonlocal best_selection_score, last_selection_timestep
        timestep = int(timestep)
        if not best_policy_selection_enabled or last_selection_timestep == timestep:
            return False
        metrics = evaluate_deterministic_minigrid_wm_policy(
            ppo_agent,
            model,
            initial_state_templates,
            goal_position_yx,
            goal_distance_map,
            hparams_world_model.attention_mask_size,
            hparams_world_model.obs_norm_values,
            selection_directions,
            selection_max_steps,
        )
        candidate = {"timestep": timestep, **metrics}
        selection_history.append(candidate)
        last_selection_timestep = timestep
        ppo_agent.save(selection_paths["last"])
        if policy_selection_is_better(candidate, best_selection_score):
            best_selection_score = candidate
            ppo_agent.save(selection_paths["best"])
            # Keep old consumers/evaluators compatible: canonical is always best.
            ppo_agent.save(selection_paths["canonical"])
            selected = True
        else:
            selected = False
        write_policy_selection_manifest()
        print(
            "[WM PPO best-policy] "
            f"t={timestep} success={candidate['success_count']}/{len(selection_directions)} "
            f"min_dist={candidate['mean_min_goal_distance']:.3f} "
            f"steps={candidate['mean_steps']:.1f} selected={selected}"
        )
        return selected

    # This establishes both a best and last artifact before the first update.
    evaluate_and_select_policy(0)
    goal_region_mask = build_goal_region_mask(states[0], goal_position_yx)
    door_topology = build_door_topology(states[0], goal_position_yx)
    goal_region_seen = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    progress_milestone_seen = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    pending_door_ids = torch.full(
        (num_imagined_envs,), -1, device=device, dtype=torch.long
    )
    pending_door_origins = torch.full_like(pending_door_ids, -1)
    rewarded_door_crossings = torch.zeros(
        (num_imagined_envs, door_topology.num_doors), device=device, dtype=torch.bool
    )
    initial_agent_positions = utils.get_agent_position_torch(states)
    initial_goal_distances = goal_distance_map[
        initial_agent_positions[:, 0].long(), initial_agent_positions[:, 1].long()
    ].clone()
    best_goal_distances = initial_goal_distances.clone()
    initial_object_map = states[0, 0].clone()
    carrying_key = torch.full(
        (num_imagined_envs,), -1, device=device, dtype=torch.long
    )
    prev_carrying_key = carrying_key.clone()
    episode_rewards = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.float32
    )
    episode_extrinsic_rewards = torch.zeros_like(episode_rewards)
    episode_shaping_rewards = torch.zeros_like(episode_rewards)
    episode_steps = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.long
    )
    last_dones = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    batch_ids = torch.arange(num_imagined_envs, device=device)
    grid_height, grid_width = states.shape[-2:]
    visited_positions = torch.zeros(
        (num_imagined_envs, grid_height, grid_width),
        device=device,
        dtype=torch.bool,
    )
    rewarded_goal_region_key_positions = torch.zeros_like(visited_positions)
    rewarded_goal_region_door_positions = torch.zeros_like(visited_positions)
    rewarded_door_positions = torch.zeros_like(visited_positions)
    key_rewarded = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    initial_agent_positions = utils.get_agent_position_torch(states)
    global_position_counts = torch.zeros(
        grid_height * grid_width, device=device, dtype=torch.float32
    )
    initial_flat_positions = (
        initial_agent_positions[:, 0].long() * grid_width
        + initial_agent_positions[:, 1].long()
    )
    global_position_counts.scatter_add_(
        0,
        initial_flat_positions,
        torch.ones_like(initial_flat_positions, dtype=torch.float32),
    )
    visited_positions[
        batch_ids,
        initial_agent_positions[:, 0].long(),
        initial_agent_positions[:, 1].long(),
    ] = True

    if sub_run is not None:
        sub_run.log({"final_tasks": wandb.Image(env.get_frame())})

    print(
        f"[WM PPO] Parallel imagined environments: {num_imagined_envs}; "
        f"temporal rollout: {update_timestep // num_imagined_envs}; "
        f"transitions/update: {update_timestep}"
    )
    print(
        "[WM PPO] Reward shaping: "
        f"enabled={reward_shaping_enabled}; main_dense={use_main_dense_reward}; "
        f"global_count_beta={global_position_count_beta:g}; "
        f"manhattan_progress={manhattan_progress_reward_scale:g}; "
        f"episodic_position_novelty={position_novelty_reward:g}; "
        f"key_pickup={key_pickup_reward:g}; "
        f"door_open={door_open_reward:g}; action_mask=False"
    )

    def new_update_diagnostics():
        scalar = lambda: torch.zeros((), device=device, dtype=torch.float64)
        return {
            "transitions": 0,
            "action_counts": torch.zeros(action_dim, device=device, dtype=torch.long),
            "path_distance": scalar(),
            "path_distance_by_env": torch.zeros(
                num_imagined_envs, device=device, dtype=torch.float64
            ),
            "moved_steps": scalar(),
            "stationary_steps": scalar(),
            "invalid_jumps": scalar(),
            "closer_steps": scalar(),
            "farther_steps": scalar(),
            "same_distance_steps": scalar(),
            "goal_progress": scalar(),
            "goal_distance_samples": scalar(),
            "goal_distance_sum": scalar(),
            "start_goal_distance": None,
            "min_goal_distance": torch.full(
                (), float("inf"), device=device, dtype=torch.float64
            ),
            "forward_attempts": scalar(),
            "forward_blocked": scalar(),
            "forward_into_wall": scalar(),
            "forward_into_layout_wall": scalar(),
            "imagined_wall_not_in_layout": scalar(),
            "forward_into_door": scalar(),
            "forward_into_key": scalar(),
            "forward_blocked_with_free_front": scalar(),
            "toggle_attempts": scalar(),
            "toggle_at_door": scalar(),
            "door_open_events": scalar(),
            "pickup_attempts": scalar(),
            "pickup_at_key": scalar(),
            "pickup_events": scalar(),
            "goal_events": scalar(),
            "agent_lost": scalar(),
            "nonzero_reward_events": scalar(),
            "novel_position_events": scalar(),
            "global_count_reward_events": scalar(),
            "manhattan_closer_events": scalar(),
            "manhattan_farther_events": scalar(),
            "rewarded_pickup_events": scalar(),
            "rewarded_door_events": scalar(),
            "extrinsic_reward_sum": scalar(),
            "position_novelty_reward_sum": scalar(),
            "global_count_reward_sum": scalar(),
            "manhattan_progress_reward_sum": scalar(),
            "key_pickup_reward_sum": scalar(),
            "door_open_reward_sum": scalar(),
            "main_progress_reward_sum": scalar(),
            "new_best_progress_count": scalar(),
            "new_best_reward_sum": scalar(),
            "goal_region_action_progress_count": scalar(),
            "goal_region_entry_events": scalar(),
            "goal_region_reward_sum": scalar(),
            "goal_region_key_pickup_count": scalar(),
            "goal_region_key_pickup_reward_sum": scalar(),
            "goal_region_door_open_count": scalar(),
            "goal_region_door_open_reward_sum": scalar(),
            "progress_milestone_count": scalar(),
            "progress_milestone_reward_sum": scalar(),
            "critical_door_crossing_count": scalar(),
            "critical_door_crossing_reward_sum": scalar(),
            "shaping_reward_sum": scalar(),
            "reward_sum": scalar(),
            "reward_min": torch.full(
                (), float("inf"), device=device, dtype=torch.float64
            ),
            "reward_max": torch.full(
                (), float("-inf"), device=device, dtype=torch.float64
            ),
        }

    update_diagnostics = new_update_diagnostics()

    def distance_at(agent_positions):
        return goal_distance_map[
            agent_positions[:, 0].long(), agent_positions[:, 1].long()
        ]

    def report_update_diagnostics(update_metrics, end_positions):
        nonlocal update_diagnostics
        update_index = int(update_metrics.get("update_count", 0))
        should_log_update = (
            (policy_diagnostics and update_index % diagnostics_every_updates == 0)
            or (sub_run is not None and update_index % wandb_log_every_updates == 0)
        )
        if not should_log_update:
            update_diagnostics = new_update_diagnostics()
            return
        diag = update_diagnostics
        transitions = max(int(diag["transitions"]), 1)

        def number(name):
            return float(diag[name].detach().cpu().item())

        end_distances = distance_at(end_positions)
        valid_end = torch.isfinite(end_distances) & (end_distances >= 0)
        end_distance = (
            float(end_distances[valid_end].float().mean().detach().cpu().item())
            if bool(valid_end.any())
            else float("nan")
        )
        start_distance = (
            float(diag["start_goal_distance"])
            if diag["start_goal_distance"] is not None
            else float("nan")
        )
        min_distance = number("min_goal_distance")
        if not np.isfinite(min_distance):
            min_distance = float("nan")
        moved_steps = number("moved_steps")
        forward_attempts = number("forward_attempts")
        forward_blocked = number("forward_blocked")
        goal_events = number("goal_events")
        reward_sum = number("reward_sum")
        extrinsic_reward_sum = number("extrinsic_reward_sum")
        shaping_reward_sum = number("shaping_reward_sum")
        reward_min = number("reward_min")
        reward_max = number("reward_max")
        move_rate = moved_steps / transitions
        path_per_env = number("path_distance") / float(num_imagined_envs)
        path_by_env = diag["path_distance_by_env"].detach().cpu().numpy()
        path_min = float(path_by_env.min())
        path_max = float(path_by_env.max())
        progress_samples = max(number("goal_distance_samples"), 1.0)
        goal_distance_mean = number("goal_distance_sum") / progress_samples
        mean_progress = number("goal_progress") / progress_samples
        blocked_rate = forward_blocked / max(forward_attempts, 1.0)
        action_counts = diag["action_counts"].detach().cpu().tolist()
        action_text = ", ".join(
            f"{name}={int(count)}({int(count) / transitions:.1%})"
            for name, count in zip(COMPACT_ACTION_NAMES, action_counts)
        )

        reasons = []
        if goal_events == 0:
            reasons.append("goal_not_reached")
            if move_rate < 0.05:
                reasons.append("agent_mostly_stationary")
            elif mean_progress <= 0.0:
                reasons.append("no_average_goal_progress")
            else:
                reasons.append("progress_but_no_goal_hit")
            if forward_attempts > 0 and blocked_rate >= 0.5:
                reasons.append("most_forward_actions_blocked")
            if number("forward_blocked_with_free_front") > 0:
                reasons.append("wm_forward_no_move_on_free_cell")
            if number("invalid_jumps") > 0:
                reasons.append("wm_agent_jump_gt_1_cell")
            if use_main_dense_reward:
                if number("main_progress_reward_sum") != 0.0:
                    reasons.append("main_bfs_progress_reward_observed")
                if number("goal_region_entry_events") > 0:
                    reasons.append("main_goal_region_reward_observed")
                reasons.append("main_dense_reward_active")
            else:
                if key_pickup_reward != 0.0 and number("rewarded_pickup_events") > 0:
                    reasons.append("key_pickup_reward_observed")
                elif number("pickup_events") > 0 and key_pickup_reward == 0.0:
                    reasons.append("pickup_has_zero_reward")
                if door_open_reward != 0.0 and number("rewarded_door_events") > 0:
                    reasons.append("door_open_reward_observed")
                elif number("door_open_events") > 0 and door_open_reward == 0.0:
                    reasons.append("door_open_has_zero_reward")
                if (
                    position_novelty_reward != 0.0
                    and number("novel_position_events") > 0
                ):
                    reasons.append("position_novelty_reward_observed")
                if (
                    global_position_count_beta != 0.0
                    and number("global_count_reward_events") > 0
                ):
                    reasons.append("global_count_reward_observed")
                if (
                    manhattan_progress_reward_scale != 0.0
                    and number("manhattan_closer_events") > 0
                ):
                    reasons.append("manhattan_progress_reward_observed")
                if reward_shaping_enabled:
                    reasons.append("auxiliary_reward_active")
                else:
                    reasons.append("sparse_goal_only_reward")
        else:
            reasons.append("goal_reward_observed")

        metrics = {
            # Main-workspace-compatible rollout schema.
            "time_step": int(time_step),
            "update": update_index,
            "loss": float(update_metrics.get("loss", 0.0) or 0.0),
            "grad_norm": float(update_metrics.get("grad_norm", 0.0) or 0.0),
            "actor_delta": float(
                update_metrics.get("actor_parameter_delta", 0.0) or 0.0
            ),
            "critic_delta": float(
                update_metrics.get("critic_parameter_delta", 0.0) or 0.0
            ),
            "actor_loss": float(update_metrics.get("actor_loss", 0.0) or 0.0),
            "critic_loss": float(update_metrics.get("critic_loss", 0.0) or 0.0),
            "entropy": float(update_metrics.get("entropy", 0.0) or 0.0),
            "train/reward_per_step": reward_sum / transitions,
            "reward_min": reward_min if np.isfinite(reward_min) else None,
            "reward_max": reward_max if np.isfinite(reward_max) else None,
            "success": int(goal_events),
            "agent_lost": int(number("agent_lost")),
            "key_ach": number("pickup_events"),
            "door_ach": number("door_open_events"),
            "new_best_progress_count": number("new_best_progress_count"),
            "new_best_reward_sum": number("new_best_reward_sum"),
            "goal_region_entry_count": number("goal_region_entry_events"),
            "goal_region_reward_sum": number("goal_region_reward_sum"),
            "goal_region_key_pickup_count": number("goal_region_key_pickup_count"),
            "goal_region_key_pickup_reward_sum": number("goal_region_key_pickup_reward_sum"),
            "goal_region_door_open_count": number("goal_region_door_open_count"),
            "goal_region_door_open_reward_sum": number("goal_region_door_open_reward_sum"),
            "progress_milestone_count": number("progress_milestone_count"),
            "progress_milestone_reward_sum": number("progress_milestone_reward_sum"),
            "critical_door_crossing_count": number("critical_door_crossing_count"),
            "critical_door_crossing_reward_sum": number("critical_door_crossing_reward_sum"),
            "topology_subgoal_completion_count": 0,
            "topology_subgoal_stage_mean": 0.0,
            "distance_metric": "bfs" if use_main_dense_reward else "legacy",
            "goal_dist_mean": goal_distance_mean,
            "goal_dist_min": min_distance,
            "subgoal_dist_mean": goal_distance_mean,
            "subgoal_dist_min": min_distance,
            "diagnostic/path_distance_per_env": path_per_env,
            "diagnostic/path_distance_per_env_min": path_min,
            "diagnostic/path_distance_per_env_max": path_max,
            "diagnostic/move_rate": move_rate,
            "diagnostic/stationary_rate": number("stationary_steps") / transitions,
            "diagnostic/invalid_jump_count": number("invalid_jumps"),
            "diagnostic/goal_distance_start": start_distance,
            "diagnostic/goal_distance_end": end_distance,
            "diagnostic/goal_distance_min": min_distance,
            "diagnostic/mean_goal_progress_per_step": mean_progress,
            "diagnostic/closer_rate": number("closer_steps") / progress_samples,
            "diagnostic/farther_rate": number("farther_steps") / progress_samples,
            "diagnostic/forward_blocked_rate": blocked_rate,
            "diagnostic/forward_into_wall": number("forward_into_wall"),
            "diagnostic/forward_into_layout_wall": number(
                "forward_into_layout_wall"
            ),
            "diagnostic/imagined_wall_not_in_layout": number(
                "imagined_wall_not_in_layout"
            ),
            "diagnostic/forward_into_door": number("forward_into_door"),
            "diagnostic/forward_into_key": number("forward_into_key"),
            "diagnostic/forward_blocked_with_free_front": number(
                "forward_blocked_with_free_front"
            ),
            "diagnostic/toggle_at_door": number("toggle_at_door"),
            "diagnostic/door_open_events": number("door_open_events"),
            "diagnostic/pickup_at_key": number("pickup_at_key"),
            "diagnostic/pickup_events": number("pickup_events"),
            "diagnostic/goal_events": goal_events,
            "diagnostic/nonzero_reward_events": number("nonzero_reward_events"),
            "diagnostic/novel_position_events": number("novel_position_events"),
            "diagnostic/global_count_reward_events": number(
                "global_count_reward_events"
            ),
            "diagnostic/manhattan_closer_events": number(
                "manhattan_closer_events"
            ),
            "diagnostic/manhattan_farther_events": number(
                "manhattan_farther_events"
            ),
            "diagnostic/global_unique_positions": float(
                (global_position_counts > 0).sum().detach().cpu().item()
            ),
            "diagnostic/rewarded_pickup_events": number("rewarded_pickup_events"),
            "diagnostic/rewarded_door_events": number("rewarded_door_events"),
            "diagnostic/extrinsic_reward_sum": extrinsic_reward_sum,
            "diagnostic/position_novelty_reward_sum": number(
                "position_novelty_reward_sum"
            ),
            "diagnostic/global_count_reward_sum": number(
                "global_count_reward_sum"
            ),
            "diagnostic/manhattan_progress_reward_sum": number(
                "manhattan_progress_reward_sum"
            ),
            "diagnostic/key_pickup_reward_sum": number("key_pickup_reward_sum"),
            "diagnostic/door_open_reward_sum": number("door_open_reward_sum"),
            "diagnostic/main_progress_reward_sum": number(
                "main_progress_reward_sum"
            ),
            "diagnostic/new_best_progress_count": number(
                "new_best_progress_count"
            ),
            "diagnostic/new_best_reward_sum": number("new_best_reward_sum"),
            "diagnostic/goal_region_action_progress_count": number(
                "goal_region_action_progress_count"
            ),
            "diagnostic/goal_region_entry_count": number(
                "goal_region_entry_events"
            ),
            "diagnostic/goal_region_reward_sum": number(
                "goal_region_reward_sum"
            ),
            "diagnostic/goal_region_key_pickup_count": number(
                "goal_region_key_pickup_count"
            ),
            "diagnostic/goal_region_key_pickup_reward_sum": number(
                "goal_region_key_pickup_reward_sum"
            ),
            "diagnostic/goal_region_door_open_count": number(
                "goal_region_door_open_count"
            ),
            "diagnostic/goal_region_door_open_reward_sum": number(
                "goal_region_door_open_reward_sum"
            ),
            "diagnostic/progress_milestone_count": number(
                "progress_milestone_count"
            ),
            "diagnostic/progress_milestone_reward_sum": number(
                "progress_milestone_reward_sum"
            ),
            "diagnostic/critical_door_crossing_count": number(
                "critical_door_crossing_count"
            ),
            "diagnostic/critical_door_crossing_reward_sum": number(
                "critical_door_crossing_reward_sum"
            ),
            "diagnostic/shaping_reward_sum": shaping_reward_sum,
            "diagnostic/reward_sum": reward_sum,
        }
        for name, count in zip(COMPACT_ACTION_NAMES, action_counts):
            metrics[f"diagnostic/action_fraction/{name}"] = int(count) / transitions
            metrics[f"action_ratio/{name}"] = int(count) / transitions

        if policy_diagnostics and update_index % diagnostics_every_updates == 0:
            print(
                f"[PolicyDiag][update={update_index} timestep={time_step}] "
                f"reward(total/extrinsic/shaping)="
                f"{reward_sum:.4f}/{extrinsic_reward_sum:.4f}/"
                f"{shaping_reward_sum:.4f}, goal_events={int(goal_events)}, "
                f"moved={int(moved_steps)}/{transitions} ({move_rate:.1%}), "
                f"path_distance/env(mean/min/max)="
                f"{path_per_env:.2f}/{path_min:.0f}/{path_max:.0f}, "
                f"goal_distance(start/end/min)="
                f"{start_distance:.2f}/{end_distance:.2f}/{min_distance:.2f}, "
                f"mean_progress/step={mean_progress:.4f}, "
                f"forward_blocked={int(forward_blocked)}/{int(forward_attempts)} "
                f"({blocked_rate:.1%}), jumps>1={int(number('invalid_jumps'))}"
            )
            print(f"[PolicyDiag][actions] {action_text}")
            print(
                "[PolicyDiag][interactions] "
                f"blocked_by wall/closed-door/key="
                f"{int(number('forward_into_wall'))}/"
                f"{int(number('forward_into_door'))}/"
                f"{int(number('forward_into_key'))}; "
                f"layout-wall/hallucinated-wall="
                f"{int(number('forward_into_layout_wall'))}/"
                f"{int(number('imagined_wall_not_in_layout'))}; "
                f"free-front-but-no-move="
                f"{int(number('forward_blocked_with_free_front'))}; "
                f"toggle_at_door/opened="
                f"{int(number('toggle_at_door'))}/{int(number('door_open_events'))}; "
                f"pickup_at_key/succeeded="
                f"{int(number('pickup_at_key'))}/{int(number('pickup_events'))}"
            )
            print(
                "[PolicyDiag][shaping] "
                f"novel_positions={int(number('novel_position_events'))}, "
                f"global_count_events="
                f"{int(number('global_count_reward_events'))}, "
                f"rewarded_pickups={int(number('rewarded_pickup_events'))}, "
                f"rewarded_doors={int(number('rewarded_door_events'))}; "
                f"reward(global/manhattan/legacy_position/key/door)="
                f"{number('global_count_reward_sum'):.4f}/"
                f"{number('manhattan_progress_reward_sum'):.4f}/"
                f"{number('position_novelty_reward_sum'):.4f}/"
                f"{number('key_pickup_reward_sum'):.4f}/"
                f"{number('door_open_reward_sum'):.4f}; "
                f"main_bfs/goal_region="
                f"{number('main_progress_reward_sum'):.4f}/"
                f"{number('goal_region_reward_sum'):.4f} "
                f"(entries={int(number('goal_region_entry_events'))})"
            )
            print(f"[PolicyDiag][reward evidence] {', '.join(reasons)}")
        if sub_run is not None and update_index % wandb_log_every_updates == 0:
            sub_run.log(metrics, step=time_step)
        update_diagnostics = new_update_diagnostics()

    def log_ppo_update(update_metrics):
        if (
            sub_run is not None
            and update_metrics.get("updated", False)
            and int(update_metrics["update_count"]) % wandb_log_every_updates == 0
        ):
            sub_run.log(
                {
                    "ppo/loss": update_metrics["loss"],
                    "ppo/actor_loss": update_metrics["actor_loss"],
                    "ppo/critic_loss": update_metrics["critic_loss"],
                    "ppo/entropy": update_metrics["entropy"],
                    "ppo/grad_norm": update_metrics["grad_norm"],
                    "ppo/parameter_delta": update_metrics["parameter_delta"],
                    "ppo/actor_parameter_delta": update_metrics[
                        "actor_parameter_delta"
                    ],
                    "ppo/critic_parameter_delta": update_metrics[
                        "critic_parameter_delta"
                    ],
                    "ppo/actor_update_enabled": int(
                        update_metrics["actor_update_enabled"]
                    ),
                    "ppo/reward_signal_in_rollout": int(
                        update_metrics["reward_signal_in_rollout"]
                    ),
                    "ppo/rollout_size": update_metrics["rollout_size"],
                    "ppo/update_count": update_metrics["update_count"],
                },
                step=time_step,
            )

    # Each loop advances B independent trajectories by one imagined step.
    while time_step < max_training_timesteps:
        next_update_index = ppo_agent.update_count + 1
        collect_update_diagnostics = (
            (policy_diagnostics and next_update_index % diagnostics_every_updates == 0)
            or (
                sub_run is not None
                and next_update_index % wandb_log_every_updates == 0
            )
        )
        previous_states = states
        previous_agent_positions = utils.get_agent_position_torch(states)
        previous_manhattan_distances = torch.abs(
            previous_agent_positions - goal_positions
        ).sum(dim=1)
        previous_goal_distances = distance_at(previous_agent_positions)
        (
            front_objects,
            front_states,
            front_y,
            front_x,
            valid_front,
        ) = minigrid_front_cells(states, previous_agent_positions)
        state_norm = utils.normalize_obs(
            states.clone(), hparams_world_model.obs_norm_values
        ).reshape(num_imagined_envs, -1)
        state_norm = append_inventory_to_policy_state(state_norm, carrying_key)
        (
            actions,
            state_buffer,
            action_buffer,
            action_logprobs,
            state_values,
        ) = ppo_agent.select_action_batch(state_norm)

        # no_grad keeps the resulting state mutable so completed batch slots
        # can be reset in place. inference_mode tensors forbid that update.
        prev_carrying_key = carrying_key.clone()
        with torch.no_grad():
            states, carrying_key = imagined_minigrid_step_batch(
                model,
                states,
                actions,
                carrying_key,
                hparams_world_model.attention_mask_size,
                hparams_world_model.valid_values_obj,
                hparams_world_model.valid_values_color,
                hparams_world_model.valid_values_state,
                agent_positions=previous_agent_positions,
            )

        lava_terminated = (actions == 2) & valid_front & (front_objects == 9)
        agent_present = (states[:, 0] == 10).flatten(1).any(dim=1)
        agent_lost = ~agent_present
        # A missing agent is an invalid WM transition, not an environment
        # outcome. This unconditional tensor selection avoids synchronizing
        # CUDA for a Python ``any`` on every imagined step.
        states = torch.where(
            agent_lost[:, None, None, None], previous_states, states
        )
        invalid_transitions = agent_lost & ~lava_terminated
        valid_transitions = ~invalid_transitions

        episode_steps += 1
        agent_positions = utils.get_agent_position_torch(states)
        position_y = agent_positions[:, 0].long()
        position_x = agent_positions[:, 1].long()
        novel_positions = ~visited_positions[batch_ids, position_y, position_x]
        visited_positions[batch_ids, position_y, position_x] = True
        step_distances = torch.abs(
            agent_positions - previous_agent_positions
        ).sum(dim=1)
        current_manhattan_distances = torch.abs(
            agent_positions - goal_positions
        ).sum(dim=1)
        manhattan_progress = (
            previous_manhattan_distances - current_manhattan_distances
        ).clamp(-1, 1)
        flat_positions = position_y * grid_width + position_x
        destination_counts = global_position_counts[flat_positions]
        global_count_rewards = (
            (step_distances > 0).float()
            * global_position_count_beta
            / torch.sqrt(destination_counts + 1.0)
        )
        global_position_counts.scatter_add_(
            0,
            flat_positions[valid_transitions],
            torch.ones_like(
                flat_positions[valid_transitions], dtype=torch.float32
            ),
        )
        manhattan_progress_rewards = (
            manhattan_progress.float() * manhattan_progress_reward_scale
        )
        current_goal_distances = distance_at(agent_positions)
        valid_goal_distance = valid_transitions & (
            torch.isfinite(previous_goal_distances)
            & torch.isfinite(current_goal_distances)
            & (previous_goal_distances >= 0)
            & (current_goal_distances >= 0)
        )
        goal_progress = torch.zeros_like(previous_goal_distances)
        goal_progress[valid_goal_distance] = (
            previous_goal_distances[valid_goal_distance]
            - current_goal_distances[valid_goal_distance]
        )
        if collect_update_diagnostics and update_diagnostics["transitions"] == 0:
            valid_start = valid_transitions & torch.isfinite(previous_goal_distances) & (
                previous_goal_distances >= 0
            )
            if bool(valid_start.any()):
                update_diagnostics["start_goal_distance"] = float(
                    previous_goal_distances[valid_start]
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
        if collect_update_diagnostics:
            update_diagnostics["transitions"] += int(valid_transitions.sum().item())
            update_diagnostics["agent_lost"] += agent_lost.double().sum()
            update_diagnostics["action_counts"] += torch.bincount(
                actions[valid_transitions].long(), minlength=action_dim
            )
            update_diagnostics["path_distance"] += step_distances[valid_transitions].double().sum()
            update_diagnostics["path_distance_by_env"] += step_distances.double() * valid_transitions.double()
            update_diagnostics["moved_steps"] += ((step_distances > 0) & valid_transitions).double().sum()
            update_diagnostics["stationary_steps"] += ((step_distances == 0) & valid_transitions).double().sum()
            update_diagnostics["invalid_jumps"] += ((step_distances > 1) & valid_transitions).double().sum()
            update_diagnostics["closer_steps"] += ((goal_progress > 0) & valid_transitions).double().sum()
            update_diagnostics["farther_steps"] += ((goal_progress < 0) & valid_transitions).double().sum()
            update_diagnostics["same_distance_steps"] += ((goal_progress == 0) & valid_transitions).double().sum()
            update_diagnostics["goal_progress"] += goal_progress[valid_transitions].double().sum()
            update_diagnostics["goal_distance_samples"] += valid_goal_distance.double().sum()
            update_diagnostics["goal_distance_sum"] += (
                current_goal_distances[valid_goal_distance].double().sum()
            )
            valid_or_inf = torch.where(
                valid_transitions & torch.isfinite(current_goal_distances) & (current_goal_distances >= 0),
                current_goal_distances.double(),
                torch.full_like(current_goal_distances, float("inf"), dtype=torch.float64),
            )
            update_diagnostics["min_goal_distance"] = torch.minimum(
                update_diagnostics["min_goal_distance"], valid_or_inf.min()
            )

        forward = actions == 2
        blocked_forward = forward & (step_distances == 0)
        toggle = actions == 4
        pickup = actions == 3
        closed_door_ahead = (front_objects == 4) & (front_states != 0)
        next_front_states = states[batch_ids, 2, front_y, front_x].long()
        door_opened = (
            toggle
            & valid_transitions
            & valid_front
            & (front_objects == 4)
            & (front_states != 0)
            & (next_front_states == 0)
        )
        if collect_update_diagnostics:
            layout_front_objects = initial_object_map[front_y, front_x].long()
            physically_blocked_ahead = (
                (front_objects == 2) | closed_door_ahead | (front_objects == 5)
                | (front_objects == 6) | (front_objects == 7) | ~valid_front
            )
            update_diagnostics["forward_attempts"] += (forward & valid_transitions).double().sum()
            update_diagnostics["forward_blocked"] += (blocked_forward & valid_transitions).double().sum()
            update_diagnostics["forward_into_wall"] += (
                blocked_forward & valid_transitions & (front_objects == 2)
            ).double().sum()
            update_diagnostics["forward_into_layout_wall"] += (
                blocked_forward & valid_transitions & (layout_front_objects == 2)
            ).double().sum()
            update_diagnostics["imagined_wall_not_in_layout"] += (
                blocked_forward & valid_transitions & (front_objects == 2)
                & (layout_front_objects != 2)
            ).double().sum()
            update_diagnostics["forward_into_door"] += (
                blocked_forward & valid_transitions & closed_door_ahead
            ).double().sum()
            update_diagnostics["forward_into_key"] += (
                blocked_forward & valid_transitions & (front_objects == 5)
            ).double().sum()
            update_diagnostics["forward_blocked_with_free_front"] += (
                blocked_forward & valid_transitions & ~physically_blocked_ahead
            ).double().sum()
            update_diagnostics["toggle_attempts"] += (toggle & valid_transitions).double().sum()
            update_diagnostics["toggle_at_door"] += (
                toggle & valid_transitions & (front_objects == 4)
            ).double().sum()
            update_diagnostics["pickup_attempts"] += (pickup & valid_transitions).double().sum()
            update_diagnostics["pickup_at_key"] += (
                pickup & valid_transitions & (front_objects == 5)
            ).double().sum()
            update_diagnostics["door_open_events"] += door_opened.double().sum()
        newly_rewarded_doors = door_opened & ~rewarded_door_positions[
            batch_ids, front_y, front_x
        ]
        rewarded_door_positions[batch_ids, front_y, front_x] |= door_opened
        reached_goal = torch.all(agent_positions == goal_positions, dim=1)
        lava_terminated = forward & valid_front & (front_objects == 9)
        just_picked_up = (
            valid_transitions & (prev_carrying_key < 0) & (carrying_key >= 0)
        )
        newly_rewarded_pickups = just_picked_up & ~key_rewarded
        key_rewarded |= just_picked_up
        position_rewards = novel_positions.float() * position_novelty_reward
        pickup_rewards = newly_rewarded_pickups.float() * key_pickup_reward
        door_rewards = newly_rewarded_doors.float() * door_open_reward
        shaping_rewards = (
            global_count_rewards
            + manhattan_progress_rewards
            + position_rewards
            + pickup_rewards
            + door_rewards
        )
        if use_main_dense_reward:
            rewards, goal_region_seen, main_reward_events = main_dense_rewards(
                previous_states,
                states,
                actions,
                goal_distance_map,
                goal_region_mask,
                goal_region_seen,
                reached_goal,
                lava_terminated,
                best_goal_distances=best_goal_distances,
                initial_goal_distances=initial_goal_distances,
                progress_milestone_seen=progress_milestone_seen,
                previous_carrying_tokens=prev_carrying_key + 1,
                current_carrying_tokens=carrying_key + 1,
                door_topology=door_topology,
                pending_door_ids=pending_door_ids,
                pending_door_origins=pending_door_origins,
                rewarded_door_crossings=rewarded_door_crossings,
                rewarded_goal_region_key_positions=rewarded_goal_region_key_positions,
                rewarded_goal_region_door_positions=rewarded_goal_region_door_positions,
                transition_valid=valid_transitions,
                **main_reward,
            )
            best_goal_distances = main_reward_events["new_best_goal_distance"]
            progress_milestone_seen = main_reward_events["progress_milestone_seen"]
            pending_door_ids = main_reward_events["pending_door_ids"]
            pending_door_origins = main_reward_events["pending_door_origins"]
            rewarded_door_crossings = main_reward_events["rewarded_door_crossings"]
            rewarded_goal_region_key_positions = main_reward_events[
                "rewarded_goal_region_key_positions"
            ]
            rewarded_goal_region_door_positions = main_reward_events[
                "rewarded_goal_region_door_positions"
            ]
            goal_rewards = reached_goal.float() * float(
                main_reward["success_reward"]
            )
            shaping_rewards = (
                torch.full_like(rewards, float(main_reward["step_penalty"]))
                + main_reward_events["progress_reward"]
                + main_reward_events["goal_region_reward"]
                + main_reward_events["goal_region_key_pickup_reward"]
                + main_reward_events["goal_region_door_open_reward"]
                + main_reward_events["progress_milestone_reward"]
                + main_reward_events["critical_door_crossing_reward"]
            )
            shaping_rewards = torch.where(
                reached_goal | lava_terminated,
                torch.zeros_like(shaping_rewards),
                shaping_rewards,
            )
            if collect_update_diagnostics: update_diagnostics["main_progress_reward_sum"] += (
                main_reward_events["progress_reward"][valid_transitions].double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["new_best_progress_count"] += (
                (main_reward_events["new_best_reward"][valid_transitions] > 0)
                .double()
                .sum()
            )
            if collect_update_diagnostics: update_diagnostics["new_best_reward_sum"] += (
                main_reward_events["new_best_reward"][valid_transitions].double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_action_progress_count"] += (
                main_reward_events["goal_region_action_transition"][valid_transitions]
                .double()
                .sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_entry_events"] += (
                main_reward_events["goal_region_entered"][valid_transitions].double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_reward_sum"] += (
                main_reward_events["goal_region_reward"][valid_transitions].double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_key_pickup_count"] += (
                main_reward_events["goal_region_key_picked_up"][valid_transitions]
                .double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_key_pickup_reward_sum"] += (
                main_reward_events["goal_region_key_pickup_reward"][valid_transitions]
                .double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_door_open_count"] += (
                main_reward_events["goal_region_door_opened"][valid_transitions]
                .double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["goal_region_door_open_reward_sum"] += (
                main_reward_events["goal_region_door_open_reward"][valid_transitions]
                .double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["progress_milestone_count"] += (
                main_reward_events["progress_milestone_crossed"][valid_transitions]
                .double()
                .sum()
            )
            if collect_update_diagnostics: update_diagnostics["progress_milestone_reward_sum"] += (
                main_reward_events["progress_milestone_reward"][valid_transitions]
                .double()
                .sum()
            )
            if collect_update_diagnostics: update_diagnostics["critical_door_crossing_count"] += (
                main_reward_events["critical_door_crossed"][valid_transitions]
                .double().sum()
            )
            if collect_update_diagnostics: update_diagnostics["critical_door_crossing_reward_sum"] += (
                main_reward_events["critical_door_crossing_reward"][valid_transitions]
                .double().sum()
            )
        else:
            goal_rewards = torch.where(
                reached_goal,
                1.0 - 0.9 * episode_steps.float() / float(max_ep_len),
                torch.zeros_like(episode_rewards),
            )
            rewards = goal_rewards + shaping_rewards + step_penalty
        rewards[invalid_transitions] = 0.0
        goal_rewards[invalid_transitions] = 0.0
        shaping_rewards[invalid_transitions] = 0.0
        if collect_update_diagnostics: update_diagnostics["pickup_events"] += just_picked_up[valid_transitions].double().sum()
        if collect_update_diagnostics: update_diagnostics["goal_events"] += reached_goal[valid_transitions].double().sum()
        if collect_update_diagnostics: update_diagnostics["nonzero_reward_events"] += (rewards[valid_transitions] != 0).double().sum()
        if collect_update_diagnostics and not use_main_dense_reward:
            update_diagnostics["novel_position_events"] += (
                novel_positions.double().sum()
            )
            update_diagnostics["global_count_reward_events"] += (
                (global_count_rewards > 0).double().sum()
            )
            update_diagnostics["manhattan_closer_events"] += (
                (manhattan_progress > 0).double().sum()
            )
            update_diagnostics["manhattan_farther_events"] += (
                (manhattan_progress < 0).double().sum()
            )
            update_diagnostics["rewarded_pickup_events"] += (
                newly_rewarded_pickups.double().sum()
            )
            update_diagnostics["rewarded_door_events"] += (
                newly_rewarded_doors.double().sum()
            )
        if collect_update_diagnostics: update_diagnostics["extrinsic_reward_sum"] += goal_rewards[valid_transitions].double().sum()
        if collect_update_diagnostics and not use_main_dense_reward:
            update_diagnostics["position_novelty_reward_sum"] += (
                position_rewards.double().sum()
            )
            update_diagnostics["global_count_reward_sum"] += (
                global_count_rewards.double().sum()
            )
            update_diagnostics["manhattan_progress_reward_sum"] += (
                manhattan_progress_rewards.double().sum()
            )
            update_diagnostics["key_pickup_reward_sum"] += (
                pickup_rewards.double().sum()
            )
            update_diagnostics["door_open_reward_sum"] += door_rewards.double().sum()
        if collect_update_diagnostics: update_diagnostics["shaping_reward_sum"] += shaping_rewards[valid_transitions].double().sum()
        if collect_update_diagnostics: update_diagnostics["reward_sum"] += rewards[valid_transitions].double().sum()
        if collect_update_diagnostics and bool(valid_transitions.any()):
            valid_rewards = rewards[valid_transitions].double()
            update_diagnostics["reward_min"] = torch.minimum(
                update_diagnostics["reward_min"], valid_rewards.min()
            )
            update_diagnostics["reward_max"] = torch.maximum(
                update_diagnostics["reward_max"], valid_rewards.max()
            )
        truncated = episode_steps >= max_ep_len
        dones = (
            reached_goal
            | (lava_terminated & use_main_dense_reward)
            | truncated
            | agent_lost
        )
        episode_rewards += rewards
        episode_extrinsic_rewards += goal_rewards
        episode_shaping_rewards += shaping_rewards

        ppo_agent.save_buffer_batch(
            state_buffer,
            action_buffer,
            action_logprobs,
            state_values,
            rewards,
            dones,
            valid_mask=valid_transitions,
        )
        time_step += num_imagined_envs
        last_dones = dones.clone()

        if ppo_agent.buffer.has_at_least_transitions(update_timestep):
            next_state_norm = utils.normalize_obs(
                states.clone(), hparams_world_model.obs_norm_values
            ).reshape(num_imagined_envs, -1)
            next_state_norm = append_inventory_to_policy_state(
                next_state_norm, carrying_key
            )
            bootstrap_values = ppo_agent.estimate_old_values_batch(next_state_norm)
            bootstrap_values[last_dones.detach().cpu()] = 0.0
            update_metrics = ppo_agent.update(
                bootstrap_value=bootstrap_values,
                collect_metrics=collect_update_diagnostics,
            )
            log_ppo_update(update_metrics)
            report_update_diagnostics(update_metrics, agent_positions)
            if best_policy_selection_enabled and time_step >= next_selection_timestep:
                evaluate_and_select_policy(time_step)
                while next_selection_timestep <= time_step:
                    next_selection_timestep += selection_eval_freq

        completed = torch.nonzero(
            (
                reached_goal
                | (lava_terminated & use_main_dense_reward)
                | truncated
            )
            & valid_transitions,
            as_tuple=False,
        ).reshape(-1)
        reset_slots = torch.nonzero(dones, as_tuple=False).reshape(-1)
        if completed.numel() > 0:
            completed_rewards = episode_rewards[completed].detach().cpu().tolist()
            completed_extrinsic_rewards = (
                episode_extrinsic_rewards[completed].detach().cpu().tolist()
            )
            completed_shaping_rewards = (
                episode_shaping_rewards[completed].detach().cpu().tolist()
            )
            completed_steps = episode_steps[completed].detach().cpu().tolist()
            completed_successes = reached_goal[completed].detach().cpu().int().tolist()
            for ep_reward, ep_extrinsic, ep_shaping, ep_steps, ep_success in zip(
                completed_rewards,
                completed_extrinsic_rewards,
                completed_shaping_rewards,
                completed_steps,
                completed_successes,
            ):
                print_running_reward += float(ep_reward)
                print_running_extrinsic_reward += float(ep_extrinsic)
                print_running_shaping_reward += float(ep_shaping)
                print_running_steps += int(ep_steps)
                print_running_successes += int(ep_success)
                print_running_episodes += 1
                recent_rewards.append(float(ep_reward))
                recent_extrinsic_rewards.append(float(ep_extrinsic))
                recent_shaping_rewards.append(float(ep_shaping))
                recent_steps.append(int(ep_steps))
                recent_successes.append(int(ep_success))
                i_episode += 1

            rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
            rolling_avg_extrinsic_reward = (
                sum(recent_extrinsic_rewards) / len(recent_extrinsic_rewards)
            )
            rolling_avg_shaping_reward = (
                sum(recent_shaping_rewards) / len(recent_shaping_rewards)
            )
            rolling_avg_steps = sum(recent_steps) / len(recent_steps)
            rolling_success_rate = sum(recent_successes) / len(recent_successes)
            if (
                sub_run is not None
                and time_step >= next_episode_log_timestep
            ):
                sub_run.log(
                    {
                        "episode/reward": sum(completed_rewards) / len(completed_rewards),
                        "episode/extrinsic_reward": sum(completed_extrinsic_rewards)
                        / len(completed_extrinsic_rewards),
                        "episode/shaping_reward": sum(completed_shaping_rewards)
                        / len(completed_shaping_rewards),
                        "episode/success": sum(completed_successes) / len(completed_successes),
                        "episode/steps": sum(completed_steps) / len(completed_steps),
                        "episode/completed_count": len(completed_rewards),
                        "episode/index": i_episode,
                        "rolling/average_reward": rolling_avg_reward,
                        "rolling/average_extrinsic_reward": (
                            rolling_avg_extrinsic_reward
                        ),
                        "rolling/average_shaping_reward": rolling_avg_shaping_reward,
                        "rolling/success_rate": rolling_success_rate,
                        "rolling/average_episode_steps": rolling_avg_steps,
                        "rolling/window_episode_count": len(recent_rewards),
                        "train/episode_return_mean": rolling_avg_reward,
                        "train/episode_length_mean": rolling_avg_steps,
                        "train/episode_success_rate": rolling_success_rate,
                        "train/rolling_success_rate": rolling_success_rate,
                        "train/episodes_completed": i_episode,
                    },
                    step=time_step,
                )
                while next_episode_log_timestep <= time_step:
                    next_episode_log_timestep += episode_log_every_steps

        # Agent-lost slots are reset but are not counted as completed episodes.
        # All other imagined trajectories keep their current state/statistics.
        if reset_slots.numel() > 0:
            reset_states = torch.stack(
                [reset_imagined_state() for _ in range(reset_slots.numel())], dim=0
            )
            states[reset_slots] = reset_states
            carrying_key[reset_slots] = -1
            prev_carrying_key[reset_slots] = -1
            episode_rewards[reset_slots] = 0.0
            episode_extrinsic_rewards[reset_slots] = 0.0
            episode_shaping_rewards[reset_slots] = 0.0
            episode_steps[reset_slots] = 0
            visited_positions[reset_slots] = False
            reset_agent_positions = utils.get_agent_position_torch(
                states[reset_slots]
            )
            visited_positions[
                reset_slots,
                reset_agent_positions[:, 0].long(),
                reset_agent_positions[:, 1].long(),
            ] = True
            rewarded_door_positions[reset_slots] = False
            rewarded_goal_region_key_positions[reset_slots] = False
            rewarded_goal_region_door_positions[reset_slots] = False
            key_rewarded[reset_slots] = False
            goal_region_seen[reset_slots] = False
            progress_milestone_seen[reset_slots] = False
            pending_door_ids[reset_slots] = -1
            pending_door_origins[reset_slots] = -1
            rewarded_door_crossings[reset_slots] = False
            reset_goal_distances = goal_distance_map[
                reset_agent_positions[:, 0].long(), reset_agent_positions[:, 1].long()
            ]
            initial_goal_distances[reset_slots] = reset_goal_distances
            best_goal_distances[reset_slots] = reset_goal_distances

        if time_step >= next_print_timestep and print_running_episodes > 0:
            rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
            rolling_avg_extrinsic_reward = (
                sum(recent_extrinsic_rewards) / len(recent_extrinsic_rewards)
            )
            rolling_avg_steps = sum(recent_steps) / len(recent_steps)
            rolling_success_rate = sum(recent_successes) / len(recent_successes)
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_extrinsic_reward = (
                print_running_extrinsic_reward / print_running_episodes
            )
            print_avg_shaping_reward = (
                print_running_shaping_reward / print_running_episodes
            )
            print_avg_steps = print_running_steps / print_running_episodes
            print_success_rate = print_running_successes / print_running_episodes
            print(
                f"Episode : {i_episode} \t Timestep : {time_step} \t "
                f"Interval Shaped Reward : {print_avg_reward:.5f} \t "
                f"Extrinsic : {print_avg_extrinsic_reward:.5f} \t "
                f"Auxiliary : {print_avg_shaping_reward:.5f} \t "
                f"Interval Success : {print_success_rate:.1%} \t "
                f"Interval Steps : {print_avg_steps:.1f} \t "
                f"Rolling({len(recent_rewards)}) Shaped Reward : "
                f"{rolling_avg_reward:.5f} \t "
                f"Extrinsic : {rolling_avg_extrinsic_reward:.5f} \t "
                f"Success : {rolling_success_rate:.1%} \t "
                f"Steps : {rolling_avg_steps:.1f}"
            )
            print_running_reward = 0
            print_running_extrinsic_reward = 0
            print_running_shaping_reward = 0
            print_running_episodes = 0
            print_running_steps = 0
            print_running_successes = 0
            while next_print_timestep <= time_step:
                next_print_timestep += print_freq

        if time_step >= next_save_timestep:
            print("--------------------------------------------------------------------------------------------")
            save_path = (
                str(selection_paths["last"])
                if best_policy_selection_enabled
                else checkpoint_path
            )
            print("saving latest model at : " + save_path)
            ppo_agent.save(save_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            while next_save_timestep <= time_step:
                next_save_timestep += save_model_freq

    # Train once more on the partial fixed-size rollout at the budget boundary.
    if ppo_agent.buffer.transition_count() >= 2:
        next_state_norm = utils.normalize_obs(
            states.clone(), hparams_world_model.obs_norm_values
        ).reshape(num_imagined_envs, -1)
        next_state_norm = append_inventory_to_policy_state(
            next_state_norm, carrying_key
        )
        bootstrap_values = ppo_agent.estimate_old_values_batch(next_state_norm)
        bootstrap_values[last_dones.detach().cpu()] = 0.0
        update_metrics = ppo_agent.update(
            bootstrap_value=bootstrap_values,
            collect_metrics=True,
        )
        log_ppo_update(update_metrics)
        report_update_diagnostics(
            update_metrics, utils.get_agent_position_torch(states)
        )
        # A partial update may be the final policy change and must be scored.
        evaluate_and_select_policy(time_step)
    else:
        ppo_agent.buffer.clear()

    # Final policy is retained separately; canonical remains the best frozen-WM
    # deterministic policy for backward-compatible consumers.
    if best_policy_selection_enabled:
        ppo_agent.save(selection_paths["last"])
        # Covers an exact-budget final PPO update; duplicate timestamps are
        # intentionally ignored by the selector.
        evaluate_and_select_policy(time_step)
        write_policy_selection_manifest()
        print(f"Best policy saved at: {selection_paths['canonical']}")
        print(f"Last policy saved at: {selection_paths['last']}")
    else:
        ppo_agent.save(checkpoint_path)
        print(f"Final policy saved at: {checkpoint_path}")
    env.close()
    if use_wandb:
        sub_run.finish()
    if compute_regret: 
        return final_norm_regret


@hydra.main(version_base=None, config_path=str(SCRIPT_ROOT / "modelBased/config"), config_name="config")
def training_agent_real_env(cfg: DictConfig):
    run_training_real_env(cfg)

def _init_policy_wandb_run(cfg, default_project):
    """Initialize WandB from the user's login without embedding credentials."""
    ppo_cfg = cfg.PPO
    wandb_group, wandb_run_name, training_source = policy_wandb_identity(cfg)
    task_name = str(cfg.domains[str(cfg.domain)].task_name)
    wandb_dir = Path(
        str(getattr(getattr(cfg, "paths", None), "wandb", utils.WM_OUTPUTS_PATH / "wandb"))
    ).expanduser().resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.login()
    resolved_ppo = OmegaConf.to_container(ppo_cfg, resolve=True)
    init_kwargs = {
        "dir": str(wandb_dir),
        "project": str(getattr(ppo_cfg, "wandb_project", default_project)),
        "name": wandb_run_name,
        "group": wandb_group,
        "job_type": task_name,
        "tags": [training_source, task_name],
        "reinit": True,
        "config": {
            "domain": str(cfg.domain),
            "task_name": task_name,
            "layout_path": str(cfg.domains[str(cfg.domain)].layout_path),
            "env_path": str(ppo_cfg.env_path),
            "training_source": training_source,
            "policy_checkpoint": str(policy_checkpoint_path(cfg)),
            "wm_checkpoint": str(getattr(ppo_cfg, "checkpoint_path_wm", "")),
            "seed": int(getattr(ppo_cfg, "seed", 0)),
            "baseline": getattr(ppo_cfg, "wandb_baseline", None),
            "wm_seed": getattr(ppo_cfg, "wandb_wm_seed", None),
            "target": getattr(ppo_cfg, "wandb_target_name", task_name),
            "policy_seed": getattr(
                ppo_cfg, "wandb_policy_seed", int(getattr(ppo_cfg, "seed", 0))
            ),
            "train_in_real_env": bool(ppo_cfg.train_in_real_env),
            "max_ep_len": int(ppo_cfg.max_ep_len),
            "rollout_steps": int(getattr(ppo_cfg, "rollout_steps", 1024)),
            "num_imagined_envs": int(
                getattr(ppo_cfg, "num_imagined_envs", 1)
            ),
            "max_training_timesteps": int(ppo_cfg.max_training_timesteps),
            "PPO": resolved_ppo,
        },
    }
    entity = getattr(ppo_cfg, "wandb_entity", None)
    if entity:
        init_kwargs["entity"] = str(entity)
    return wandb.init(**init_kwargs)


def run_training_real_env(cfg):
    # parameters
    hparams = cfg
    hparams_PPO = hparams.PPO
    seed = int(getattr(hparams_PPO, "seed", 0))
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    max_ep_len = int(hparams_PPO.max_ep_len)
    max_training_timesteps = int(hparams_PPO.max_training_timesteps)
    print_freq = int(getattr(hparams_PPO, "console_log_every_steps", 1000))
    episode_log_every_steps = int(
        getattr(hparams_PPO, "episode_log_every_steps", print_freq)
    )
    wandb_log_every_updates = int(
        getattr(hparams_PPO, "wandb_log_every_updates", 1)
    )
    if print_freq < 1 or episode_log_every_steps < 1:
        raise ValueError("PPO console/episode log intervals must be at least 1")
    if wandb_log_every_updates < 1:
        raise ValueError("PPO.wandb_log_every_updates must be at least 1")
    save_model_freq = int(hparams_PPO.save_model_freq)
    update_timestep = int(getattr(hparams_PPO, "rollout_steps", 1024))
    if update_timestep < 2:
        raise ValueError("PPO.rollout_steps must be at least 2")
    print_running_reward = 0
    print_running_episodes = 0
    print_running_steps = 0
    print_running_successes = 0
    next_print_timestep = print_freq
    next_episode_log_timestep = episode_log_every_steps
    rolling_window_episodes = int(
        getattr(hparams_PPO, "rolling_window_episodes", 50)
    )
    if rolling_window_episodes < 1:
        raise ValueError("PPO.rolling_window_episodes must be at least 1")
    recent_rewards = deque(maxlen=rolling_window_episodes)
    recent_steps = deque(maxlen=rolling_window_episodes)
    recent_successes = deque(maxlen=rolling_window_episodes)
    start_time = datetime.now().replace(microsecond=0)
    env_type =  hparams_PPO.env_type

    time_step = 0
    i_episode = 0
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    checkpoint_path = str(policy_checkpoint_path(cfg))

    # param for agent
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    gamma = hparams_PPO.gamma
    lr_actor = hparams_PPO.lr_actor  # learning rate for actor network
    lr_critic = hparams_PPO.lr_critic  # learning rate for critic network
    action_std = hparams_PPO.action_std  # default std for action distribution (can be overwritten by action_std_decay_rate)
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    env_path = hparams_PPO.env_path
    use_wandb = hparams_PPO.use_wandb
    step_penalty = float(getattr(hparams_PPO, "step_penalty", 0.0))
    use_main_dense_reward = bool(
        getattr(hparams_PPO, "use_main_dense_reward", False)
    )
    main_reward = main_dense_reward_settings(hparams_PPO)
    obs_norm_values = getattr(hparams.attention_model, "obs_norm_values", [10, 5, 3])
    entropy_coef = float(getattr(hparams_PPO, "entropy_coef", 0.01))
    normalize_advantages = bool(getattr(hparams_PPO, "normalize_advantages", True))
    normalize_returns = bool(getattr(hparams_PPO, "normalize_returns", False))
    max_grad_norm = float(getattr(hparams_PPO, "max_grad_norm", 0.5))
    freeze_actor_until_first_reward = bool(
        getattr(hparams_PPO, "freeze_actor_until_first_reward", True)
    )

    if use_wandb:
        subrun = _init_policy_wandb_run(cfg, default_project="minigrid_policy_training")


    # state space dimension
    seed_policy_training(seed)
    print(f"[PPO] Seed: {seed}")
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=max_ep_len, render_mode=None,
                        **stochastic_env_kwargs(cfg)))
    
    state_dim = np.prod(env.observation_space['image'].shape) + INVENTORY_TOKEN_COUNT

    # action space dimension
    if has_continuous_action_space:
        action_dim = int(np.prod(env.action_space.shape))
    else:
        action_dim = MODEL_ACTION_COUNT

    seed_policy_training(seed)
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        max_grad_norm=max_grad_norm,
        freeze_actor_until_first_reward=freeze_actor_until_first_reward,
        minibatch_size=int(getattr(hparams_PPO, "minibatch_size", 0)),
    )


    # training loop
    while time_step < max_training_timesteps:
        observation, _ = env.reset()
        current_ep_reward = 0
        current_ep_native_reward = 0.0
        current_ep_shaping_reward = 0.0
        current_ep_goal_region_entries = 0
        current_ep_goal_region_reward = 0.0
        current_ep_goal_region_key_pickups = 0
        current_ep_goal_region_key_pickup_reward = 0.0
        current_ep_goal_region_door_opens = 0
        current_ep_goal_region_door_open_reward = 0.0
        current_ep_progress_milestone_count = 0
        current_ep_progress_milestone_reward = 0.0
        current_ep_critical_door_crossing_count = 0
        current_ep_critical_door_crossing_reward = 0.0
        current_ep_goal_distance_sum = 0.0
        current_ep_goal_distance_count = 0
        current_ep_goal_distance_min = float("inf")
        episode_succeeded = False
        raw_state = torch.as_tensor(
            utils.ColRowCanl_to_CanlRowCol(observation["image"]), device=device
        )
        goal_yx = find_position(raw_state.detach().cpu().numpy(), (8, 1, 0))
        if goal_yx is None:
            raise ValueError("The real MiniGrid layout does not contain a goal")
        real_goal_distance_map = build_main_goal_distance_map(raw_state, goal_yx)
        real_goal_region_mask = build_goal_region_mask(raw_state, goal_yx)
        real_door_topology = build_door_topology(raw_state, goal_yx)
        real_goal_region_seen = torch.zeros(1, device=device, dtype=torch.bool)
        real_rewarded_goal_region_key_positions = torch.zeros(
            (1, *raw_state.shape[-2:]), device=device, dtype=torch.bool
        )
        real_rewarded_goal_region_door_positions = torch.zeros_like(
            real_rewarded_goal_region_key_positions
        )
        real_progress_milestone_seen = torch.zeros(
            1, device=device, dtype=torch.bool
        )
        real_pending_door_ids = torch.full((1,), -1, device=device, dtype=torch.long)
        real_pending_door_origins = torch.full_like(real_pending_door_ids, -1)
        real_rewarded_door_crossings = torch.zeros(
            (1, real_door_topology.num_doors), device=device, dtype=torch.bool
        )
        real_initial_position = (raw_state[0] == 10).nonzero(as_tuple=False)[0]
        real_initial_goal_distance = real_goal_distance_map[
            real_initial_position[0], real_initial_position[1]
        ].reshape(1)
        real_best_goal_distance = real_initial_goal_distance.clone()
        state = preprocess_observation(
            observation['image'],
            obs_norm_values,
            inventory_token=carrying_token_from_env(env),
            inventory_classes=INVENTORY_TOKEN_COUNT,
        ).to(device)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action, state_buffer, action_buffer, action_logprob, state_val = ppo_agent.select_action(state)
            previous_raw_state = raw_state
            previous_inventory_token = carrying_token_from_env(env)
            observation, native_reward, terminated, truncated, _ = env.step(
                compact_to_native(action)
            )
            raw_state = torch.as_tensor(
                utils.ColRowCanl_to_CanlRowCol(observation["image"]), device=device
            )
            step_succeeded = bool(terminated and native_reward > 0.0)
            lava_terminated = bool(terminated and not step_succeeded)
            episode_succeeded |= step_succeeded
            current_ep_native_reward += float(native_reward)
            if use_main_dense_reward:
                dense_reward, real_goal_region_seen, reward_events = main_dense_rewards(
                    previous_raw_state.unsqueeze(0),
                    raw_state.unsqueeze(0),
                    torch.as_tensor([action], device=device),
                    real_goal_distance_map,
                    real_goal_region_mask,
                    real_goal_region_seen,
                    torch.as_tensor([step_succeeded], device=device),
                    torch.as_tensor([lava_terminated], device=device),
                    best_goal_distances=real_best_goal_distance,
                    initial_goal_distances=real_initial_goal_distance,
                    progress_milestone_seen=real_progress_milestone_seen,
                    previous_carrying_tokens=torch.as_tensor(
                        [previous_inventory_token], device=device
                    ),
                    current_carrying_tokens=torch.as_tensor(
                        [carrying_token_from_env(env)], device=device
                    ),
                    door_topology=real_door_topology,
                    pending_door_ids=real_pending_door_ids,
                    pending_door_origins=real_pending_door_origins,
                    rewarded_door_crossings=real_rewarded_door_crossings,
                    rewarded_goal_region_key_positions=real_rewarded_goal_region_key_positions,
                    rewarded_goal_region_door_positions=real_rewarded_goal_region_door_positions,
                    **main_reward,
                )
                real_best_goal_distance = reward_events["new_best_goal_distance"]
                real_progress_milestone_seen = reward_events[
                    "progress_milestone_seen"
                ]
                real_pending_door_ids = reward_events["pending_door_ids"]
                real_pending_door_origins = reward_events["pending_door_origins"]
                real_rewarded_door_crossings = reward_events["rewarded_door_crossings"]
                real_rewarded_goal_region_key_positions = reward_events[
                    "rewarded_goal_region_key_positions"
                ]
                real_rewarded_goal_region_door_positions = reward_events[
                    "rewarded_goal_region_door_positions"
                ]
                reward = float(dense_reward.item())
                current_ep_shaping_reward += (
                    0.0 if step_succeeded or lava_terminated else reward
                )
                current_ep_goal_region_entries += int(
                    reward_events["goal_region_entered"].item()
                )
                current_ep_goal_region_reward += float(
                    reward_events["goal_region_reward"].item()
                )
                current_ep_goal_region_key_pickups += int(
                    reward_events["goal_region_key_picked_up"].item()
                )
                current_ep_goal_region_key_pickup_reward += float(
                    reward_events["goal_region_key_pickup_reward"].item()
                )
                current_ep_goal_region_door_opens += int(
                    reward_events["goal_region_door_opened"].item()
                )
                current_ep_goal_region_door_open_reward += float(
                    reward_events["goal_region_door_open_reward"].item()
                )
                current_ep_progress_milestone_count += int(
                    reward_events["progress_milestone_crossed"].item()
                )
                current_ep_progress_milestone_reward += float(
                    reward_events["progress_milestone_reward"].item()
                )
                current_ep_critical_door_crossing_count += int(
                    reward_events["critical_door_crossed"].item()
                )
                current_ep_critical_door_crossing_reward += float(
                    reward_events["critical_door_crossing_reward"].item()
                )
                current_distance = float(reward_events["current_distance"].item())
                if np.isfinite(current_distance):
                    current_ep_goal_distance_sum += current_distance
                    current_ep_goal_distance_count += 1
                    current_ep_goal_distance_min = min(
                        current_ep_goal_distance_min, current_distance
                    )
            else:
                reward = float(native_reward) + step_penalty
            done = terminated or truncated
            state = preprocess_observation(
                observation['image'],
                obs_norm_values,
                inventory_token=carrying_token_from_env(env),
                inventory_classes=INVENTORY_TOKEN_COUNT,
            ).to(device)
            # saving reward and is_terminals
            ppo_agent.save_buffer(state_buffer, action_buffer, action_logprob, state_val, reward, done)



            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                if len(ppo_agent.buffer.rewards) > 1:
                    bootstrap_value = 0.0 if done else ppo_agent.estimate_old_value(state)
                    update_metrics = ppo_agent.update(bootstrap_value=bootstrap_value)
                    if (
                        use_wandb
                        and update_metrics.get("updated", False)
                        and int(update_metrics["update_count"]) % wandb_log_every_updates == 0
                    ):
                        subrun.log(
                            {
                                "time_step": int(time_step),
                                "update": int(update_metrics["update_count"]),
                                "loss": update_metrics["loss"],
                                "grad_norm": update_metrics["grad_norm"],
                                "actor_delta": update_metrics.get(
                                    "actor_parameter_delta", 0.0
                                ),
                                "critic_delta": update_metrics.get(
                                    "critic_parameter_delta", 0.0
                                ),
                                "actor_loss": update_metrics.get("actor_loss", 0.0),
                                "critic_loss": update_metrics.get("critic_loss", 0.0),
                                "entropy": update_metrics.get("entropy", 0.0),
                                "ppo/loss": update_metrics["loss"],
                                "ppo/grad_norm": update_metrics["grad_norm"],
                                "ppo/parameter_delta": update_metrics["parameter_delta"],
                                "ppo/rollout_size": update_metrics["rollout_size"],
                                "ppo/update_count": update_metrics["update_count"],
                            },
                            step=time_step,
                        )

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            budget_exhausted = time_step >= max_training_timesteps
            if done or t == max_ep_len or budget_exhausted:
                break
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        print_running_steps += t
        reported_success = (
            episode_succeeded if use_main_dense_reward else current_ep_reward > 0.0
        )
        print_running_successes += int(reported_success)
        recent_rewards.append(float(current_ep_reward))
        recent_steps.append(int(t))
        recent_successes.append(int(reported_success))

        rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
        rolling_avg_steps = sum(recent_steps) / len(recent_steps)
        rolling_success_rate = sum(recent_successes) / len(recent_successes)

        if (
            use_wandb
            and (
                time_step >= next_episode_log_timestep
                or time_step >= max_training_timesteps
            )
        ):
            subrun.log(
                {
                    "episode/reward": current_ep_reward,
                    "episode/native_reward": current_ep_native_reward,
                    "episode/shaping_reward": current_ep_shaping_reward,
                    "episode/success": int(reported_success),
                    "episode/steps": t,
                    "episode/index": i_episode,
                    "rolling/average_reward": rolling_avg_reward,
                    "rolling/success_rate": rolling_success_rate,
                    "rolling/average_episode_steps": rolling_avg_steps,
                    "rolling/window_episode_count": len(recent_rewards),
                    "train/reward_per_step": current_ep_reward / max(t, 1),
                    "goal_region_entry_count": current_ep_goal_region_entries,
                    "goal_region_reward_sum": current_ep_goal_region_reward,
                    "goal_region_key_pickup_count": current_ep_goal_region_key_pickups,
                    "goal_region_key_pickup_reward_sum": current_ep_goal_region_key_pickup_reward,
                    "goal_region_door_open_count": current_ep_goal_region_door_opens,
                    "goal_region_door_open_reward_sum": current_ep_goal_region_door_open_reward,
                    "progress_milestone_count": current_ep_progress_milestone_count,
                    "progress_milestone_reward_sum": current_ep_progress_milestone_reward,
                    "critical_door_crossing_count": current_ep_critical_door_crossing_count,
                    "critical_door_crossing_reward_sum": current_ep_critical_door_crossing_reward,
                    "goal_dist_mean": current_ep_goal_distance_sum
                    / max(current_ep_goal_distance_count, 1),
                    "goal_dist_min": current_ep_goal_distance_min
                    if np.isfinite(current_ep_goal_distance_min)
                    else None,
                    "train/goal_dist_mean": current_ep_goal_distance_sum
                    / max(current_ep_goal_distance_count, 1),
                    "train/goal_dist_min": current_ep_goal_distance_min
                    if np.isfinite(current_ep_goal_distance_min)
                    else None,
                    "train/episode_return_mean": rolling_avg_reward,
                    "train/episode_length_mean": rolling_avg_steps,
                    "train/episode_success_rate": rolling_success_rate,
                    "train/rolling_success_rate": rolling_success_rate,
                    "train/episodes_completed": i_episode + 1,
                },
                step=time_step,
            )
            while next_episode_log_timestep <= time_step:
                next_episode_log_timestep += episode_log_every_steps

        if time_step >= next_print_timestep and print_running_episodes > 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_steps = print_running_steps / print_running_episodes
            print_success_rate = print_running_successes / print_running_episodes
            print(
                f"Episode : {i_episode} \t Timestep : {time_step} \t "
                f"Interval Reward : {print_avg_reward:.5f} \t "
                f"Interval Success : {print_success_rate:.1%} \t "
                f"Interval Steps : {print_avg_steps:.1f} \t "
                f"Rolling({len(recent_rewards)}) Reward : {rolling_avg_reward:.5f} \t "
                f"Success : {rolling_success_rate:.1%} \t "
                f"Steps : {rolling_avg_steps:.1f}"
            )
            print_running_reward = 0
            print_running_episodes = 0
            print_running_steps = 0
            print_running_successes = 0
            while next_print_timestep <= time_step:
                next_print_timestep += print_freq

        i_episode += 1

    # Fixed-size rollouts may leave one partial batch at the end of training.
    if len(ppo_agent.buffer.rewards) >= 2:
        bootstrap_value = 0.0 if done else ppo_agent.estimate_old_value(state)
        update_metrics = ppo_agent.update(bootstrap_value=bootstrap_value)
        if (
            use_wandb
            and update_metrics.get("updated", False)
            and int(update_metrics["update_count"]) % wandb_log_every_updates == 0
        ):
            subrun.log(
                {
                    "time_step": int(time_step),
                    "update": int(update_metrics["update_count"]),
                    "loss": update_metrics["loss"],
                    "grad_norm": update_metrics["grad_norm"],
                    "actor_delta": update_metrics.get(
                        "actor_parameter_delta", 0.0
                    ),
                    "critic_delta": update_metrics.get(
                        "critic_parameter_delta", 0.0
                    ),
                    "actor_loss": update_metrics.get("actor_loss", 0.0),
                    "critic_loss": update_metrics.get("critic_loss", 0.0),
                    "entropy": update_metrics.get("entropy", 0.0),
                    "ppo/loss": update_metrics["loss"],
                    "ppo/grad_norm": update_metrics["grad_norm"],
                    "ppo/parameter_delta": update_metrics["parameter_delta"],
                    "ppo/rollout_size": update_metrics["rollout_size"],
                    "ppo/update_count": update_metrics["update_count"],
                },
                step=time_step,
            )
    else:
        ppo_agent.buffer.clear()

    # Always persist the policy from the final update, even when the total
    # budget is not an exact multiple of save_model_freq.
    ppo_agent.save(checkpoint_path)
    print(f"Final policy saved at: {checkpoint_path}")
    env.close()
    if use_wandb:
        subrun.finish()


def run_policy_evaluation(cfg: DictConfig):
    # Keep one authoritative MiniGrid evaluation path so direct callers and
    # run_pipeline use the same dense/native reward and goal-region metrics.
    from modelBased.policy_training.PPO_world_test import validate_policy

    return validate_policy(cfg)

@hydra.main(version_base=None, config_path=str(SCRIPT_ROOT / "modelBased/config"), config_name="config")
def main(cfg: DictConfig):
    control_mode = str(getattr(cfg.PPO, "wm_control_mode", "ppo")).lower()
    if control_mode not in {"ppo", "mpc", "mcts", "astar"}:
        raise ValueError(
            "PPO.wm_control_mode must be 'ppo', 'mpc', 'mcts', or 'astar'"
        )
    if control_mode == "mpc":
        if getattr(cfg.PPO, "train_in_real_env", False):
            raise ValueError("PPO.wm_control_mode=mpc requires PPO.train_in_real_env=false")
        from modelBased.policy_training.mpc_planner import run_online_mpc
        print("Running online MPC with world-model rollouts...")
        run_online_mpc(cfg)
        return
    if control_mode == "mcts":
        if getattr(cfg.PPO, "train_in_real_env", False):
            raise ValueError("PPO.wm_control_mode=mcts requires PPO.train_in_real_env=false")
        from modelBased.policy_training.mcts_planner import run_online_mcts
        print("Running online MCTS with world-model rollouts...")
        run_online_mcts(cfg)
        return
    if control_mode == "astar":
        if getattr(cfg.PPO, "train_in_real_env", False):
            raise ValueError("PPO.wm_control_mode=astar requires PPO.train_in_real_env=false")
        from modelBased.policy_training.dijkstra_planner import (
            plan_and_validate_world_model,
        )
        print("Running receding-horizon A* with frozen world-model rollouts...")
        plan_and_validate_world_model(cfg)
        return
    if getattr(cfg.PPO, "train_in_real_env", False):
        print("Training PPO directly in the REAL environment...")
        run_training_real_env(cfg)
    else:
        print("Training PPO using the World Model...")
        run_ppo_wm(cfg)

if __name__ == "__main__":
    main()
