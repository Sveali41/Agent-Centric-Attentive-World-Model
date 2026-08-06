import sys
import random
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelBased.common.utils import PROJECT_ROOT
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
from modelBased.policy_training.PPO import PPO
import hydra
from datetime import datetime
from modelBased.common import utils
from domain.minigrid.action_codec import compact_to_native, MODEL_ACTION_COUNT

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

def process_data(state, maks_size):
    return utils.extract_masked_state_torch(state, maks_size)

def evaluate_policy(policy, env, episodes, obs_norm_values):
    """
    Evaluate the policy by running it in the environment for a number of episodes.
    """
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()[0]['image']
        state = utils.ColRowCanl_to_CanlRowCol(obs)
        done = False
        ep_reward = 0
        for _ in range(env.max_steps):
            state_tensor = torch.tensor(utils.normalize_obs(state, obs_norm_values)).to(device)
            action, _, _, _, _ = policy.select_action(state_tensor.flatten())
            obs, reward, done, _, _ = env.step(compact_to_native(action))
            state = utils.ColRowCanl_to_CanlRowCol(obs['image'])
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward
    return total_reward / episodes

# add the function add objects into the inventory
def add_object_to_inventory(delta_state, info):
    """
    Update the imagined MiniGrid inventory after a key is removed.
    
    Args:
        for a keydoor environment
        info['carrying_key'] (bool): Whether the agent is carrying a key.
    """

    # MiniGrid encodes key=5 and empty=1, hence pickup produces 1 - 5 = -4.
    # Keep -5 for compatibility with older observations that used unseen=0.
    if (delta_state[0, :, :] == -4).any() or (delta_state[0, :, :] == -5).any():
        info['carrying_key'] = True
    return info


def apply_known_minigrid_interaction(state_masked, action, info):
    """Apply deterministic pickup/toggle rules in compact action space.

    Random WM data contains very few successful interaction transitions. These
    rules prevent a missed pickup from blocking every subsequent imagined step;
    movement and turning dynamics are still predicted by the world model.
    """
    action = int(action)
    next_state = state_masked.clone()
    next_info = dict(info or {})
    carrying_key = bool(next_info.get('carrying_key', False))

    center = state_masked.shape[-1] // 2
    direction = int(state_masked[2, center, center].item())
    direction_delta = {
        0: (0, 1),   # right
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (-1, 0),  # up
    }
    dy, dx = direction_delta[direction]
    front_y, front_x = center + dy, center + dx
    if not (0 <= front_y < state_masked.shape[-2] and 0 <= front_x < state_masked.shape[-1]):
        return next_state, next_info

    front_object = int(state_masked[0, front_y, front_x].item())
    front_status = int(state_masked[2, front_y, front_x].item())

    if action == 3 and front_object == 5 and not carrying_key:  # pickup key
        next_state[:, front_y, front_x] = next_state.new_tensor([1, 0, 0])
        next_info['carrying_key'] = True
    elif action == 4 and front_object == 4:  # toggle door
        if front_status == 2 and carrying_key:      # locked -> open
            next_state[2, front_y, front_x] = 0
        elif front_status == 0:                     # open -> closed
            next_state[2, front_y, front_x] = 1
        elif front_status == 1:                     # unlocked closed -> open
            next_state[2, front_y, front_x] = 0

    return next_state, next_info


def apply_known_minigrid_interaction_batch(
    state_masked, actions, carrying_key
):
    """Vectorized pickup/toggle dynamics for parallel imagined MiniGrid."""
    next_state = state_masked.clone()
    actions = actions.to(state_masked.device).long().reshape(-1)
    carrying_key = carrying_key.to(state_masked.device).bool().reshape(-1).clone()
    batch_size = state_masked.shape[0]
    center_y = state_masked.shape[-2] // 2
    center_x = state_masked.shape[-1] // 2

    directions = state_masked[:, 2, center_y, center_x].long()
    direction_deltas = torch.tensor(
        [[0, 1], [1, 0], [0, -1], [-1, 0]],
        device=state_masked.device,
        dtype=torch.long,
    )
    deltas = direction_deltas[directions]
    front_y = center_y + deltas[:, 0]
    front_x = center_x + deltas[:, 1]
    batch_ids = torch.arange(batch_size, device=state_masked.device)
    front_object = state_masked[batch_ids, 0, front_y, front_x].long()
    front_status = state_masked[batch_ids, 2, front_y, front_x].long()

    pickup = (actions == 3) & (front_object == 5) & (~carrying_key)
    if pickup.any():
        pickup_ids = batch_ids[pickup]
        pickup_y = front_y[pickup]
        pickup_x = front_x[pickup]
        next_state[pickup_ids, 0, pickup_y, pickup_x] = 1
        next_state[pickup_ids, 1, pickup_y, pickup_x] = 0
        next_state[pickup_ids, 2, pickup_y, pickup_x] = 0
        carrying_key[pickup] = True

    toggle = (actions == 4) & (front_object == 4)
    open_door = toggle & (
        ((front_status == 2) & carrying_key) | (front_status == 1)
    )
    close_door = toggle & (front_status == 0)
    if open_door.any():
        next_state[
            batch_ids[open_door],
            2,
            front_y[open_door],
            front_x[open_door],
        ] = 0
    if close_door.any():
        next_state[
            batch_ids[close_door],
            2,
            front_y[close_door],
            front_x[close_door],
        ] = 1

    return next_state, carrying_key


def _nearest_valid_values(values, valid_values):
    valid = torch.as_tensor(
        valid_values, device=values.device, dtype=values.dtype
    )
    indices = torch.argmin(torch.abs(values.unsqueeze(-1) - valid), dim=-1)
    return valid[indices]


def imagined_minigrid_step_batch(
    model,
    states,
    actions,
    carrying_key,
    attention_mask_size,
    valid_values_obj,
    valid_values_color,
    valid_values_state,
):
    """Advance B imagined states with one batched WM call.

    Movement and turning use the learned model. Pickup and toggle retain the
    same known interaction dynamics as the previous serial implementation.
    """
    agent_positions = utils.get_agent_position_torch(states)
    masked = utils.extract_masked_state_torch(
        states, attention_mask_size, agent_positions
    )
    predicted = masked.clone().float()
    learned = actions < 3

    if learned.any():
        learned_info = {"carrying_key": carrying_key[learned]}
        wm_out, _, _ = model(masked[learned], actions[learned], learned_info)
        if getattr(model, "out_channel", 3) == 21:
            predicted[learned] = torch.stack(
                (
                    torch.argmax(wm_out[:, 0:11], dim=1),
                    torch.argmax(wm_out[:, 11:17], dim=1),
                    torch.argmax(wm_out[:, 17:21], dim=1),
                ),
                dim=1,
            ).float()
        else:
            predicted[learned] = masked[learned] + wm_out

    interactions = ~learned
    if interactions.any():
        interaction_state, interaction_keys = apply_known_minigrid_interaction_batch(
            masked[interactions], actions[interactions], carrying_key[interactions]
        )
        predicted[interactions] = interaction_state.float()
        carrying_key = carrying_key.clone()
        carrying_key[interactions] = interaction_keys

    predicted[:, 0] = _nearest_valid_values(
        predicted[:, 0], valid_values_obj
    )
    predicted[:, 1] = _nearest_valid_values(
        predicted[:, 1], valid_values_color
    )
    predicted[:, 2] = _nearest_valid_values(
        predicted[:, 2], valid_values_state
    )
    next_states = utils.put_back_masked_state_torch(
        predicted, states, attention_mask_size, agent_positions
    )
    return next_states, carrying_key




@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
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
    if module_class is not None:
        model = module_class(
            hparams_world_model.data_type,  
            hparams_world_model.grid_shape, 
            hparams_world_model.attention_mask_size, 
            hparams_world_model.embed_dim, 
            hparams_world_model.num_heads,
            env_type=hparams_world_model.env_type,
            frame_stack=hparams_world_model.frame_stack,
        )
    else:
        print(f"Model type: {hparams_world_model.model_type} not supported")
        exit()
    utils.load_model_weight(model, hparams_world_model.model_save_path)
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
    checkpoint_path = hparams_PPO.checkpoint_path
    env_path = hparams_PPO.env_path
    visualize_flag = hparams_PPO.visualize
    env_type =  hparams_PPO.env_type
    use_wandb = hparams_PPO.use_wandb
    wandb_run_name = hparams_PPO.wandb_run_name
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
    entropy_coef = float(getattr(hparams_PPO, "entropy_coef", 0.01))
    normalize_advantages = bool(getattr(hparams_PPO, "normalize_advantages", True))
    normalize_returns = bool(getattr(hparams_PPO, "normalize_returns", False))
    max_grad_norm = float(getattr(hparams_PPO, "max_grad_norm", 0.5))
    rolling_window_episodes = int(
        getattr(hparams_PPO, "rolling_window_episodes", 50)
    )
    if rolling_window_episodes < 1:
        raise ValueError("PPO.rolling_window_episodes must be at least 1")
    

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
                        max_steps=max_ep_len, render_mode=None))
    # 4. Initialize training
    i_episode = 0
    print_freq = 1000
    print_running_reward = 0
    print_running_episodes = 0
    print_running_steps = 0
    print_running_successes = 0
    next_print_timestep = print_freq
    recent_rewards = deque(maxlen=rolling_window_episodes)
    recent_steps = deque(maxlen=rolling_window_episodes)
    recent_successes = deque(maxlen=rolling_window_episodes)
    time_step = 0
    next_save_timestep = save_model_freq
    step_penalty = float(getattr(hparams_PPO, "step_penalty", 0.0))
    final_norm_regret = None
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = int(np.prod(env.action_space.shape))
    else:
        # Keep PPO's action IDs identical to the MiniGrid environment and WM
        # dataset (the full MiniGrid action space is normally 0..6).
        # PPO/WM use the compact five-action space. ``drop`` and ``done`` are
        # excluded during data collection and are mapped only at env.step().
        action_dim = MODEL_ACTION_COUNT
    state_dim = np.prod(env.observation_space['image'].shape)
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
    )
    if compute_regret:
        real_policy_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)
        real_policy_agent.load(real_policy_path)

    

    def reset_imagined_state():
        observation = env.reset()[0]["image"]
        return torch.as_tensor(
            utils.ColRowCanl_to_CanlRowCol(observation), device=device
        )

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
    carrying_key = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    episode_rewards = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.float32
    )
    episode_steps = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.long
    )
    last_dones = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )

    if sub_run is not None:
        sub_run.log({"final_tasks": wandb.Image(env.get_frame())})

    print(
        f"[WM PPO] Parallel imagined environments: {num_imagined_envs}; "
        f"temporal rollout: {update_timestep // num_imagined_envs}; "
        f"transitions/update: {update_timestep}"
    )

    def log_ppo_update(update_metrics):
        if sub_run is not None and update_metrics.get("updated", False):
            sub_run.log(
                {
                    "ppo/loss": update_metrics["loss"],
                    "ppo/grad_norm": update_metrics["grad_norm"],
                    "ppo/parameter_delta": update_metrics["parameter_delta"],
                    "ppo/rollout_size": update_metrics["rollout_size"],
                    "ppo/update_count": update_metrics["update_count"],
                },
                step=time_step,
            )

    # Each loop advances B independent trajectories by one imagined step.
    while time_step < max_training_timesteps:
        state_norm = utils.normalize_obs(
            states.clone(), hparams_world_model.obs_norm_values
        ).reshape(num_imagined_envs, -1)
        (
            actions,
            state_buffer,
            action_buffer,
            action_logprobs,
            state_values,
        ) = ppo_agent.select_action_batch(state_norm)

        # no_grad keeps the resulting state mutable so completed batch slots
        # can be reset in place. inference_mode tensors forbid that update.
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
            )

        episode_steps += 1
        agent_positions = utils.get_agent_position_torch(states)
        reached_goal = torch.all(agent_positions == goal_positions, dim=1)
        rewards = torch.where(
            reached_goal,
            1.0 - 0.9 * episode_steps.float() / float(max_ep_len),
            torch.zeros_like(episode_rewards),
        )
        rewards = rewards + step_penalty
        truncated = episode_steps >= max_ep_len
        dones = reached_goal | truncated
        episode_rewards += rewards

        ppo_agent.save_buffer_batch(
            state_buffer,
            action_buffer,
            action_logprobs,
            state_values,
            rewards,
            dones,
        )
        time_step += num_imagined_envs
        last_dones = dones.clone()

        if time_step % update_timestep == 0:
            next_state_norm = utils.normalize_obs(
                states.clone(), hparams_world_model.obs_norm_values
            ).reshape(num_imagined_envs, -1)
            bootstrap_values = ppo_agent.estimate_old_values_batch(next_state_norm)
            bootstrap_values[last_dones.detach().cpu()] = 0.0
            update_metrics = ppo_agent.update(
                bootstrap_value=bootstrap_values
            )
            log_ppo_update(update_metrics)

        completed = torch.nonzero(dones, as_tuple=False).reshape(-1)
        if completed.numel() > 0:
            completed_rewards = episode_rewards[completed].detach().cpu().tolist()
            completed_steps = episode_steps[completed].detach().cpu().tolist()
            completed_successes = reached_goal[completed].detach().cpu().int().tolist()
            for ep_reward, ep_steps, ep_success in zip(
                completed_rewards, completed_steps, completed_successes
            ):
                print_running_reward += float(ep_reward)
                print_running_steps += int(ep_steps)
                print_running_successes += int(ep_success)
                print_running_episodes += 1
                recent_rewards.append(float(ep_reward))
                recent_steps.append(int(ep_steps))
                recent_successes.append(int(ep_success))
                i_episode += 1

            rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
            rolling_avg_steps = sum(recent_steps) / len(recent_steps)
            rolling_success_rate = sum(recent_successes) / len(recent_successes)
            if sub_run is not None:
                sub_run.log(
                    {
                        "episode/reward": sum(completed_rewards) / len(completed_rewards),
                        "episode/success": sum(completed_successes) / len(completed_successes),
                        "episode/steps": sum(completed_steps) / len(completed_steps),
                        "episode/completed_count": len(completed_rewards),
                        "episode/index": i_episode,
                        "rolling/average_reward": rolling_avg_reward,
                        "rolling/success_rate": rolling_success_rate,
                        "rolling/average_episode_steps": rolling_avg_steps,
                        "rolling/window_episode_count": len(recent_rewards),
                    },
                    step=time_step,
                )

            # Reset only completed slots; all other imagined trajectories keep
            # their current states and episode statistics.
            reset_states = torch.stack(
                [reset_imagined_state() for _ in range(completed.numel())], dim=0
            )
            states[completed] = reset_states
            carrying_key[completed] = False
            episode_rewards[completed] = 0.0
            episode_steps[completed] = 0

        if time_step >= next_print_timestep and print_running_episodes > 0:
            rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
            rolling_avg_steps = sum(recent_steps) / len(recent_steps)
            rolling_success_rate = sum(recent_successes) / len(recent_successes)
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

        if time_step >= next_save_timestep:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
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
        bootstrap_values = ppo_agent.estimate_old_values_batch(next_state_norm)
        bootstrap_values[last_dones.detach().cpu()] = 0.0
        update_metrics = ppo_agent.update(bootstrap_value=bootstrap_values)
        log_ppo_update(update_metrics)
    else:
        ppo_agent.buffer.clear()

    # Final save after loop completes
    ppo_agent.save(checkpoint_path)
    print(f"Final policy saved at: {checkpoint_path}")
    env.close()
    if use_wandb:
        sub_run.finish()
    if compute_regret: 
        return final_norm_regret


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent_real_env(cfg: DictConfig):
    run_training_real_env(cfg)

def _init_policy_wandb_run(cfg, default_project):
    """Initialize WandB from the user's login without embedding credentials."""
    ppo_cfg = cfg.PPO
    wandb.login()
    init_kwargs = {
        "project": str(getattr(ppo_cfg, "wandb_project", default_project)),
        "name": str(ppo_cfg.wandb_run_name),
        "group": str(
            getattr(
                ppo_cfg,
                "wandb_group",
                f"{cfg.domain}_{cfg.domains[str(cfg.domain)].task_name}_policy",
            )
        ),
        "reinit": True,
        "config": {
            "domain": str(cfg.domain),
            "task_name": str(cfg.domains[str(cfg.domain)].task_name),
            "seed": int(getattr(ppo_cfg, "seed", 0)),
            "train_in_real_env": bool(ppo_cfg.train_in_real_env),
            "max_ep_len": int(ppo_cfg.max_ep_len),
            "rollout_steps": int(getattr(ppo_cfg, "rollout_steps", 1024)),
            "num_imagined_envs": int(
                getattr(ppo_cfg, "num_imagined_envs", 1)
            ),
            "max_training_timesteps": int(ppo_cfg.max_training_timesteps),
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
    print_freq = 1000
    save_model_freq = int(hparams_PPO.save_model_freq)
    update_timestep = int(getattr(hparams_PPO, "rollout_steps", 1024))
    if update_timestep < 2:
        raise ValueError("PPO.rollout_steps must be at least 2")
    print_running_reward = 0
    print_running_episodes = 0
    print_running_steps = 0
    print_running_successes = 0
    next_print_timestep = print_freq
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
    wandb_run_name = hparams_PPO.wandb_run_name

    time_step = 0
    i_episode = 0
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    checkpoint_path = hparams_PPO.checkpoint_path

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
    obs_norm_values = getattr(hparams.attention_model, "obs_norm_values", [10, 5, 3])
    entropy_coef = float(getattr(hparams_PPO, "entropy_coef", 0.01))
    normalize_advantages = bool(getattr(hparams_PPO, "normalize_advantages", True))
    normalize_returns = bool(getattr(hparams_PPO, "normalize_returns", False))
    max_grad_norm = float(getattr(hparams_PPO, "max_grad_norm", 0.5))

    if use_wandb:
        subrun = _init_policy_wandb_run(cfg, default_project="minigrid_policy_training")


    # state space dimension
    seed_policy_training(seed)
    print(f"[PPO] Seed: {seed}")
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=max_ep_len, render_mode=None))
    
    state_dim = np.prod(env.observation_space['image'].shape)

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
    )


    # training loop
    while time_step < max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        state = preprocess_observation(state[0]['image'], obs_norm_values).to(device)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action, state_buffer, action_buffer, action_logprob, state_val = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(compact_to_native(action))
            reward += step_penalty
            done = terminated or truncated
            state = preprocess_observation(state['image'], obs_norm_values).to(device)
            # saving reward and is_terminals
            ppo_agent.save_buffer(state_buffer, action_buffer, action_logprob, state_val, reward, done)



            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                if len(ppo_agent.buffer.rewards) > 1:
                    bootstrap_value = 0.0 if done else ppo_agent.estimate_old_value(state)
                    update_metrics = ppo_agent.update(bootstrap_value=bootstrap_value)
                    if use_wandb and update_metrics.get("updated", False):
                        subrun.log(
                            {
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
        print_running_successes += int(current_ep_reward > 0.0)
        recent_rewards.append(float(current_ep_reward))
        recent_steps.append(int(t))
        recent_successes.append(int(current_ep_reward > 0.0))

        rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
        rolling_avg_steps = sum(recent_steps) / len(recent_steps)
        rolling_success_rate = sum(recent_successes) / len(recent_successes)

        if use_wandb:
            subrun.log(
                {
                    "episode/reward": current_ep_reward,
                    "episode/success": int(current_ep_reward > 0.0),
                    "episode/steps": t,
                    "episode/index": i_episode,
                    "rolling/average_reward": rolling_avg_reward,
                    "rolling/success_rate": rolling_success_rate,
                    "rolling/average_episode_steps": rolling_avg_steps,
                    "rolling/window_episode_count": len(recent_rewards),
                },
                step=time_step,
            )

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
        if use_wandb and update_metrics.get("updated", False):
            subrun.log(
                {
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
    hparams = cfg
    # 1. World Model
    hparams_world_model = hparams.attention_model
    # 2. PPO
    # hyperparameters
    # compute regret
    hparams_PPO = hparams.PPO

    lr_actor = hparams_PPO.lr_actor
    lr_critic = hparams_PPO.lr_critic
    gamma = hparams_PPO.gamma
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    action_std = hparams_PPO.action_std
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    checkpoint_path = hparams_PPO.checkpoint_path_wm
    env_path = hparams_PPO.env_path
    env_type =  hparams_PPO.env_type
    episodes = hparams_PPO.get("episodes_eval")

    # 3. Real environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
 
    # action space dimension
    if has_continuous_action_space:
        action_dim = int(np.prod(env.action_space.shape))
    else:
        action_dim = MODEL_ACTION_COUNT
    state_dim = np.prod(env.observation_space['image'].shape)

    policy_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std)
    policy_agent.load(checkpoint_path)
    Reward = evaluate_policy(policy_agent, env, episodes, hparams_world_model.obs_norm_values)
    return Reward

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def main(cfg: DictConfig):
    if getattr(cfg.PPO, "train_in_real_env", False):
        print("Training PPO directly in the REAL environment...")
        run_training_real_env(cfg)
    else:
        print("Training PPO using the World Model...")
        run_ppo_wm(cfg)

if __name__ == "__main__":
    main()
