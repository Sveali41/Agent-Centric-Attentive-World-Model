import sys
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

    if visualize_flag:
        visualize = utils.Visualization(hparams_world_model)
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

    

    # training loop
    done = False
    state_0 = None
    while time_step < max_training_timesteps:

        state_init = env.reset()[0]['image']
        if time_step == 0 and sub_run is not None:
            img = env.get_frame()
            sub_run.log({"final_tasks": wandb.Image(img)})
        state_0 = utils.ColRowCanl_to_CanlRowCol(state_init)
        goal_position_yx = find_position(state_0, (8, 1, 0)) # find the goal position
        current_ep_reward = 0
        info = {'carrying_key': False}
        for t in range(1, max_ep_len + 1):
            # self.buffer.states = [state.squeeze(0) if state.dim() > 1 else state for state in self.buffer.states]
            if t==1:
                # state = utils.normalize_obs(state_0, hparams_world_model.obs_norm_values)
                state_0 = torch.tensor(state_0).to(device)
            
            # Keep the raw discrete state for the WM and reward detector.
            # normalize_obs operates in-place on tensors, so passing state_0
            # directly here corrupts the state used by the next WM step.
            state_norm = utils.normalize_obs(
                state_0.clone(), hparams_world_model.obs_norm_values
            )
            action, state_buffer, action_buffer, action_logprob, state_val = ppo_agent.select_action(state_norm.flatten()) # state is the dimension of flatten

 
            state_masked = process_data(state_0.clone(), hparams_world_model.attention_mask_size)
            if int(action) in (3, 4):
                state_pre_masked, info = apply_known_minigrid_interaction(
                    state_masked, int(action), info
                )
            else:
                with torch.no_grad():
                    # The WM was trained with compact dataset action IDs.
                    wm_out, _, _ = model(state_masked, int(action), info)
                
                if getattr(model, 'out_channel', 3) == 21:
                    # Cross Entropy mode: output is categorical logits
                    # wm_out shape is (21, H, W) because batch dim is squeezed
                    obj_pred = torch.argmax(wm_out[0:11, ...], dim=0, keepdim=True)
                    color_pred = torch.argmax(wm_out[11:17, ...], dim=0, keepdim=True)
                    state_pred = torch.argmax(wm_out[17:21, ...], dim=0, keepdim=True)
                    state_pre_masked = torch.cat([obj_pred, color_pred, state_pred], dim=0).float()
                else:
                    # Legacy MSE mode: output is continuous delta
                    delta_masked = wm_out
                    state_pre_masked = state_masked + delta_masked
            if visualize_flag:
                visualize.compare_states(state_masked, state_pre_masked, action, t, True)
            # delta_state_pre = delta_state_pre.to(dtype=torch.float32)
            # denorm the state
            # state_pre_denorm = utils.denormalize_obj(state_pre, hparams_world_model.obs_norm_values)

            state_pre_masked = utils.map_obs_to_nearest_value(state_pre_masked, 
                                                              hparams_world_model.valid_values_obj,
                                                              hparams_world_model.valid_values_color,
                                                              hparams_world_model.valid_values_state)

            info = add_object_to_inventory((state_pre_masked - state_masked), info)
            agent_postion_yx = utils.get_agent_position_torch(state_0)
            state_pre = utils.put_back_masked_state_torch(
                state_pre_masked,
                state_0,
                hparams_world_model.attention_mask_size,
                agent_postion_yx,
            )
            

                
            state_0 = state_pre
            # obtain reward from the state representation & done
            reached_goal, reward = get_destination(
                state_0, t, max_ep_len, goal_position_yx
            )
            reward += step_penalty
            # Match Gymnasium's real-env semantics: the horizon is a terminal
            # truncation for the trajectory even when the goal was not reached.
            done = reached_goal or t == max_ep_len
            # saving reward and is_terminals
            ppo_agent.save_buffer(state_buffer, action_buffer, action_logprob, state_val, reward, done)
            

            time_step += 1
            current_ep_reward += reward
    
            # Use the same fixed-size rollout/update schedule as direct
            # real-environment training. Episode boundaries stay in the buffer.
            if time_step % update_timestep == 0:
                if len(ppo_agent.buffer.rewards) > 1:
                    next_state_norm = utils.normalize_obs(
                        state_0.clone(), hparams_world_model.obs_norm_values
                    ).flatten()
                    bootstrap_value = (
                        0.0
                        if done
                        else ppo_agent.estimate_old_value(next_state_norm)
                    )
                    update_metrics = ppo_agent.update(
                        bootstrap_value=bootstrap_value
                    )
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

            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            budget_exhausted = time_step >= max_training_timesteps
            if done or budget_exhausted:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        print_running_steps += t
        success = int(reached_goal)
        print_running_successes += success
        recent_rewards.append(float(current_ep_reward))
        recent_steps.append(int(t))
        recent_successes.append(success)

        rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)
        rolling_avg_steps = sum(recent_steps) / len(recent_steps)
        rolling_success_rate = sum(recent_successes) / len(recent_successes)

        if sub_run is not None:
            sub_run.log(
                {
                    "episode/reward": current_ep_reward,
                    "episode/success": success,
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

    # Train once more on the partial fixed-size rollout at the budget boundary.
    if len(ppo_agent.buffer.rewards) >= 2:
        next_state_norm = utils.normalize_obs(
            state_0.clone(), hparams_world_model.obs_norm_values
        ).flatten()
        bootstrap_value = (
            0.0 if done else ppo_agent.estimate_old_value(next_state_norm)
        )
        update_metrics = ppo_agent.update(bootstrap_value=bootstrap_value)
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
        "reinit": True,
        "config": {
            "domain": str(cfg.domain),
            "task_name": str(cfg.domains[str(cfg.domain)].task_name),
            "train_in_real_env": bool(ppo_cfg.train_in_real_env),
            "max_ep_len": int(ppo_cfg.max_ep_len),
            "rollout_steps": int(getattr(ppo_cfg, "rollout_steps", 1024)),
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
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=max_ep_len, render_mode=None))
    
    state_dim = np.prod(env.observation_space['image'].shape)

    # action space dimension
    if has_continuous_action_space:
        action_dim = int(np.prod(env.action_space.shape))
    else:
        action_dim = MODEL_ACTION_COUNT

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
