from .common.utils import normalize_obs, ColRowCanl_to_CanlRowCol, WORLD_MODEL_PATH, PROJECT_ROOT, Visualization
from domain.minigrid.minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from path import *
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from tqdm import tqdm
import torch
import wandb


import os
import hydra
from omegaconf import DictConfig
import numpy as np
from multiprocessing import Pool, get_context
import copy
from matplotlib import pyplot as plt
from modelBased.common import utils

def visualize_env(env, cfg: DictConfig, save_img=False):
    env.reset()[0]
    img = env.get_frame()
    return img 


# def run_env(env, cfg: DictConfig, policy=None, rmax_exploration=None):
#     obs_list, obs_next_list, act_list, rew_list, done_list = [], [], [], [], []
#     episodes = 0
#     obs = env.reset()[0]

#     # Use tqdm to provide a progress bar
#     with tqdm(total=cfg.episodes, desc="Collecting Episodes") as pbar:
#         while episodes < cfg.episodes:
#             obs_list.append([obs['image']])
#             if policy is None:
#                 act = env.action_space.sample()  # Restrict the number of actions to 2
#             else:
#                 state_norm = normalize(obs['image']).to(device)
#                 act = policy.select_action(state_norm)
#             obs, reward, done, _, _ = env.step(act)
#             act_list.append([act])
#             obs_next_list.append([obs['image']])
#             rew_list.append([reward])
#             done_list.append([done])

#             # Update RMax visit count and store interaction
#             # if apply rmax exploration
#             if rmax_exploration is not None:
#                 rmax_exploration.update_visit_count(obs['image'], act)


#             if cfg.visualize:  # Set to false to hide the GUI
#                 env.render()
#                 time.sleep(0.1)

#             if done:
#                 episodes += 1
#                 pbar.update(1)  # Update the progress bar

#                 if episodes % 100 == 0:
#                     print("Episode", episodes)
#                 env.reset()

#     obs_np = np.concatenate(obs_list)
#     obs_next_np = np.concatenate(obs_next_list)
#     act_np = np.concatenate(act_list)
#     rew_np = np.concatenate(rew_list)
#     done_np = np.concatenate(done_list)

#     print(obs_np.shape)
#     print(obs_next_np.shape)
#     print(rew_np.shape)
#     print(done_np.shape)
#     print("Num episodes started: ", episodes)

#     return obs_np, obs_next_np, act_np, rew_np, done_np

def run_env_vectorized(env, cfg: DictConfig, wandb_run, policy=None, rmax_exploration=None, save_img=False):
    import copy
    from gym.vector import AsyncVectorEnv
    """
    Vectorized data collection with AsyncVectorEnv, handling variable episode lengths.
    Collects cfg.collect.episodes full episodes per parallel env.
    """
        # set device to cpu or cuda
    device = torch.device('cpu')
    if save_img and wandb_run is not None:
        obs = env.reset()[0]
        img = env.get_frame()
        wandb_run.log({"Mini-tasks": wandb.Image(img)})

    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    num_envs = cfg.collect.num_workers

        # Build factory that deep-copies the single env into independent instances
    def make_env():
        base = copy.deepcopy(env)
        # convert dict observation to image-only Box space
        return ImgObsWrapper(base)

    envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
    # Reset all envs and prepare trackers
    obs_batch = envs.reset()  # shape: (num_envs, H, W, C)
    episodes_done = [0] * num_envs
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right, env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    # Continue until each env has completed desired episodes
    while any(ed < cfg.collect.episodes for ed in episodes_done):
        # Sample actions
        if policy is None:
            acts = np.random.choice(
                meaningful_actions,
                size=num_envs,
                p=[0.3,0.15,0.15,0.2,0.2]
            )
        else:
            # Batch forward
            state_norm = normalize_obs(obs_batch['image']).to(device)
            act = policy.select_action(state_norm)

        # Step all envs
        next_obs, rewards, dones, _, infos = envs.step(acts)

        # Optionally add custom info fields, for example whether agent is carrying a key
        if hasattr(env.unwrapped, "carrying") and env.unwrapped.carrying:
            infos["carrying_key"] = (env.unwrapped.carrying.type == 'key')
        else:
            infos["carrying_key"] = False

                # Record transitions for envs still collecting
        for i in range(num_envs):
            if episodes_done[i] < cfg.collect.episodes:
                obs_list.append(obs_batch[i])
                obs_next_list.append(next_obs[i])
                act_list.append([acts[i]])
                rew_list.append([rewards[i]])
                done_list.append([dones[i]])
                info_list.append(infos[i])
                if dones[i]:
                    episodes_done[i] += 1

        obs_batch = next_obs

    # Convert lists to arrays
    obs_buf      = np.stack(obs_list,      axis=0)
    obs_next_buf = np.stack(obs_next_list, axis=0)
    act_buf      = np.array(act_list,      dtype=np.int32)
    rew_buf      = np.array(rew_list,      dtype=np.float32)
    done_buf     = np.array(done_list,     dtype=bool)
    # infos may be a list of dicts; keep as list or convert to object array
    info_buf     = np.array(info_list,     dtype=object)
    envs.close()
    print(f"Collected: {obs_buf.shape[0]} steps from {num_envs} envs")
    return obs_buf, obs_next_buf, act_buf, rew_buf, done_buf, info_buf


def augment_interactions_keydoor_only(
    obs, obs_next, act, rew, done, info, actions_to_oversample, N=10, shuffle=True
):
    """
    Oversample only relevant 'key-door' interactions involving pickup/toggle and carrying a key.

    Parameters:
        obs: np.ndarray           - current observations
        obs_next: np.ndarray      - next observations
        act: np.ndarray           - actions taken
        rew: np.ndarray           - rewards received
        done: np.ndarray          - episode termination flags
        info: list of dict        - metadata per step (e.g., "carrying_key")
        actions_to_oversample: iterable - actions to target for oversampling
        N: int                    - number of times to repeat each key interaction

    Returns:
        obs_aug, obsn_aug, act_aug, rew_aug, done_aug, info_aug
        Each output is shuffled in unison, and `info` is included in the augmentation.
    """
    # If no oversampling requested, return inputs as-is
    if N <= 1:
        return obs, obs_next, act, rew, done, info

    num = obs.shape[0]
    flat_act = act.reshape(num)

    # Detect any change in observation -> indicates an interaction happened
    changed = np.any(obs != obs_next, axis=tuple(range(1, obs.ndim)))

    # Flag steps where the agent is carrying the key
    keydoor_flags = np.array([i.get("carrying_key", False) for i in info])

    # Build a mask for actions that we want to oversample
    mask_key = np.zeros(num, dtype=bool)
    for a in actions_to_oversample:
        mask_key |= (flat_act == a)

    # Combine masks: action is in the target set, state changed, carrying the key
    mask = mask_key & changed & keydoor_flags

    # Split data into key (to oversample) and normal parts
    obs_key, obsn_key = obs[mask], obs_next[mask]
    act_key, rew_key, done_key = act[mask], rew[mask], done[mask]
    info_key = [info[i] for i, m in enumerate(mask) if m]

    obs_norm, obsn_norm = obs[~mask], obs_next[~mask]
    act_norm, rew_norm, done_norm = act[~mask], rew[~mask], done[~mask]
    info_norm = [info[i] for i, m in enumerate(mask) if not m]

    # Create augmented data: repeat key samples N times, keep normal once
    obs_aug  = np.concatenate([obs_norm] + [obs_key] * N, axis=0)
    obsn_aug = np.concatenate([obsn_norm] + [obsn_key] * N, axis=0)
    act_aug  = np.concatenate([act_norm] + [act_key] * N, axis=0)
    rew_aug  = np.concatenate([rew_norm] + [rew_key] * N, axis=0)
    done_aug = np.concatenate([done_norm] + [done_key] * N, axis=0)
    info_aug = info_norm + info_key * N

    # Shuffle all arrays together to maintain alignment
    if shuffle:
        idx = np.random.permutation(len(obs_aug))
        return (
            obs_aug[idx],
            obsn_aug[idx],
            act_aug[idx],
            rew_aug[idx],
            done_aug[idx],
            [info_aug[i] for i in idx]
        )
    else:
        return (
            obs_aug,
            obsn_aug,
            act_aug,
            rew_aug,
            done_aug,
            info_aug
        )



def run_env_worker(args):
    env_fn, cfg, wandb_run, policy, rmax_exploration, save_img = args
    env = env_fn()  # 每个子进程单独创建自己的环境
    return run_env(env, cfg, wandb_run, policy, rmax_exploration, save_img)

def run_env_multiprocess(cfg, wandb_run, policy=None, rmax_exploration=None, save_img=False, num_workers=4):
    import multiprocessing as mp
    from modelBased.common.utils import get_env

    # 设置多进程启动方式（只需设置一次）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 如果已经设置过就忽略

    # 环境构造函数：每个子进程用它创建独立环境
    env_fn = lambda: get_env(cfg.env.name)

    # 多个子进程的参数列表
    args_list = [(env_fn, cfg, wandb_run, policy, rmax_exploration, save_img) for _ in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_env_worker, args_list)

    # 合并结果
    obs_np, obs_next_np, act_np, rew_np, done_np, info_np = zip(*results)

    return (
        np.concatenate(obs_np),
        np.concatenate(obs_next_np),
        np.concatenate(act_np),
        np.concatenate(rew_np),
        np.concatenate(done_np),
        np.concatenate(info_np),
    )

def run_env(env, cfg: DictConfig, wandb_run, log_name, policy=None, rmax_exploration=None, save_img=False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    episodes = 0
    obs = env.reset()[0]
    has_carried_key_this_episode = False  # 新增：本轮是否已经捡过钥匙
    step_in_episode = 0  # 当前 episode 中的 step 计数器
    task_name = cfg.env.collect.env_type
    maximum_dataset_size = cfg.env.collect.maximum_dataset_size


    if save_img and wandb_run is not None:
        img = env.get_frame()
        wandb_run.log({log_name: wandb.Image(img)})
    # Visit count for RMax or exploration tracking
    visit_count = {}

    if cfg.env.save_visualized_img:
        img = env.get_frame()
        # --- save locally ---
        os.makedirs(cfg.env.visualize_save_path, exist_ok=True)
        save_path = os.path.join(cfg.env.visualize_save_path, f"{log_name}_start.png")

        import matplotlib.pyplot as plt
        plt.imsave(save_path, img)

        print(f"[Saved Frame] {save_path}")

    # Define meaningful actions (forward, turn_left, turn_right)
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right, env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    # Use tqdm for progress tracking
    visual_func = Visualization(cfg.attention_model)
    with tqdm(total=cfg.env.collect.episodes, desc="Collecting Episodes") as pbar:
        info_list.append([{'carrying_key': False}])  

        while episodes < cfg.env.collect.episodes:
            step_in_episode += 1

            obs_list.append([obs['image']])
            # teleport_near_important_tiles(env, obs, step_in_episode, interval=50)

            if "wall" in task_name:
                action_probs = [0.6, 0.2, 0.2, 0.0, 0.0]  # 多往前撞墙
            elif "lava" in task_name:
                action_probs = [0.7, 0.15, 0.15, 0.0, 0.0]  # 多尝试前进（踩lava）
            elif "key" in task_name:
                action_probs = [0.2, 0.15, 0.15, 0.4, 0.1]  # pickup + toggle 多点
            elif "door" in task_name:
                action_probs = [0.1, 0.1, 0.1, 0.3, 0.4]  # 重点在toggle
            else:
                action_probs = [0.3, 0.15, 0.15, 0.2, 0.2]  # 默认
            # Select an action
            if policy is None:
                act = np.random.choice(meaningful_actions, p=action_probs)  # Weighted random sampling
            else:
                state_norm = normalize_obs(obs['image']).to(device)
                act = policy.select_action(state_norm)

            # Step in the environment
            obs_next, reward, done, trunc, info = env.step(act)


            if env.env.carrying and env.env.carrying.type == 'key':
                info['carrying_key'] = True
            else:
                info['carrying_key'] = False
            # check the data
            # if not has_carried_key_this_episode and info['carrying_key']:
            #     tqdm.write(f"[Episode {episodes}] First time carrying key at step {step_in_episode}! Action: {act}")

            #     obs_diff = obs_next['image'].astype(int) - obs['image'].astype(int)
            #     tqdm.write(f"obs_next - obs (nonzero count): {obs_diff}")
            #     # 可选：如果图像小，可以直接打印出差值矩阵
            #     # tqdm.write(f"Diff:\n{obs_diff}")

            #     has_carried_key_this_episode = True
            # before = obs['image'].astype(int)
            # after = obs_next['image'].astype(int)

            # # Mask where door changed from closed (4,4,2) to open (4,4,0)
            # mask = np.all(before == (4, 4, 2), axis=-1) & np.all(after == (4, 4, 0), axis=-1)

            # if np.any(mask):
            #     coords = np.argwhere(mask)  # (row, col) positions
            #     tqdm.write(
            #         f"[Episode {episodes}] Door opened at step {step_in_episode}! "
            #         f"Action: {act}, Position(s): {coords.tolist()}"
            #     )


            # visual_func.visualize_single_state(obs_next['image'], act, info, ep=episodes, index=index,save_flag=True)

            # Collect data
            # mapping toggle to 4 for the PPO training
            act_list.append([act])
            obs_next_list.append([obs_next['image']])   
            rew_list.append([reward])
            done_list.append([done])
            info_list.append([info])

            # --- early stop if dataset size reached ---
            if maximum_dataset_size is not None and len(obs_list) >= maximum_dataset_size:
                print(f"[STOP] Reached maximum dataset size: {maximum_dataset_size}")
                break



            # Update visit count and RMax exploration
            state_action_key = (tuple(obs['image'].flatten()), act)
            visit_count[state_action_key] = visit_count.get(state_action_key, 0) + 1
            if rmax_exploration is not None:
                rmax_exploration.update_visit_count(obs['image'], act)

            # Visualize if needed
            if cfg.env.visualize:
                env.render()
                time.sleep(0.1)

            # Reset environment on episode end
            if done or trunc:
                info_list.pop()
                episodes += 1
                pbar.update(1)
                obs = env.reset()[0]
                info_list.append([{'carrying_key': False}])  
                has_carried_key_this_episode = False  # 重置本轮状态
                step_in_episode = 0
            else:
                obs = obs_next

    info_list.pop()
    # Convert collected data to numpy arrays
    obs_np = np.concatenate(obs_list)
    obs_next_np = np.concatenate(obs_next_list)
    act_np = np.concatenate(act_list)
    # preprocess actions: map toggle (5) to drop (4) just for the model training --> don't forget to change it back during policy training
    act_np[act_np == 5] = 4  
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)
    info_np = np.concatenate(info_list)

    # Log statistics
    print(f"Observation shape: {obs_np.shape}")
    print(f"Next observation shape: {obs_next_np.shape}")
    print(f"Actions shape: {act_np.shape}")
    print(f"Rewards shape: {rew_np.shape}")
    print(f"Dones shape: {done_np.shape}")
    print(f"Number of episodes started: {episodes}")
    print(f"Unique state-action pairs visited: {len(visit_count)}")
    env.close()

    return obs_np, obs_next_np, act_np, rew_np, done_np, info_np


def teleport_near_important_tiles(env, obs, step_in_episode, interval):
    """
    Every `interval` steps, teleport agent to an EMPTY tile near key/door/lava.
    """
    if step_in_episode % interval != 0:
        return

    import random
    import numpy as np

    grid_type = obs['image'][:, :, 0]  # tile type channel

    H, W = grid_type.shape

    KEY = 5
    DOOR = 4
    LAVA = 9
    EMPTY = 1  # empty floor tile

    # find important tiles
    important_tiles = list(zip(*np.where(
        (grid_type == KEY) | 
        (grid_type == DOOR) | 
        (grid_type == LAVA)
    )))

    if len(important_tiles) == 0:
        return  # minitask has no key/door/lava

    # pick random important tile
    tx, ty = random.choice(important_tiles)

    # neighbor offsets (4-direction)
    offsets = [(-1,0), (1,0), (0,-1), (0,1)]

    # only choose neighbors that are EMPTY
    candidates = []
    for dx, dy in offsets:
        nx, ny = tx + dx, ty + dy

        # boundaries
        if not (0 <= nx < H and 0 <= ny < W):
            continue

        # must be empty tile
        if grid_type[nx, ny] == EMPTY:
            candidates.append((nx, ny))

    # no valid teleport positions
    if len(candidates) == 0:
        return

    # random empty neighbor cell
    new_x, new_y = random.choice(candidates)

    # Teleport agent safely
    env.unwrapped.agent_pos = np.array([new_x, new_y])

    # randomize agent direction to avoid facing outside
    env.unwrapped.agent_dir = random.randint(0, 3)


def augment_uniform_dataset(obs_np, obs_next_np, act_np, rew_np, done_np, info_np, target_size):
    """
    Uniformly resample (with replacement) to reach target_size.
    If target_size <= current number of samples, return as-is.
    """
    N = obs_np.shape[0]
    if target_size <= N:
        print(f"[augment_uniform_dataset] target_size ({target_size}) <= current size ({N}), no augmentation.")
        return obs_np, obs_next_np, act_np, rew_np, done_np, info_np

    idx = np.random.randint(0, N, size=target_size)
    print(f"[augment_uniform_dataset] Expanding dataset from {N} → {target_size} via uniform resampling.")
    return (
        obs_np[idx],
        obs_next_np[idx],
        act_np[idx],
        rew_np[idx],
        done_np[idx],
        info_np[idx],
    )


def run_env_uniform(env, cfg, wandb_run, log_name, policy=None, rmax_exploration=None, save_img=False):
    """
    Uniformly traverse the environment in (x, y, dir, action) space
    to collect transition data.

    - Uses cfg.env.collect.episodes as the target number of transitions.
    - If target < coverage → early stop
    - If target > coverage → uniform resampling (augmentation)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    width, height = env.width, env.height
    target_samples = getattr(cfg.env.collect, "episodes", 10000)
    step_count = 0

    meaningful_actions = [
        env.unwrapped.actions.forward,
        env.unwrapped.actions.left,
        env.unwrapped.actions.right,
        env.unwrapped.actions.pickup,
        env.unwrapped.actions.toggle
    ]

    # Optional W&B image logging
    if save_img and wandb_run is not None:
        try:
            img = env.get_frame()
            wandb_run.log({log_name: wandb.Image(img)})
        except Exception:
            pass

    info_list.append([{'carrying_key': False}])
    print(f"Collecting uniform transitions from {width}x{height} grid, target={target_samples} samples.")

    for x in tqdm(range(width), desc="X grid"):
        for y in range(height):
            for dir in range(4):  # 4 facing directions
                if step_count >= target_samples:
                    print(f"Reached target={target_samples}, stopping early.")
                    break

                env.unwrapped.agent_pos = (x, y)
                env.unwrapped.agent_dir = dir
                try:
                    obs_dict = env.reset()[0]
                    obs = obs_dict["image"]
                except Exception:
                    continue

                for a in meaningful_actions:
                    if step_count >= target_samples:
                        break

                    backup_pos = np.array(env.unwrapped.agent_pos)
                    backup_dir = int(env.unwrapped.agent_dir)

                    obs_list.append([obs])
                    act_list.append([a])

                    try:
                        obs_next_dict, reward, done, trunc, info = env.step(a)
                        obs_next = obs_next_dict["image"]

                        # Carrying key info (consistent with run_env)
                        if env.env.carrying and env.env.carrying.type == 'key':
                            info["carrying_key"] = True
                        else:
                            info["carrying_key"] = False

                        obs_next_list.append([obs_next])
                        rew_list.append([reward])
                        done_list.append([done])
                        info_list.append([info])

                    except Exception as e:
                        obs_next_list.append([obs])
                        rew_list.append([0.0])
                        done_list.append([True])
                        info_list.append([{'error': str(e), 'carrying_key': False}])

                    # Restore agent state
                    env.unwrapped.agent_pos = tuple(backup_pos)
                    env.unwrapped.agent_dir = backup_dir

                    step_count += 1
                    if step_count >= target_samples:
                        break
            if step_count >= target_samples:
                break
        if step_count >= target_samples:
            break

    info_list.pop()

    # Convert to numpy arrays
    obs_np = np.concatenate(obs_list)
    obs_next_np = np.concatenate(obs_next_list)
    act_np = np.concatenate(act_list)
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)
    info_np = np.concatenate(info_list)

    # Map toggle (5) → pickup (4)
    if np.any(act_np == 5):
        print("Mapping action 5 (toggle) → 4 (pickup) for consistency.")
        act_np[act_np == 5] = 4

    # Theoretical coverage size
    full_coverage = width * height * 4 * len(meaningful_actions)
    print(f"Collected {obs_np.shape[0]} transitions (full coverage={full_coverage})")

    # Augment if user wants more than full coverage
    if target_samples > obs_np.shape[0]:
        obs_np, obs_next_np, act_np, rew_np, done_np, info_np = augment_uniform_dataset(
            obs_np, obs_next_np, act_np, rew_np, done_np, info_np, target_size=target_samples
        )

    print("\n=== Uniform Collection Summary ===")
    print(f"Final transitions: {obs_np.shape[0]}")
    print(f"Observation shape: {obs_np.shape}, Next obs shape: {obs_next_np.shape}")
    print(f"Unique actions: {np.unique(act_np)}")

    env.close()
    return obs_np, obs_next_np, act_np, rew_np, done_np, info_np

def uniformize_dataset_by_position(obs, obs_next, act, rew, done, info):
    """
    Post-process a randomly collected dataset into a position-balanced one.
    Keeps real transitions, only reweights by spatial coverage.
    """
    from collections import defaultdict
    obs_trans = obs.transpose(0, 3, 1, 2)
    positions = utils.get_agent_position(obs_trans)
    buckets = defaultdict(list)
    for i, pos in enumerate(positions):
        key = tuple(pos)
        if key != (-1, -1):
            buckets[key].append(i)

    min_size = min(len(v) for v in buckets.values() if len(v) > 0)
    selected_idx = []
    for v in buckets.values():
        if len(v) >= min_size:
            selected_idx.append(np.random.choice(v, size=min_size, replace=False))
    idx = np.concatenate(selected_idx)
    np.random.shuffle(idx)

    print(f"[uniformize_dataset_by_position] Total positions={len(buckets)}, "
          f"min_size per pos={min_size}, total kept={len(idx)}")

    return (
        obs[idx],
        obs_next[idx],
        act[idx],
        rew[idx],
        done[idx],
        info[idx],
    )

def uniform_collect_data_postprocess(env, cfg, wandb_run, log_name, policy=None, rmax_exploration=None, save_img=False):
    obs, obs_next, act, rew, done, info = run_env(env, cfg, wandb_run, log_name, save_img=save_img)
    obs_u, obsn_u, act_u, rew_u, done_u, info_u = uniformize_dataset_by_position(
                                                obs, obs_next, act, rew, done, info
                                                )
    return obs_u, obsn_u, act_u, rew_u, done_u, info_u


def save_experiments(cfg: DictConfig, obs, obs_next, act, rew, done, info=None):
    obs = ColRowCanl_to_CanlRowCol(obs)
    obs_next = ColRowCanl_to_CanlRowCol(obs_next)
    np.savez_compressed(cfg.collect.data_save_path, a=obs, b=obs_next, c=act, d=rew, e=done, f=info)

def data_augmentation(cfg: DictConfig, obs, obs_next, act, rew, done):
    """
    Adding more forward and turning data for empty env
    """
    obs_aug = np.concatenate([obs, obs_next])
    obs_next_aug = np.concatenate([obs_next, obs])
    act_aug = np.concatenate([act, act])
    rew_aug = np.concatenate([rew, rew])
    done_aug = np.concatenate([done, done])

    # Shuffle the data
    idx = np.random.permutation(len(obs_aug))
    obs_aug = obs_aug[idx]
    obs_next_aug = obs_next_aug[idx]
    act_aug = act_aug[idx]
    rew_aug = rew_aug[idx]
    done_aug = done_aug[idx]

    print("Data Augmented")
    print(obs_aug.shape)
    print(obs_next_aug.shape)
    print(rew_aug.shape)
    print(done_aug.shape)

    return obs_aug, obs_next_aug, act_aug, rew_aug, done_aug


def sample_keydoor_pref(
    obs, obs_next, act, rew, done, info,
    key_repeat=5,            
    move_keep_ratio=0.2      
):

    num = obs.shape[0]
    flat_act = act.reshape(num)
    KEYDOOR_ACTIONS = [env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    is_keydoor = np.zeros(num, dtype=bool)
 
    for a in KEYDOOR_ACTIONS:
        is_keydoor |= (flat_act == a)

    is_keydoor |= np.array([i.get("carrying_key", False) for i in info])

    kd_idx  = np.where(is_keydoor)[0]
    mov_idx = np.where(~is_keydoor)[0]

  
    obs_kd      = np.repeat(obs[kd_idx],      key_repeat, axis=0)
    obsn_kd     = np.repeat(obs_next[kd_idx], key_repeat, axis=0)
    act_kd      = np.repeat(act[kd_idx],      key_repeat, axis=0)
    rew_kd      = np.repeat(rew[kd_idx],      key_repeat, axis=0)
    done_kd     = np.repeat(done[kd_idx],     key_repeat, axis=0)
    info_kd     = [info[i] for i in kd_idx for _ in range(key_repeat)]


    keep_mov = np.random.rand(len(mov_idx)) < move_keep_ratio
    mov_keep_idx = mov_idx[keep_mov]

    obs_mov      = obs[mov_keep_idx]
    obsn_mov     = obs_next[mov_keep_idx]
    act_mov      = act[mov_keep_idx]
    rew_mov      = rew[mov_keep_idx]
    done_mov     = done[mov_keep_idx]
    info_mov     = [info[i] for i in mov_keep_idx]

 
    obs_aug  = np.concatenate([obs_kd,  obs_mov ], axis=0)
    obsn_aug = np.concatenate([obsn_kd, obsn_mov], axis=0)
    act_aug  = np.concatenate([act_kd,  act_mov ], axis=0)
    rew_aug  = np.concatenate([rew_kd,  rew_mov ], axis=0)
    done_aug = np.concatenate([done_kd, done_mov], axis=0)
    info_aug = info_kd + info_mov

    idx = np.random.permutation(len(obs_aug))
    return (
        obs_aug[idx],
        obsn_aug[idx],
        act_aug[idx],
        rew_aug[idx],
        done_aug[idx],
        [info_aug[i] for i in idx]
    )

def filter_keydoor_only(env, obs, obs_next, act, rew, done, info, move_keep_ratio=0.2):
    """
    保留与 key/door 有关的交互行为，丢弃大部分 random move。
    - keydoor: pickup / toggle / carrying_key=True
    - move: 其他动作，仅保留一定比例
    """
    num = obs.shape[0]
    flat_act = act.reshape(num)

    # 关键动作（key, door交互）
    KEYDOOR_ACTIONS = [env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    is_keydoor = np.zeros(num, dtype=bool)
    for a in KEYDOOR_ACTIONS:
        is_keydoor |= (flat_act == a)

    # carrying key 的步骤也保留
    is_keydoor |= np.array([i.get("carrying_key", False) for i in info])

    # Movement action → 剩余的全是移动
    move_idx = np.where(~is_keydoor)[0]
    keep_move = np.random.rand(len(move_idx)) < move_keep_ratio
    move_keep_idx = move_idx[keep_move]

    # 保留的关键交互
    keydoor_idx = np.where(is_keydoor)[0]

    # 合并最终保留的 index
    final_idx = np.concatenate([keydoor_idx, move_keep_idx])
    np.random.shuffle(final_idx)

    return (
        obs[final_idx],
        obs_next[final_idx],
        act[final_idx],
        rew[final_idx],
        done[final_idx],
        [info[i] for i in final_idx]
    )

def _keydoor_mask(env, obs, obs_next, act, info, require_changed=True):
    N = obs.shape[0]
    flat_act = act.reshape(N)

    A_PICKUP = env.unwrapped.actions.pickup
    A_TOGGLE = env.unwrapped.actions.toggle

    is_kd = (flat_act == A_PICKUP) | (flat_act == A_TOGGLE)
    carry_key = np.array([bool(i.get("carrying_key", False)) for i in (list(info) if not isinstance(info, list) else info)])
    is_kd |= carry_key

    if require_changed:
        changed = np.any(obs != obs_next, axis=tuple(range(1, obs.ndim)))
        is_kd &= changed

    return is_kd


def downsample_moves_only(
    env,
    obs, obs_next, act, rew, done, info,
    move_keep_ratio,     # 仅对“非 key/door”样本随机保留这部分比例
    require_changed=True,     # 只把真正改变状态的样本视为 key/door
    min_keep_moves=1,         # 至少保留这么多移动样本，防止空
    shuffle=True
):
    """
    downsampling the data which is just moving around,
    to improving the training of the interact with key-door info
    """
    # 统一 info 为 list
    info_list = info if isinstance(info, list) else list(info)

    is_keydoor = _keydoor_mask(env, obs, obs_next, act, info_list, require_changed=require_changed)
    kd_idx  = np.where(is_keydoor)[0]
    mov_idx = np.where(~is_keydoor)[0]

    # 若没有 key/door 样本（例如 Empty），直接按比例下采移动；否则仅对 mov 下采
    if kd_idx.size == 0:
        keep_mask = np.random.rand(mov_idx.size) < move_keep_ratio
        mov_keep_idx = mov_idx[keep_mask]
        if mov_keep_idx.size < min_keep_moves:
            mov_keep_idx = mov_idx[:min(mov_idx.size, max(min_keep_moves, 1))]
        final_idx = mov_keep_idx
    else:
        keep_mask = np.random.rand(mov_idx.size) < move_keep_ratio
        mov_keep_idx = mov_idx[keep_mask]
        if mov_keep_idx.size < min_keep_moves:
            extra = mov_idx[:min(mov_idx.size, max(min_keep_moves - mov_keep_idx.size, 0))]
            mov_keep_idx = np.unique(np.concatenate([mov_keep_idx, extra]))
        final_idx = np.concatenate([kd_idx, mov_keep_idx])

    if final_idx.size == 0:
        # 兜底：至少保留一条
        final_idx = np.array([0])

    if shuffle:
        np.random.shuffle(final_idx)

    obs_out   = obs[final_idx]
    obsn_out  = obs_next[final_idx]
    act_out   = act[final_idx]
    rew_out   = rew[final_idx]
    done_out  = done[final_idx]
    info_out  = [info_list[i] for i in final_idx]

    return obs_out, obsn_out, act_out, rew_out, done_out, info_out



@hydra.main(version_base=None, config_path = str(WORLD_MODEL_PATH / "config"), config_name="config")
def data_collect(cfg: DictConfig):
    hparam = cfg.env
    mode =None
    if hparam.visualize:
        mode = 'human'
    env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=hparam.env_path, 
                                        custom_mission="Find the key and open the door.",
                                        max_steps=10000, render_mode=mode))
    obs, obs_next, act,rew, done = run_env(env, hparam, log_name="train", wandb_run=None, save_img=False)
    save_experiments(cfg.env,obs,obs_next, act, rew, done)
    env.close()


def data_collect_api_multiprocess(cfg: DictConfig, env, wandb_run, save_img=False):
    hparam = cfg.env
    obs, obs_next, act, rew, done, info = run_env_multiprocess(env, hparam, wandb_run, save_img=save_img)
    save_experiments(cfg.env,obs,obs_next, act, rew, done, info)

def data_collect_api(cfg: DictConfig, env, wandb_run, save_img, log_name, max_steps=10000):
    hparam = copy.deepcopy(cfg)
    original_episodes = hparam.env.collect.episodes

    # === DATA BUFFERS ===
    obs_all, obsn_all, act_all, rew_all, done_all, info_all = [], [], [], [], [], []

    # ------------------------------------------------------
    # CASE 1: Single-run mode (NO ITERATION)
    # ------------------------------------------------------
    if max_steps is None:
        print(f"[Single-run mode] Collecting {hparam.env.collect.episodes} episodes...")

        if cfg.env.collect.data_type == 'uniform':
            obs, obs_next, act, rew, done, info = uniform_collect_data_postprocess(
                env, hparam, wandb_run, log_name,
                policy=None, rmax_exploration=None, save_img=save_img
            )
        else:
            obs, obs_next, act, rew, done, info = run_env(
                env, hparam, wandb_run, log_name, save_img=save_img
            )

        # store results
        obs_all.append(obs)
        obsn_all.append(obs_next)
        act_all.append(act)
        rew_all.append(rew)
        done_all.append(done)
        info_all.append(info)

        print("[Single-run] Finished collecting. Saving dataset...")
            # jump to the saving section
        return _finalize_and_save(cfg, env, obs_all, obsn_all, act_all, rew_all, done_all, info_all)
    

    # ------------------------------------------------------
    # CASE 2: Multi-round mode (original behavior)
    # ------------------------------------------------------
    total_steps = 0
    round_idx = 0

    while total_steps < max_steps:
        print(f"Round {round_idx+1}, collecting {hparam.env.collect.episodes} episodes...")

        if cfg.env.collect.data_type == 'uniform':
            obs, obs_next, act, rew, done, info = uniform_collect_data_postprocess(
                env, hparam, wandb_run, log_name,
                policy=None, rmax_exploration=None, save_img=False
            )
        else:
            obs, obs_next, act, rew, done, info = run_env(
                env, hparam, wandb_run, log_name, save_img=save_img
            )

        obs_all.append(obs)
        obsn_all.append(obs_next)
        act_all.append(act)
        rew_all.append(rew)
        done_all.append(done)
        info_all.append(info)

        total_steps += len(obs)
        print(f"Total steps collected: {total_steps}")

        if total_steps < max_steps:
            hparam.env.collect.episodes = max(1, original_episodes // 3)

        round_idx += 1
        save_img = False

    return _finalize_and_save(cfg, env, obs_all, obsn_all, act_all, rew_all, done_all, info_all)



# ----------------------------------------------------------
# Helper: final save + visualization
# ----------------------------------------------------------
def _finalize_and_save(cfg, env, obs_all, obsn_all, act_all, rew_all, done_all, info_all):

    # merge
    obs_all = np.concatenate(obs_all, axis=0)
    obsn_all = np.concatenate(obsn_all, axis=0)
    act_all = np.concatenate(act_all, axis=0)
    rew_all = np.concatenate(rew_all, axis=0)
    done_all = np.concatenate(done_all, axis=0)
    info_all = np.concatenate(info_all, axis=0)

    print(f"Final data shape: {obs_all.shape}")

    save_experiments(cfg.env, obs_all, obsn_all, act_all, rew_all, done_all, info_all)

    # visualization
    data_path = cfg.env.collect.data_save_path
    if cfg.env.collect.save_coverage_visualize and os.path.exists(data_path):
        filename = getattr(cfg.env.collect, "visualize_filename", None)
        save_path = os.path.join(cfg.env.collect.visualize_save_path, filename)
        data = np.load(data_path, allow_pickle=True)

        visualize_agent_coverage(
            data,
            save_path=save_path,
            title=f"Agent Position Coverage ({cfg.env.collect.data_type})"
        )

    env.close()




def visualize_agent_coverage(data, save_path=None, title="Agent Position Coverage"):
    """
    可视化数据集中 agent 的位置覆盖热力图
    - 自动调用 get_agent_position() 提取 agent 坐标
    """

    # -------------------------------
    # Step 1: 调用已有函数提取位置
    # -------------------------------
    obs_np = data['a']
    # 如果是 (N, H, W, C) 则转换，否则保持不变
    if obs_np.shape[1] not in [3, 4, 5]:  # 通道数异常 -> 应该是放在最后
        obs_np = np.moveaxis(obs_np, -1, 1)
    positions = utils.get_agent_position(obs_np)  # (N, 2)

    # 自动推断地图大小
    if isinstance(obs_np, torch.Tensor):
        obs_np = obs_np.detach().cpu().numpy()
    if len(obs_np.shape) != 4:
        raise ValueError(f"Input must be (N, C, H, W), but got {obs_np.shape}")
    H, W = obs_np.shape[2], obs_np.shape[3]

    # -------------------------------
    # Step 2: 统计访问次数
    # -------------------------------
    heatmap = np.zeros((H, W))
    for (y, x) in positions:
        if 0 <= y < H and 0 <= x < W:
            heatmap[y, x] += 1

    # -------------------------------
    # Step 3: 绘图
    # -------------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap="viridis", origin="upper")
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.colorbar(label="Occurrences")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

def visualize_saved_dataset(data_path, save_path, fig_name):
    '''
    Visualize the saved dataset coverage heatmap
    '''
    if os.path.exists(data_path):
        print(f"Visualizing coverage from dataset: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        # --- use custom filename from config if available ---
        visualize_agent_coverage(
            data,
            save_path=save_path,
            title=f"Agent Position Coverage ({fig_name})"
        )
        print(f"Coverage heatmap saved to {save_path}")

if __name__ == "__main__": 
    data_collect() 
