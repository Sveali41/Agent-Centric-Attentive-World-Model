import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Package import (e.g., `python -m modelBased.data.data_collect`)
    from ..common.utils import normalize_obs, WORLD_MODEL_PATH, PROJECT_ROOT
except ImportError:
    # Script import (e.g., `python modelBased/data/data_collect.py`)
    from modelBased.common.utils import normalize_obs, WORLD_MODEL_PATH, PROJECT_ROOT
from modelBased.common.dataset_identity import metadata_array
from domain.minigrid.action_codec import (
    carrying_token_from_env,
    compact_to_native,
    native_to_compact,
)
from domain.minigrid.minigrid_support import (
    ColRowCanl_to_CanlRowCol,
    Visualization,
    get_agent_position,
)
from domain.minigrid.minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
try:
    from domain.bipedalwalker.bipedalwalker_support import (
        wrap_env_from_text as wrap_bipedal_env_from_text,
    )
    from domain.bipedalwalker.bipedalwalker_support import read_layout_from_txt
    from domain.bipedalwalker.bipedal_walker_custom import (
        TERRAIN_GRASS,
        TERRAIN_LENGTH,
        TERRAIN_STARTPAD,
        TERRAIN_STEP,
    )
except Exception as e:
    wrap_bipedal_env_from_text = None
    read_layout_from_txt = None
    TERRAIN_GRASS = TERRAIN_LENGTH = TERRAIN_STARTPAD = TERRAIN_STEP = None
    _BIPEDAL_IMPORT_ERROR = e
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
from collections import deque
from multiprocessing import Pool, get_context
import copy
from matplotlib import pyplot as plt
from gymnasium.wrappers import TimeLimit


def _coerce_render_image(img):
    """
    Best-effort normalization of render outputs to an image ndarray.
    Some env wrappers return nested tuples/lists or object arrays.
    """
    depth = 0
    while isinstance(img, (tuple, list)) and len(img) > 0 and depth < 8:
        img = img[0]
        depth += 1

    if isinstance(img, np.ndarray) and img.dtype == object and img.size > 0:
        probe = img.reshape(-1)[0]
        if isinstance(probe, (tuple, list, np.ndarray)):
            img = probe
            depth = 0
            while isinstance(img, (tuple, list)) and len(img) > 0 and depth < 8:
                img = img[0]
                depth += 1

    if not isinstance(img, np.ndarray):
        try:
            img = np.asarray(img)
        except Exception:
            return None

    if not isinstance(img, np.ndarray):
        return None

    if img.dtype == object or img.ndim < 2:
        return None

    return img


class BipedalMixedPolicy:
    expects_raw_obs = True

    def __init__(self, expert_policy, heuristic_policy, expert_ratio=0.5):
        self.expert_policy = expert_policy
        self.heuristic_policy = heuristic_policy
        self.expert_ratio = float(expert_ratio)
        self.current_policy = heuristic_policy

    def reset(self):
        # Decide which policy to use for the next episode
        if np.random.random() < self.expert_ratio:
            self.current_policy = self.expert_policy
        else:
            self.current_policy = self.heuristic_policy
        
        if hasattr(self.current_policy, "reset"):
            return self.current_policy.reset()
        return None

    def select_action(self, obs, add_noise=True):
        return self.current_policy.select_action(obs, add_noise=add_noise)


class SB3BipedalPolicy:
    expects_raw_obs = True

    def __init__(self, model, action_space, noise_std=0.15, deterministic=False):
        self.model = model
        self.action_space = action_space
        self.noise_std = float(noise_std)
        self.deterministic = bool(deterministic)

    def reset(self):
        # Stateless wrapper to match the heuristic policy interface.
        return None

    def select_action(self, obs, add_noise=True):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        if add_noise and self.noise_std > 0:
            action = action + np.random.normal(0.0, self.noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)


def _resolve_coverage_vis_target(cfg):
    collect_cfg = cfg.env.collect if hasattr(cfg.env, "collect") else cfg.env
    filename = getattr(collect_cfg, "visualize_filename", "coverage.png")
    vis_save_path = getattr(collect_cfg, "visualize_save_path", "outputs/dataset_visualization")
    env_type = str(getattr(cfg.env, "env_type", "")).lower()
    data_type = str(getattr(collect_cfg, "data_type", "random")).lower()

    if env_type == "bipedalwalker":
        stem, ext = os.path.splitext(filename)
        if data_type and data_type not in stem:
            if stem.endswith("_coverage"):
                stem = stem[: -len("_coverage")] + f"_{data_type}_coverage"
            else:
                stem = f"{stem}_{data_type}"
        filename = f"{stem}{ext or '.png'}"

        domain_name = str(getattr(cfg, "domain", env_type))
        task_group = None
        if hasattr(cfg, "domains") and domain_name in cfg.domains:
            task_group = str(getattr(cfg.domains[domain_name], "task_group", ""))
        if task_group == "target":
            vis_save_path = os.path.join(
                PROJECT_ROOT,
                "trainer",
                "logs",
                "dataset_visualization",
                "target",
                "bipedal",
            )

    return filename, vis_save_path


def _build_bipedal_behavior_policy(cfg, env):
    from domain.bipedalwalker.bipedalwalker_support import BipedalHeuristicPolicy

    if not hasattr(cfg, "domains") or "bipedalwalker" not in cfg.domains:
        print("[Policy] behavior_policy=heuristic | source=default_fallback | prior_weight=0.3")
        return BipedalHeuristicPolicy(prior_weight=0.3)

    domain_cfg = cfg.domains.bipedalwalker
    prior_weight = float(getattr(domain_cfg, "bipedal_prior_weight", 0.3))

    # Auto-select: uniform collection -> always use pretrained_sb3 for better coverage
    collect_cfg = getattr(cfg.env, "collect", cfg.env) if hasattr(cfg, "env") else None
    data_type = str(getattr(collect_cfg, "data_type", "random")).lower() if collect_cfg else "random"
    if data_type == "uniform":
        behavior_policy = "pretrained_sb3"
        print("[Policy] data_type=uniform -> auto-selecting pretrained_sb3 policy for BipedalWalker.")
    else:
        behavior_policy = str(getattr(domain_cfg, "behavior_policy", "heuristic")).lower()

    if behavior_policy == "heuristic":
        print(
            "[Policy] behavior_policy=heuristic | "
            f"prior_weight={prior_weight}"
        )
        return BipedalHeuristicPolicy(prior_weight=prior_weight)

    def load_sb3_policy():
        model_path = getattr(domain_cfg, "sb3_model_path", None)
        algo_name = str(getattr(domain_cfg, "sb3_algo", "SAC")).upper()
        deterministic = bool(getattr(domain_cfg, "sb3_deterministic", False))
        noise_std = float(getattr(domain_cfg, "sb3_noise_std", 0.15))

        try:
            if algo_name == "SAC":
                from stable_baselines3 import SAC as SB3Algo
            elif algo_name == "PPO":
                from stable_baselines3 import PPO as SB3Algo
            else:
                raise ValueError(f"Unsupported SB3 algo for bipedal behavior policy: {algo_name}")
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is required for behavior_policy='pretrained_sb3' or 'mix'. "
                "Install it or use 'heuristic'. "
                f"Original import error: {exc}"
            ) from exc

        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SB3 model not found: {model_path}")
            model = SB3Algo.load(model_path, device='cpu')
            source_desc = f"local:{model_path}"
        else:
            try: from huggingface_sb3 import load_from_hub
            except ImportError as exc: raise ImportError("huggingface_sb3 required to auto-download model.")
            repo_id = getattr(domain_cfg, "sb3_repo_id", "sb3/sac-BipedalWalkerHardcore-v3")
            filename = getattr(domain_cfg, "sb3_filename", "sac-BipedalWalkerHardcore-v3.zip")
            custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
            checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
            model = SB3Algo.load(checkpoint, custom_objects=custom_objects, device='cpu')
            source_desc = f"hf:{repo_id}/{filename}"

        print(
            "[Policy] behavior_policy=pretrained_sb3 | "
            f"algo={algo_name} | "
            f"deterministic={deterministic} | "
            f"noise_std={noise_std} | "
            f"source={source_desc}"
        )

        return SB3BipedalPolicy(
            model=model,
            action_space=env.action_space,
            noise_std=noise_std,
            deterministic=deterministic,
        )

    if behavior_policy == "pretrained_sb3":
        return load_sb3_policy()

    if behavior_policy == "mix":
        expert_ratio = float(getattr(domain_cfg, "expert_ratio", 0.5))
        expert_policy = load_sb3_policy()
        heuristic_policy = BipedalHeuristicPolicy(prior_weight=prior_weight)
        print(
            "[Policy] behavior_policy=mix | "
            f"expert_ratio={expert_ratio} | "
            f"heuristic_prior_weight={prior_weight}"
        )
        return BipedalMixedPolicy(expert_policy, heuristic_policy, expert_ratio=expert_ratio)

    return BipedalHeuristicPolicy(prior_weight=prior_weight)

def visualize_env(env, cfg: DictConfig, save_img=False):
    env.reset()[0]
    try:
        img = env.get_frame()
    except AttributeError:
        img = env.unwrapped.get_frame()
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
        try:
             img = env.get_frame()
        except AttributeError:
             img = env.unwrapped.get_frame()
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
    meaningful_actions = [
        env.unwrapped.actions.forward,
        env.unwrapped.actions.left,
        env.unwrapped.actions.right,
        env.unwrapped.actions.pickup,
        env.unwrapped.actions.toggle,
        env.unwrapped.actions.drop,
    ]

    # Continue until each env has completed desired episodes
    while any(ed < cfg.collect.episodes for ed in episodes_done):
        # Sample actions
        if policy is None:
            acts = np.random.choice(
                meaningful_actions,
                size=num_envs,
                p=[1.0 / len(meaningful_actions)] * len(meaningful_actions)
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
    env = env_fn()  # Each subprocess creates its own environment instance.
    return run_env(env, cfg, wandb_run, policy, rmax_exploration, save_img)

def run_env_multiprocess(cfg, wandb_run, policy=None, rmax_exploration=None, save_img=False, num_workers=4):
    import multiprocessing as mp
    from modelBased.common.utils import get_env

    # Configure the multiprocessing start method once.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Ignore the error if it was already configured.

    # Environment factory used by each subprocess.
    env_fn = lambda: get_env(cfg.env.name)

    # Argument list for all subprocess workers.
    args_list = [(env_fn, cfg, wandb_run, policy, rmax_exploration, save_img) for _ in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_env_worker, args_list)

    # Merge worker outputs.
    obs_np, obs_next_np, act_np, rew_np, done_np, info_np = zip(*results)

    return (
        np.concatenate(obs_np),
        np.concatenate(obs_next_np),
        np.concatenate(act_np),
        np.concatenate(rew_np),
        np.concatenate(done_np),
        np.concatenate(info_np),
    )

def run_env(
    env,
    cfg: DictConfig,
    wandb_run,
    log_name,
    policy=None,
    rmax_exploration=None,
    save_img=False,
    randomize_inventory=False,
    intrinsic_reward_fn=None,
    store_intrinsic_reward=True,
):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    inv_list_cur, inv_list_next = [], []  # Crafter only: inventory (16-dim)
    episodes = 0
    obs = env.reset()[0]
    has_carried_key_this_episode = False  # Track whether a key was carried in this episode.
    step_in_episode = 0  # Step counter within the current episode.
    max_x_in_episode = 0.0  # BipedalWalker only: Odometry tracking
    max_x_global = 0.0  # BipedalWalker only: max distance across the full collection run
    collect_cfg = cfg.env.collect if hasattr(cfg, "env") else cfg.collect
    if hasattr(collect_cfg, "env_type") and collect_cfg.env_type:
        task_name = str(collect_cfg.env_type).lower()
    elif hasattr(cfg, "env") and hasattr(cfg.env, "env_type"):
        task_name = str(cfg.env.env_type).lower()
    else:
        task_name = ""
    is_crafter = ("crafter" in task_name)
    is_bipedal = ("bipedal" in task_name)
    is_minigrid = (not is_crafter) and (not is_bipedal)
    data_type = str(getattr(collect_cfg, "data_type", "")).lower()
    maximum_dataset_size = getattr(collect_cfg, "maximum_dataset_size", None)
    domain_cfg = getattr(getattr(cfg, "domains", None), "bipedalwalker", None) if is_bipedal else None

    def _minigrid_carrying_token():
        """Return 0 for empty hands, otherwise key-colour-id + 1."""
        return carrying_token_from_env(env) if is_minigrid else 0

    if policy is not None and hasattr(policy, "begin_rollout"):
        rollout_budget = (
            maximum_dataset_size
            if maximum_dataset_size is not None
            else getattr(collect_cfg, "mini_dataset_size", 0)
        )
        policy.begin_rollout(rollout_budget)

    # --- 1. Exhaustive Shuffled Queue for 100% Coverage ---

    ground_tiles_queue = []
    if randomize_inventory and is_crafter:
        cg = env.unwrapped.char_grid
        h_grid, w_grid = cg.shape
        # Collect all non-water tiles.
        all_terrestrial = [(c, r) for r in range(h_grid) for c in range(w_grid) if cg[r,c] != 'W']
        np.random.shuffle(all_terrestrial)
        ground_tiles_queue = all_terrestrial
        print(f"[RunEnv] Exhaustive Queue initialized with {len(ground_tiles_queue)} unique spawn points.")

    def get_next_spawn_point():
        nonlocal ground_tiles_queue
        if not ground_tiles_queue:
            return None
        # Cycle through the queue by popping from the front.
        pt = ground_tiles_queue.pop(0)
        # Re-append to keep the queue exhaustive across repeated passes.
        ground_tiles_queue.append(pt)
        return pt

    def _maybe_add_bipedal_spawn(reset_kwargs):
        if not (is_bipedal and data_type == "uniform"):
            return reset_kwargs
        fixed_start_ratio = float(getattr(domain_cfg, "uniform_fixed_start_ratio", 0.3))
        fixed_start_ratio = min(max(fixed_start_ratio, 0.0), 1.0)
        use_fixed_start = np.random.random() < fixed_start_ratio
        if use_fixed_start:
            return reset_kwargs

        max_x = (TERRAIN_LENGTH - 10) * TERRAIN_STEP
        spawn_x = np.random.uniform(TERRAIN_STARTPAD * TERRAIN_STEP, max_x)
        reset_kwargs = dict(reset_kwargs)
        if "options" in reset_kwargs and reset_kwargs["options"] is not None:
            reset_kwargs["options"] = dict(reset_kwargs["options"])
        else:
            reset_kwargs["options"] = {}
        reset_kwargs["options"]["spawn_x"] = spawn_x
        return reset_kwargs

    # --- Initial Reset ---
    reset_kwargs = {}
    if randomize_inventory and is_crafter:
        sp = get_next_spawn_point()
        if sp: reset_kwargs['agent_pos'] = sp
    reset_kwargs = _maybe_add_bipedal_spawn(reset_kwargs)

    obs = env.reset(**reset_kwargs)[0]

    # --- BOOST 1: Randomize the very FIRST episode if needed ---
    if randomize_inventory and is_crafter:
        player = env.unwrapped.env._player

        for stat in ['health', 'food', 'drink', 'energy']:
            setattr(player, stat, float(np.random.randint(5, 10)))
        all_possible_items = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
        for item in all_possible_items:
            # Bias toward carrying advanced tools with 70% probability.
            player.inventory[item] = 1.0 if (('pickaxe' in item or 'sword' in item) and np.random.random() < 0.7) else float(np.random.randint(0, 5))
        # Refresh the observation immediately after inventory edits.
        obs = env.unwrapped._extract_obs()


    if save_img and wandb_run is not None:
        try:
            img = env.get_frame()
        except AttributeError:
             img = env.unwrapped.get_frame()
        wandb_run.log({log_name: wandb.Image(img)})
    # Visit count for RMax or exploration tracking
    visit_count = {}

    # Check both collect_cfg and cfg.env for save flags to support older configs
    should_save_vis = getattr(collect_cfg, "save_env_visualize", False) or getattr(cfg.env, "save_visualized_img", False)
    if should_save_vis:
        raw_img = None
        try:
            try:
                img = env.get_frame()
            except AttributeError:
                img = env.unwrapped.get_frame()
        except Exception:
            if hasattr(env, "render_global"):
                img = env.render_global()
            elif hasattr(env.unwrapped, "render_global"):
                img = env.unwrapped.render_global()
            else:
                try:
                    img = env.render(mode="rgb_array")
                except Exception:
                    img = env.render()
        
        raw_img = img
        img = _coerce_render_image(img)
        
        # --- save locally ---
        env_vis_path = getattr(collect_cfg, "env_visualize_save_path", getattr(cfg.env, "visualize_save_path", "outputs/env_visualization"))
        os.makedirs(env_vis_path, exist_ok=True)
        img_filename = getattr(collect_cfg, "env_visualize_filename", f"{log_name}_env.png")
        save_path = os.path.join(env_vis_path, img_filename)

        import matplotlib.pyplot as plt

        if img is None:
            print(f"[Saved Frame Skipped] Unsupported render output type: {type(raw_img)}")
        elif is_crafter and isinstance(obs, dict) and 'inventory' in obs:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.axis('off')
            
            inv = obs['inventory']
            inv_labels = ['Health', 'Food', 'Drink', 'Energy', 'Wood', 'Stone', 'Coal', 'Iron', 'Diamond', 'Sapling', 'Wood_Pickaxe', 'Stone_Pickaxe', 'Iron_Pickaxe', 'Wood_Sword', 'Stone_Sword', 'Iron_Sword']
            items = []
            for i, val in enumerate(inv):
                if val > 0 or i < 4:
                    items.append(f"{inv_labels[i]}:{int(val)}")
            
            # Join every 4 items with a newline to prevent overflowing
            lines = [" | ".join(items[i:i+4]) for i in range(0, len(items), 4)]
            title_str = "Initial Inventory:\n" + "\n".join(lines)
            
            ax.set_title(title_str, fontsize=10, fontfamily='monospace', color='black')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[Saved Frame] {save_path}")
        else:
            plt.imsave(save_path, img)
            print(f"[Saved Frame] {save_path}")

    # Define meaningful actions (forward, turn_left, turn_right)
    if is_crafter:
        meaningful_actions = list(range(env.action_space.n))
    elif is_bipedal:
        meaningful_actions = None
        # Build the default behavior policy only for BipedalWalker collection.
        if policy is None:
            bipedal_prior_policy = _build_bipedal_behavior_policy(cfg, env)
    else:
        meaningful_actions = [
            env.unwrapped.actions.forward,
            env.unwrapped.actions.left,
            env.unwrapped.actions.right,
            env.unwrapped.actions.pickup,
            env.unwrapped.actions.toggle,
            env.unwrapped.actions.drop,
        ]

    # Use tqdm for progress tracking
    target_episodes = collect_cfg.episodes
    mini_dataset_size = getattr(collect_cfg, "mini_dataset_size", 0)
    obs_norm_values = getattr(getattr(cfg, "attention_model", None), "obs_norm_values", None)
    if is_bipedal:
        stuck_window = int(getattr(domain_cfg, "stuck_window", 40))
        stuck_x_threshold = float(getattr(domain_cfg, "stuck_x_threshold", 0.25))
        stuck_action_threshold = float(getattr(domain_cfg, "stuck_action_threshold", 0.05))
        stuck_reward_threshold = float(getattr(domain_cfg, "stuck_reward_threshold", -0.2))
        stuck_min_steps = int(getattr(domain_cfg, "stuck_min_steps", 60))
        recent_hull_x = deque(maxlen=stuck_window)
        recent_rewards = deque(maxlen=stuck_window)
        recent_actions = deque(maxlen=stuck_window)

    def _build_policy_state(obs_img):
        obs_np = np.asarray(obs_img)
        if obs_np.ndim == 3:
            # HWC -> CHW
            obs_chw = np.transpose(obs_np, (2, 0, 1))
        else:
            obs_chw = obs_np
        if obs_norm_values is not None:
            obs_chw = normalize_obs(obs_chw, obs_norm_values)
        return torch.tensor(obs_chw, dtype=torch.float32, device=device)

    with tqdm(total=target_episodes, desc="Collecting Episodes") as pbar:
        info_list.append([{
            'carrying_key': False,
            'carrying_token': _minigrid_carrying_token(),
        }])

        # Condition: Keep collecting if (episodes < target) OR (total_steps < mini_dataset_size)
        # We check len(obs_list) for total steps.
        while episodes < target_episodes or (mini_dataset_size > 0 and len(obs_list) < mini_dataset_size):
            step_in_episode += 1

            # --- 1. CORE RANDOMIZATION (Every Step in Uniform Mode) ---
            if randomize_inventory and is_crafter:
                player = env.unwrapped.env._player
                
                # Sample survival stats uniformly from [1, 9].
                for stat in ['health', 'food', 'drink', 'energy']:
                    setattr(player, stat, float(np.random.randint(1, 10)))
                
                # Sample basic materials uniformly from [0, 9].
                materials = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
                for item in materials:
                    player.inventory[item] = float(np.random.randint(0, 10))
                
                # Sample tools and weapons from a uniform subset distribution.
                tools = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
                num_tools = np.random.randint(0, len(tools) + 1)
                selected_tools = np.random.choice(tools, num_tools, replace=False)
                for t in tools:
                    player.inventory[t] = 1.0 if t in selected_tools else 0.0
                
                # Refresh the observation after editing the inventory.
                obs = env.unwrapped._extract_obs()

            # Record observation
            obs_image = obs['image'] if isinstance(obs, dict) else obs
            obs_list.append([obs_image.copy()])
            if is_crafter and isinstance(obs, dict) and 'inventory' in obs:
                inv_list_cur.append(obs['inventory'].copy())

            # Action Selection
            if is_crafter:
                if randomize_inventory:
                    # --- DRILL-DOWN EXPERT: Boost 'do' action if facing a resource ---
                    player = env.unwrapped.env._player
                    world = env.unwrapped.env._world
                    tx, ty = int(player.pos[0] + player.facing[0]), int(player.pos[1] + player.facing[1])
                    target_material = None
                    if 0 <= tx < world.area[0] and 0 <= ty < world.area[1]:
                        mat_id = world._mat_map[tx, ty]
                        mat_names = [None, 'water', 'grass', 'stone', 'path', 'sand', 'tree', 'lava', 'coal', 'iron', 'diamond', 'table', 'furnace']
                        if 0 <= mat_id < len(mat_names):
                            target_material = mat_names[mat_id]
                    resources = ['tree', 'stone', 'coal', 'iron', 'diamond', 'table', 'furnace']
                    if target_material in resources and np.random.random() < 0.9:
                        act = 5 # 'do' action in Crafter
                    else:
                        act = np.random.choice(meaningful_actions)
                elif policy is not None:
                    # P2E / PPO policy injection for Crafter
                    if hasattr(policy, "expects_raw_obs") and policy.expects_raw_obs:
                        act = policy.select_action(obs_image)
                    else:
                        state_norm = _build_policy_state(obs_image)
                        act = policy.select_action(state_norm)
                else:
                    action_probs = np.ones(len(meaningful_actions), dtype=np.float32)
                    action_probs /= action_probs.sum()
                    act = np.random.choice(meaningful_actions, p=action_probs)
            elif is_bipedal:
                if policy is None:
                    # Use the configured bipedal behavior policy instead of pure random sampling.
                    act = bipedal_prior_policy.select_action(obs_image, add_noise=True)

                else:
                    act = policy.select_action(obs_image)
                    if torch.is_tensor(act):
                        act = act.detach().cpu().numpy()
                    act = np.asarray(act, dtype=np.float32).reshape(env.action_space.shape)
                    act = np.clip(act, env.action_space.low, env.action_space.high)
            else:
                action_probs = np.full(
                    len(meaningful_actions), 1.0 / len(meaningful_actions)
                )
                if policy is None:
                    act = np.random.choice(meaningful_actions, p=action_probs)
                else:
                    if hasattr(policy, "expects_raw_obs") and policy.expects_raw_obs:
                        act = policy.select_action(obs_image)
                    else:
                        state_norm = _build_policy_state(obs_image)
                        act = policy.select_action(state_norm)

            # --- 3. Step in Environment ---
            env_action = (
                compact_to_native(act)
                if is_minigrid and policy is not None
                else act
            )
            obs_next, reward, done, trunc, info = env.step(env_action)
            env_reward_for_metrics = float(reward)

            # --- BipedalWalker Odometry Update ---
            if is_bipedal:
                current_x = getattr(env.unwrapped, "hull", None).position[0] if hasattr(env.unwrapped, "hull") else 0.0
                max_x_in_episode = max(max_x_in_episode, current_x)
                max_x_global = max(max_x_global, current_x)
                stuck_terminated = False


            # P2E learns from intrinsic reward. Callers such as MAC may still
            # need the original environment reward in the saved dataset for
            # success/solvability checks.
            if intrinsic_reward_fn is not None:
                obs_next_image = obs_next['image'] if isinstance(obs_next, dict) else obs_next
                intrinsic_reward = intrinsic_reward_fn(obs_image, act, obs_next_image)
                if policy is not None and hasattr(policy, "record_transition"):
                    policy.record_transition(
                        float(intrinsic_reward),
                        bool(done or trunc),
                        env_reward=env_reward_for_metrics,
                        obs=obs_image,
                        obs_next=obs_next_image,
                        action=act,
                        is_success=bool(
                            (info.get("success", False) if isinstance(info, dict) else False)
                            or ((done or trunc) and env_reward_for_metrics > 0.0)
                        ),
                    )
                if store_intrinsic_reward:
                    reward = intrinsic_reward

            if is_bipedal:
                recent_hull_x.append(float(current_x))
                recent_rewards.append(float(reward))
                recent_actions.append(np.asarray(act, dtype=np.float32).reshape(-1).copy())
                if len(recent_hull_x) == stuck_window and step_in_episode >= stuck_min_steps:
                    x_progress = float(max(recent_hull_x) - min(recent_hull_x))
                    mean_reward = float(np.mean(recent_rewards))
                    action_std = float(np.mean(np.std(np.stack(recent_actions, axis=0), axis=0)))
                    low_progress = x_progress < stuck_x_threshold
                    low_action_change = action_std < stuck_action_threshold
                    low_reward = mean_reward < stuck_reward_threshold
                    if low_progress and (low_action_change or low_reward):
                        done = True
                        stuck_terminated = True

            if is_minigrid:
                next_carrying_token = _minigrid_carrying_token()
                info['carrying_key'] = next_carrying_token > 0
                info['carrying_token'] = next_carrying_token
                # info_list is aligned with the current observation. Attach the
                # post-action inventory target to that same transition.
                info_list[-1][0]['next_carrying_key'] = next_carrying_token > 0
                info_list[-1][0]['next_carrying_token'] = next_carrying_token
            else:
                info['carrying_key'] = False
            if policy is not None and hasattr(policy, "set_carrying_key"):
                # This context is consumed only by the next P2E action. Clear it
                # at terminal transitions because the collector resets the env.
                policy.set_carrying_key(
                    bool(info['carrying_key']) if not (done or trunc) else False
                )
            if is_bipedal:
                info['hull_x'] = float(current_x)
                info['episode_done'] = bool(done or trunc)
                info['stuck_terminated'] = bool(stuck_terminated) if 'stuck_terminated' in locals() else False
            # check the data
            # if not has_carried_key_this_episode and info['carrying_key']:
            #     tqdm.write(f"[Episode {episodes}] First time carrying key at step {step_in_episode}! Action: {act}")

            #     obs_diff = obs_next['image'].astype(int) - obs['image'].astype(int)
            #     tqdm.write(f"obs_next - obs (nonzero count): {obs_diff}")
            #     # Optional: print the raw difference matrix directly for tiny images.
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
            obs_next_image = obs_next['image'] if isinstance(obs_next, dict) else obs_next
            obs_next_list.append([obs_next_image.copy()])
            # Crafter only: collect next inventory
            if is_crafter and isinstance(obs_next, dict) and 'inventory' in obs_next:
                inv_list_next.append(obs_next['inventory'].copy())   
            rew_list.append([reward])
            done_list.append([done])
            info_list.append([info])

            # For Crafter uniform interaction, we use a balanced interval (e.g. 50 steps)
            # enough to interact but frequent enough for variety.
            uniform_reset_steps = getattr(collect_cfg, "uniform_reset_steps", 50 if is_crafter else 300)
            if randomize_inventory and step_in_episode >= uniform_reset_steps:
                done = True

            # --- early stop if dataset size reached ---
            if maximum_dataset_size is not None and len(obs_list) >= maximum_dataset_size:
                print(f"[STOP] Reached maximum dataset size: {maximum_dataset_size}")
                break



            # Update visit count and RMax exploration
            if is_bipedal:
                act_key = tuple(np.asarray(act).reshape(-1).tolist())
            else:
                act_key = int(act)
            state_action_key = (tuple(np.asarray(obs_image).flatten()), act_key)
            visit_count[state_action_key] = visit_count.get(state_action_key, 0) + 1
            if rmax_exploration is not None:
                rmax_exploration.update_visit_count(obs_image, act)

            # Visualize if needed
            if getattr(cfg.env, "visualize", False):
                env.render()
                # time.sleep(0.1)

            # Reset environment on episode end
            if done or trunc:
                if not is_bipedal:
                    info_list.pop()
                episodes += 1
                pbar.update(1)
                
                # --- BipedalWalker Odometry Print ---
                if is_bipedal:
                    if info.get("stuck_terminated", False):
                        pbar.write(
                            f"[Episode {episodes}] Early stop: stuck detected at step {step_in_episode}. "
                            f"Hull x={info['hull_x']:.2f}"
                        )
                    pbar.write(
                        f"[Episode {episodes}] Finished. "
                        f"Episode Max Distance: {max_x_in_episode:.2f}m | "
                        f"Global Max Distance: {max_x_global:.2f}m"
                    )
                    max_x_in_episode = 0.0  # Reset
                    recent_hull_x.clear()
                    recent_rewards.clear()
                    recent_actions.clear()
                
                # --- BOOST 2: Exhaustive Spawning (Cycling through all tiles) ---
                reset_kwargs = {}
                if randomize_inventory and is_crafter:
                    sp = get_next_spawn_point()
                    if sp: reset_kwargs['agent_pos'] = sp

                reset_kwargs = _maybe_add_bipedal_spawn(reset_kwargs)

                obs = env.reset(**reset_kwargs)[0]
                
                # Reset heuristic internal state on done
                if is_bipedal and policy is None:
                    bipedal_prior_policy.reset()
                
                if randomize_inventory and is_crafter:
                    player = env.unwrapped.env._player
                    for stat in ['health', 'food', 'drink', 'energy']:
                        setattr(player, stat, float(np.random.randint(1, 10)))
                    materials = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
                    for item in materials:
                        player.inventory[item] = float(np.random.randint(0, 10))
                    tools = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
                    num_tools = np.random.randint(0, len(tools) + 1)
                    sel = np.random.choice(tools, num_tools, replace=False)
                    for t in tools: player.inventory[t] = 1.0 if t in sel else 0.0
                    
                    # Sync observation after injection
                    obs = env.unwrapped._extract_obs()

                info_list.append([{
                    'carrying_key': False,
                    'carrying_token': _minigrid_carrying_token(),
                }])
                has_carried_key_this_episode = False  # Reset per-episode state.
                step_in_episode = 0
            else:
                obs = obs_next

    # A transition budget can stop collection in the middle of an episode.
    # Mark that boundary so PPO returns do not propagate into the next,
    # differently configured mini-environment.
    if policy is not None and hasattr(policy, "mark_rollout_boundary"):
        policy.mark_rollout_boundary()

    info_list.pop()
    if is_bipedal:
        print(f"[RunEnv] Final global max distance across collected data: {max_x_global:.2f}m")
    # Convert collected data to numpy arrays
    obs_np = np.concatenate(obs_list)
    obs_next_np = np.concatenate(obs_next_list)
    act_np = np.concatenate(act_list)
    # Only for MiniGrid convention.
    if is_minigrid:
        act_np = native_to_compact(act_np)
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)
    info_np = np.concatenate(info_list)
    # Crafter only: build inventory arrays
    if is_crafter and len(inv_list_cur) > 0:
        inv_np = np.stack(inv_list_cur, axis=0).astype(np.float32)       # (N, 16)
        inv_next_np = np.stack(inv_list_next, axis=0).astype(np.float32) # (N, 16)
        # Align length with obs (episode boundary pop)
        min_len = min(len(obs_np), len(inv_np))
        inv_np = inv_np[:min_len]
        inv_next_np = inv_next_np[:min_len]
        obs_np = obs_np[:min_len]
        obs_next_np = obs_next_np[:min_len]
        act_np = act_np[:min_len]
        rew_np = rew_np[:min_len]
        done_np = done_np[:min_len]
        info_np = info_np[:min_len]
    else:
        inv_np, inv_next_np = None, None

    # Log statistics
    print(f"Observation shape: {obs_np.shape}")
    print(f"Next observation shape: {obs_next_np.shape}")
    print(f"Actions shape: {act_np.shape}")
    print(f"Rewards shape: {rew_np.shape}")
    print(f"Dones shape: {done_np.shape}")
    print(f"Number of episodes started: {episodes}")
    print(f"Unique state-action pairs visited: {len(visit_count)}")
    if is_minigrid and len(obs_np) > 0:
        action_vals, action_counts = np.unique(act_np.astype(np.int64), return_counts=True)
        action_hist = {int(a): int(c) for a, c in zip(action_vals.tolist(), action_counts.tolist())}
        agent_positions = get_agent_position(np.transpose(obs_np, (0, 3, 1, 2)))
        unique_agent_positions = int(len(np.unique(agent_positions, axis=0))) if len(agent_positions) > 0 else 0
        print(f"MiniGrid action histogram: {action_hist}")
        print(f"MiniGrid unique agent positions: {unique_agent_positions}")
    env.close()

    return obs_np, obs_next_np, act_np, rew_np, done_np, info_np, inv_np, inv_next_np


def teleport_near_important_tiles(env, obs, step_in_episode, interval):
    """
    Every `interval` steps, teleport agent to an EMPTY tile near key/door/lava.
    """
    if step_in_episode % interval != 0:
        return

    import random

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
        env.unwrapped.actions.toggle,
        env.unwrapped.actions.drop,
    ]

    # Optional W&B image logging
    if save_img and wandb_run is not None:
        try:
            try:
                img = env.get_frame()
            except AttributeError:
                img = env.unwrapped.get_frame()
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

                # Rely on env.reset() to place agent at (x,y,dir)
                # For MiniGrid, env.reset() will place the agent randomly if not specified.
                # To achieve uniform coverage, we need to ensure the environment is reset
                # to specific (x,y,dir) states. This might require modifying the env.reset()
                # or using a custom reset function that allows setting agent_pos and agent_dir.
                # Assuming env.reset() can be made to place the agent at (x,y,dir) for uniform collection.
                # If not, the original agent_pos/dir manipulation is necessary.
                # For now, we remove the direct manipulation as per instruction and assume reset handles it.
                try:
                    # Reset environment to a specific (x,y,dir) state for uniform coverage
                    # This part might need environment-specific implementation if env.reset()
                    # does not support setting initial agent state directly.
                    # For MiniGrid, agent_pos and agent_dir are usually set randomly or by map.
                    # The original code directly manipulated agent_pos/dir, which is now removed.
                    # To achieve uniform coverage, we would need a mechanism to ensure env.reset()
                    # places the agent at (x,y,dir). For now, we just call reset.
                    obs_dict = env.reset(agent_pos=(x,y), agent_dir=dir)[0] # Assuming reset can take args
                    obs = obs_dict["image"]
                except TypeError: # If env.reset() doesn't take agent_pos/dir
                    # Fallback: reset normally and then try to set, or just reset and hope for coverage
                    obs_dict = env.reset()[0]
                    env.unwrapped.agent_pos = (x, y)
                    env.unwrapped.agent_dir = dir
                    obs = env.unwrapped._extract_obs()  # Refresh the observation immediately.
                except Exception:
                    continue

                for a in meaningful_actions:
                    if step_count >= target_samples:
                        break

                    # No backup/restore of agent_pos/dir as per instruction
                    # The state is assumed to be fixed by the outer loops' reset or manipulation.

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

    # Compact the live MiniGrid toggle id (5) into dataset/PPO toggle id (4).
    # Raw env id 4 is drop and is never included in meaningful_actions.
    if np.any(act_np == 5):
        print("Mapping live action 5 (toggle) → compact action 4 (toggle).")
    act_np = native_to_compact(act_np)

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
    positions = get_agent_position(obs_trans)
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
    obs, obs_next, act, rew, done, info, _, _ = run_env(
        env, cfg, wandb_run, log_name, save_img=save_img
    )
    obs_u, obsn_u, act_u, rew_u, done_u, info_u = uniformize_dataset_by_position(
                                                obs, obs_next, act, rew, done, info
                                                )
    return obs_u, obsn_u, act_u, rew_u, done_u, info_u


def save_experiments(cfg: DictConfig, obs, obs_next, act, rew, done, info=None, inv=None, inv_next=None):
    # Grid observations (MiniGrid/Crafter) are 4-D; vector observations (e.g., Bipedal) are 2-D.
    if obs.ndim == 4:
        # Crafter symbolic grid has C=2 in the last axis before conversion.
        if obs.shape[-1] == 2:
            # Crafter: (N, H, W, C) -> (N, C, H, W)
            obs = np.moveaxis(obs, -1, 1)
            obs_next = np.moveaxis(obs_next, -1, 1)
        else:
            # MiniGrid: (N, Col, Row, Canl) -> (N, Canl, Row, Col)
            obs = ColRowCanl_to_CanlRowCol(obs)
            obs_next = ColRowCanl_to_CanlRowCol(obs_next)
    elif obs.ndim == 2:
        obs = obs.astype(np.float32, copy=False)
        obs_next = obs_next.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported observation shape for saving: {obs.shape}")
    save_path = cfg.env.collect.data_save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_kwargs = dict(
        a=obs,
        b=obs_next,
        c=act,
        d=rew,
        e=done,
        f=info,
        metadata=metadata_array(cfg),
    )
    if inv is not None:
        save_kwargs['g'] = inv        # inventory at time t
        save_kwargs['h'] = inv_next   # inventory at time t+1
    np.savez_compressed(save_path, **save_kwargs)

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
    KEYDOOR_ACTIONS = [
        env.unwrapped.actions.pickup,
        env.unwrapped.actions.toggle,
        env.unwrapped.actions.drop,
    ]

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
    Keep interactions related to key/door mechanics and discard most random movement.
    - keydoor: pickup / toggle / carrying_key=True
    - move: all other actions, retained only at a configurable ratio
    """
    num = obs.shape[0]
    flat_act = act.reshape(num)

    # Key interaction actions (key / door related).
    KEYDOOR_ACTIONS = [
        env.unwrapped.actions.pickup,
        env.unwrapped.actions.toggle,
        env.unwrapped.actions.drop,
    ]

    is_keydoor = np.zeros(num, dtype=bool)
    for a in KEYDOOR_ACTIONS:
        is_keydoor |= (flat_act == a)

    # Also retain steps where the agent is carrying a key.
    is_keydoor |= np.array([i.get("carrying_key", False) for i in info])

    # Remaining actions are treated as movement.
    move_idx = np.where(~is_keydoor)[0]
    keep_move = np.random.rand(len(move_idx)) < move_keep_ratio
    move_keep_idx = move_idx[keep_move]

    # Indices for retained key/door interactions.
    keydoor_idx = np.where(is_keydoor)[0]

    # Merge the final retained indices.
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
    move_keep_ratio,     # Randomly retain this fraction of non-key/door samples.
    require_changed=True,     # Treat a sample as key/door only if it changes state.
    min_keep_moves=1,         # Keep at least this many movement samples.
    shuffle=True
):
    """
    downsampling the data which is just moving around,
    to improving the training of the interact with key-door info
    """
    # Normalize `info` to a Python list.
    info_list = info if isinstance(info, list) else list(info)

    is_keydoor = _keydoor_mask(env, obs, obs_next, act, info_list, require_changed=require_changed)
    kd_idx  = np.where(is_keydoor)[0]
    mov_idx = np.where(~is_keydoor)[0]

    # If there are no key/door samples, downsample movement directly.
    # Otherwise, only downsample the movement subset.
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
        # Fallback: keep at least one sample.
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
    env_type = str(getattr(cfg.env, "env_type", "")).lower()
    mode = 'human' if getattr(cfg.env, "visualize", False) else 'rgb_array'

    if env_type == "crafter":
        # Workaround for numba cache issues seen in some environments when importing crafter.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        env_path = getattr(cfg.env, "env_path", None)
        crafter_cfg = getattr(cfg.env, "crafter", {})
        stochastic = getattr(crafter_cfg, "stochastic", False) if hasattr(crafter_cfg, "stochastic") else crafter_cfg.get("stochastic", False) if isinstance(crafter_cfg, dict) else False
        slippery_prob = getattr(crafter_cfg, "slippery_prob", 0.0) if hasattr(crafter_cfg, "slippery_prob") else crafter_cfg.get("slippery_prob", 0.0) if isinstance(crafter_cfg, dict) else 0.0
        max_steps = getattr(cfg.env, "max_steps", 10000)
        from domain.crafter.crafter_custom_env import CustomCrafterEnv
        if env_path and os.path.exists(env_path):
            env = CustomCrafterEnv(txt_file_path=env_path, max_steps=max_steps,
                                   ai_enabled=stochastic, slippery_prob=slippery_prob)
        else:
            env = CustomCrafterEnv(max_steps=max_steps,
                                   ai_enabled=stochastic, slippery_prob=slippery_prob)
    elif env_type == "bipedalwalker":
        if mode != 'human':
            # Force PyGame into headless mode; some systems still flash a window otherwise.
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        env_path = getattr(cfg.env, "env_path", None)
        max_steps = getattr(cfg.env, "max_steps", 10000)
        from domain.bipedalwalker.custom_bipedal_env import CustomBipedalEnv
        if env_path and os.path.exists(env_path):
            env = wrap_bipedal_env_from_text(env_path, render_mode=mode)
        else:
            env = CustomBipedalEnv(render_mode=mode)
        env = TimeLimit(env, max_episode_steps=max_steps)
    else:
        env_path = getattr(cfg.env, "env_path", None)
        max_steps = getattr(cfg.env, "max_steps", 10000)
        replace_start_with_empty = bool(
            getattr(cfg.env.collect, "replace_start_with_empty", False)
        )
        env = FullyObsWrapper(
            CustomMiniGridEnv(
                txt_file_path=env_path,
                replace_start_with_empty=replace_start_with_empty,
                custom_mission="Find the key and open the door.",
                max_steps=max_steps,
                render_mode=mode,
            )
        )
        if replace_start_with_empty:
            print(
                "[Data Collection] Treating layout S as E; "
                "MiniGrid starts from a random empty position."
            )

    collect_cfg = getattr(cfg.env, "collect", cfg.env)
    data_type = str(getattr(collect_cfg, "data_type", "random")).lower()
    randomize = (data_type == "uniform" and env_type == "crafter")
    log_name = getattr(collect_cfg, "visualize_filename", "train.png").split('.')[0]

    obs, obs_next, act, rew, done, info, inv, inv_next = run_env(
        env, cfg, log_name=log_name, wandb_run=None, save_img=False, randomize_inventory=randomize
    )
    save_experiments(cfg, obs, obs_next, act, rew, done, info, inv=inv, inv_next=inv_next)

    # coverage visualization
    data_path = cfg.env.collect.data_save_path
    if getattr(cfg.env.collect, "save_coverage_visualize", False) and os.path.exists(data_path):
        filename, vis_save_path = _resolve_coverage_vis_target(cfg)
        os.makedirs(vis_save_path, exist_ok=True)
        save_path = os.path.join(vis_save_path, filename)
        data = np.load(data_path, allow_pickle=True)
        collect_cfg = cfg.env.collect if hasattr(cfg.env, "collect") else cfg.env
        dataset_type = getattr(collect_cfg, "data_type", "Random")
        if data["a"].ndim == 2 and cfg.env.env_type == "bipedalwalker":
            layout_source = getattr(cfg.env.collect, "current_layout_str", None) or getattr(cfg.env, "env_path", None)
            visualize_bipedal_coverage_1d(
                data,
                layout_source=layout_source,
                save_path=save_path,
                title=f"Bipedal 1D Coverage ({dataset_type})",
            )
        else:
            visualize_agent_coverage(
                data,
                save_path=save_path,
                title=f"Agent Position Coverage ({dataset_type})"
            )

    env.close()


def data_collect_api_multiprocess(cfg: DictConfig, env, wandb_run, save_img=False):
    hparam = cfg.env
    obs, obs_next, act, rew, done, info = run_env_multiprocess(env, hparam, wandb_run, save_img=save_img)
    save_experiments(cfg, obs, obs_next, act, rew, done, info)

def data_collect_api(
    cfg: DictConfig,
    env,
    wandb_run,
    save_img,
    log_name,
    max_steps,
    policy=None,
    intrinsic_reward_fn=None,
    store_intrinsic_reward=True,
):
    runtime_bipedal_layout_source = _resolve_runtime_bipedal_layout_source(cfg, env)
    hparam = copy.deepcopy(cfg)
    original_episodes = hparam.env.collect.episodes

    # === DATA BUFFERS ===
    obs_all, obsn_all, act_all, rew_all, done_all, info_all = [], [], [], [], [], []
    inv_all, invn_all = [], []

    # ------------------------------------------------------
    # CASE 1: Single-run mode (NO ITERATION)
    # ------------------------------------------------------
    if max_steps is None:
        print(f"[Single-run mode] Collecting {hparam.env.collect.episodes} episodes...")

        collect_cfg = cfg.env.collect if hasattr(cfg.env, "collect") else cfg.env
        if collect_cfg.data_type.lower() == 'uniform':
            if str(hparam.env.env_type).lower() == 'crafter':
                obs, obs_next, act, rew, done, info, inv, inv_next = run_env(
                    env, hparam, wandb_run, log_name, save_img=save_img,
                    randomize_inventory=True, policy=policy,
                    intrinsic_reward_fn=intrinsic_reward_fn,
                    store_intrinsic_reward=store_intrinsic_reward,
                )
            else:
                obs, obs_next, act, rew, done, info = uniform_collect_data_postprocess(
                    env, hparam, wandb_run, log_name,
                    policy=None, rmax_exploration=None, save_img=save_img
                )
                inv, inv_next = None, None
        else:
            obs, obs_next, act, rew, done, info, inv, inv_next = run_env(
                env, hparam, wandb_run, log_name, save_img=save_img,
                policy=policy, intrinsic_reward_fn=intrinsic_reward_fn,
                store_intrinsic_reward=store_intrinsic_reward,
            )

        # store results
        obs_all.append(obs)
        obsn_all.append(obs_next)
        act_all.append(act)
        rew_all.append(rew)
        done_all.append(done)
        info_all.append(info)
        if inv is not None:
            inv_all.append(inv)
            invn_all.append(inv_next)

        print("[Single-run] Finished collecting. Saving dataset...")
            # jump to the saving section
        return _finalize_and_save(
            cfg,
            env,
            obs_all,
            obsn_all,
            act_all,
            rew_all,
            done_all,
            info_all,
            inv_all,
            invn_all,
            layout_source=runtime_bipedal_layout_source,
        )
    

    # ------------------------------------------------------
    # CASE 2: Multi-round mode (original behavior)
    # ------------------------------------------------------
    total_steps = 0
    round_idx = 0

    while total_steps < max_steps:
        collect_cfg = hparam.env.collect if hasattr(hparam.env, "collect") else hparam.env
        if collect_cfg.data_type.lower() == 'uniform':
            if str(hparam.env.env_type).lower() == 'crafter':
                obs, obs_next, act, rew, done, info, inv, inv_next = run_env(
                    env, hparam, wandb_run, log_name, save_img=save_img,
                    randomize_inventory=True, policy=policy,
                    intrinsic_reward_fn=intrinsic_reward_fn,
                    store_intrinsic_reward=store_intrinsic_reward,
                )
            else:
                obs, obs_next, act, rew, done, info = uniform_collect_data_postprocess(
                    env, hparam, wandb_run, log_name,
                    policy=None, rmax_exploration=None, save_img=False
                )
                inv, inv_next = None, None
        else:
            obs, obs_next, act, rew, done, info, inv, inv_next = run_env(
                env, hparam, wandb_run, log_name, save_img=save_img,
                policy=policy, intrinsic_reward_fn=intrinsic_reward_fn,
                store_intrinsic_reward=store_intrinsic_reward,
            )

        obs_all.append(obs)
        obsn_all.append(obs_next)
        act_all.append(act)
        rew_all.append(rew)
        done_all.append(done)
        info_all.append(info)
        if inv is not None:
            inv_all.append(inv)
            invn_all.append(inv_next)

        total_steps += len(obs)
        print(f"Total steps collected: {total_steps}")

        if total_steps < max_steps:
            hparam.env.collect.episodes = max(1, original_episodes // 3)

        round_idx += 1
        save_img = False

    return _finalize_and_save(
        cfg,
        env,
        obs_all,
        obsn_all,
        act_all,
        rew_all,
        done_all,
        info_all,
        inv_all,
        invn_all,
        layout_source=runtime_bipedal_layout_source,
    )



# ----------------------------------------------------------
# Helper: final save + visualization
# ----------------------------------------------------------
def _finalize_and_save(
    cfg,
    env,
    obs_all,
    obsn_all,
    act_all,
    rew_all,
    done_all,
    info_all,
    inv_all=None,
    invn_all=None,
    layout_source=None,
):

    # merge
    obs_all = np.concatenate(obs_all, axis=0)
    obsn_all = np.concatenate(obsn_all, axis=0)
    act_all = np.concatenate(act_all, axis=0)
    rew_all = np.concatenate(rew_all, axis=0)
    done_all = np.concatenate(done_all, axis=0)
    if isinstance(info_all[0], (list, np.ndarray)):
        info_all = np.concatenate(info_all, axis=0)
    else:
        # Fallback if info is a list of objects but not naturally concatenatable
        info_all = np.array(info_all, dtype=object)

    inv_final, invn_final = None, None
    if inv_all is not None and len(inv_all) > 0:
        inv_final = np.concatenate(inv_all, axis=0)
        invn_final = np.concatenate(invn_all, axis=0)

    print(f"Final data shape: {obs_all.shape}")

    save_experiments(cfg, obs_all, obsn_all, act_all, rew_all, done_all, info_all, inv=inv_final, inv_next=invn_final)

    # visualization
    data_path = cfg.env.collect.data_save_path
    if cfg.env.collect.save_coverage_visualize and os.path.exists(data_path):
        filename, vis_save_path = _resolve_coverage_vis_target(cfg)
        os.makedirs(vis_save_path, exist_ok=True)
        save_path = os.path.join(vis_save_path, filename)
        data = np.load(data_path, allow_pickle=True)
        collect_cfg = cfg.env.collect if hasattr(cfg.env, "collect") else cfg.env
        dataset_type = getattr(collect_cfg, "data_type", "Random")
        if data["a"].ndim == 2 and cfg.env.env_type == "bipedalwalker":
            layout_source = layout_source or _resolve_runtime_bipedal_layout_source(cfg, env)
            visualize_bipedal_coverage_1d(
                data,
                layout_source=layout_source,
                save_path=save_path,
                title=f"Bipedal 1D Coverage ({dataset_type})",
            )
        else:
            visualize_agent_coverage(
                data,
                save_path=save_path,
                title=f"Agent Position Coverage ({dataset_type})"
            )

    env.close()




def visualize_agent_coverage(data, save_path=None, title="Agent Position Coverage"):
    """
    Visualize agent-position coverage as a heatmap over the dataset.
    - Automatically extracts agent coordinates with `get_agent_position()`
    """

    # -------------------------------
    # Step 1: extract positions with the existing helper.
    # -------------------------------
    obs_np = data['a']
    raw_obs = obs_np
    # If the array is NHWC, convert it to NCHW; Crafter symbolic observations use C=2.
    if obs_np.shape[1] not in [2, 3, 4, 5]:  # Unusual channel count likely means channels-last.
        obs_np = np.moveaxis(obs_np, -1, 1)
        
    is_crafter_symbolic = (obs_np.shape[1] == 2)

    # Crafter symbolic observations use their own object-id mapping and must
    # not fall back to the MiniGrid agent-position helper.
    if obs_np.shape[1] == 2:
        from domain.crafter.crafter_support import extract_player_positions
        positions = extract_player_positions(obs_np)  # (N, 2)
    else:
        positions = get_agent_position(obs_np)  # (N, 2)

    # Infer the map size automatically after standardizing to NCHW.
    if isinstance(obs_np, torch.Tensor):
        obs_np = obs_np.detach().cpu().numpy()
    
    if obs_np.ndim == 4 and obs_np.shape[1] > 8: # If still NHWC for some reason
        obs_np = ColRowCanl_to_CanlRowCol(obs_np)
        
    Height, Width = obs_np.shape[2], obs_np.shape[3]
    if not is_crafter_symbolic:
        positions = get_agent_position(obs_np)  # Returns (y, x) for NCHW

    # -------------------------------
    # Step 2: count visits at each position.
    # -------------------------------
    heatmap = np.zeros((Height, Width)) 
    
    for (y, x) in positions:
        # In NCHW processed by ColRowCanl_to_CanlRowCol, p1 is y, p2 is x
        if 0 <= y < Height and 0 <= x < Width:
            heatmap[y, x] += 1

    # Heatmap is now naturally (Row, Col) which matches imshow expectation.

    # Heatmap is now naturally (Row, Col) which matches imshow expectation.

    # -------------------------------
    # Step 3: Handle Inventory Stats if available
    # -------------------------------
    import matplotlib
    if save_path:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'g' in data and len(data['g']) > 0:
        inv_data = data['g']
        if len(inv_data.shape) == 3: # (N, _, 16)
            inv_data = inv_data[:, 0, :]
        max_inv = np.max(inv_data, axis=0)
        
        inv_labels = ['Health', 'Food', 'Drink', 'Energy', 
                     'Wood', 'Stone', 'Coal', 'Iron', 'Diamond', 'Sapling', 
                     'Wood_Pickaxe', 'Stone_Pickaxe', 'Iron_Pickaxe', 
                     'Wood_Sword', 'Stone_Sword', 'Iron_Sword']
                     
        inv_stats = np.mean(inv_data > 0, axis=0) * 100
        
        # Create a figure with two subplots: Heatmap and Bar chart
        fig = plt.figure(figsize=(14, 6))
        
        # 1. Heatmap
        ax1 = plt.subplot(1, 2, 1)
        im = ax1.imshow(heatmap, cmap="viridis", origin="upper")
        # ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")
        plt.colorbar(im, ax=ax1, label="Occurrences", fraction=0.046, pad=0.04)
        
        # 2. Inventory Stats
        ax2 = plt.subplot(1, 2, 2)
        colors = ['#3498db']*4 + ['#2ecc71']*6 + ['#e74c3c']*3 + ['#f39c12']*3
        bars = ax2.bar(range(len(inv_labels)), inv_stats, color=colors, alpha=0.8, edgecolor='black')
        
        ax2.set_title("Agent Inventory Acquisition Rate", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Presence Rate (%)")
        ax2.set_ylim(0, 115) # Leave space for text
        ax2.set_xticks(range(len(inv_labels)))
        ax2.set_xticklabels(inv_labels, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add values on top of bars with better formatting
        for i, v in enumerate(inv_stats):
            color = 'black' if v < 95 else 'darkred'
            ax2.text(i, v + 2, f"{v:.1f}", ha='center', fontsize=8, fontweight='bold', color=color)
            
        # Add a small legend for groups
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#3498db', lw=4, label='Physiological Stats'),
            Line2D([0], [0], color='#2ecc71', lw=4, label='Basic Materials'),
            Line2D([0], [0], color='#e74c3c', lw=4, label='Pickaxes'),
            Line2D([0], [0], color='#f39c12', lw=4, label='Swords')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.5)
        
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(heatmap, cmap="viridis", origin="upper")
        # ax.set_title(title)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        plt.colorbar(im, ax=ax, label="Occurrences", fraction=0.046, pad=0.04)
        plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap and Stats saved to {save_path}")
    else:
        plt.show()
    plt.close()


def _extract_bipedal_x_positions_and_episode_max(data):
    info_np = data.get("f", None)
    if info_np is None:
        raise ValueError("Saved dataset does not contain info needed for Bipedal coverage visualization.")

    x_positions = []
    episode_maxima = []
    current_max = None

    for item in np.asarray(info_np).reshape(-1):
        info = item.item() if isinstance(item, np.ndarray) and item.shape == () else item
        if isinstance(info, (list, tuple)) and len(info) == 1:
            info = info[0]
        if not isinstance(info, dict):
            continue
        if "hull_x" not in info:
            continue
        x_val = float(info["hull_x"])
        x_positions.append(x_val)
        current_max = x_val if current_max is None else max(current_max, x_val)
        if bool(info.get("episode_done", False)):
            episode_maxima.append(current_max if current_max is not None else x_val)
            current_max = None

    if current_max is not None:
        episode_maxima.append(current_max)

    if not x_positions:
        raise ValueError(
            "No 'hull_x' values found in dataset info. Re-collect Bipedal data after enabling odometry logging."
        )

    return np.asarray(x_positions, dtype=np.float32), np.asarray(episode_maxima, dtype=np.float32)


def _resolve_bipedal_layout_string(layout_source):
    if layout_source is None:
        raise ValueError("Bipedal coverage visualization requires a layout path or a raw layout string.")

    if isinstance(layout_source, os.PathLike):
        return read_layout_from_txt(layout_source)

    if isinstance(layout_source, str):
        if os.path.exists(layout_source):
            return read_layout_from_txt(layout_source)
        return layout_source.strip()

    raise TypeError(f"Unsupported bipedal layout source type: {type(layout_source)!r}")


def _resolve_runtime_bipedal_layout_source(cfg, env=None):
    collect_cfg = cfg.env.collect if hasattr(cfg.env, "collect") else cfg.env
    layout_source = getattr(collect_cfg, "current_layout_str", None)
    if layout_source:
        return layout_source
    if env is not None and hasattr(env, "layout_str") and getattr(env, "layout_str", None):
        return env.layout_str
    return getattr(cfg.env, "env_path", None)


def _compute_bipedal_layout_regions(layout_source):
    layout_str = _resolve_bipedal_layout_string(layout_source)
    import re

    tokens = re.findall(r'([RGSPT])(-?\d+\.?\d*)', layout_str.replace(" ", ""))
    x_cursor = 0.0
    regions = []
    active_roughness = None
    active_roughness_start = None

    def _close_roughness_region(x_end):
        nonlocal active_roughness, active_roughness_start
        if active_roughness is None or active_roughness_start is None or x_end <= active_roughness_start:
            return
        regions.append(
            {
                "type": "roughness",
                "x_start": active_roughness_start,
                "x_end": x_end,
                "value": float(active_roughness),
            }
        )
        active_roughness = None
        active_roughness_start = None

    for type_char, value_str in tokens:
        value = float(value_str)
        if type_char == "R":
            _close_roughness_region(x_cursor)
            active_roughness = value
            active_roughness_start = x_cursor
            continue
        if type_char == "G":
            width = int(value) * TERRAIN_STEP
            if width > 0:
                regions.append({"type": "grass", "x_start": x_cursor, "x_end": x_cursor + width})
            x_cursor += width
            continue
        if type_char == "S":
            width = max(TERRAIN_STEP, float(value) * TERRAIN_STEP)
            regions.append({"type": "stump", "x_start": x_cursor, "x_end": x_cursor + width})
            x_cursor += width
            continue
        if type_char == "P":
            width = (int(value) + 2) * TERRAIN_STEP
            regions.append({"type": "pit", "x_start": x_cursor, "x_end": x_cursor + width})
            x_cursor += width
            continue
        if type_char == "T":
            stair_width = 4
            stair_steps = abs(int(value)) if int(value) != 0 else 3
            width = stair_steps * stair_width * TERRAIN_STEP
            regions.append({"type": "stairs", "x_start": x_cursor, "x_end": x_cursor + width})
            x_cursor += width

    _close_roughness_region(x_cursor)

    spawn_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2.0
    target_x = TERRAIN_STEP * (TERRAIN_LENGTH - TERRAIN_GRASS)
    return {
        "layout_str": layout_str,
        "regions": regions,
        "spawn_x": spawn_x,
        "target_x": target_x,
    }


def visualize_bipedal_coverage_1d(data, layout_source, save_path=None, title="Bipedal Coverage"):
    x_positions, episode_maxima = _extract_bipedal_x_positions_and_episode_max(data)
    layout_meta = _compute_bipedal_layout_regions(layout_source)

    import matplotlib
    if save_path:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    max_data_x = float(np.max(x_positions)) if len(x_positions) > 0 else 0.0
    plot_max_x = max(max_data_x, float(layout_meta["target_x"])) + TERRAIN_STEP
    bins = max(20, int(np.ceil(plot_max_x / TERRAIN_STEP)))

    fig, (ax_hist, ax_layout) = plt.subplots(
        2, 1, figsize=(14, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.9]}
    )
    ax_hist.set_facecolor("white")
    ax_layout.set_facecolor("white")

    counts, edges = np.histogram(x_positions, bins=bins, range=(0.0, plot_max_x))
    counts = counts.astype(np.float32)
    heat = counts / counts.max() if counts.max() > 0 else counts
    if counts.max() > 0:
        heat = np.where(counts > 0, 0.25 + 0.75 * heat, 0.0)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "coverage_white_blue",
        ["#ffffff", "#d6e9ff", "#7db7ff", "#1f6fd5", "#0b2f6b"],
    )
    band_y = 0.0
    band_h = 1.0
    for idx in range(len(counts)):
        if counts[idx] <= 0:
            continue
        left = edges[idx]
        width = edges[idx + 1] - edges[idx]
        ax_hist.bar(
            left,
            band_h,
            bottom=band_y,
            width=width,
            align="edge",
            color=cmap(float(heat[idx])),
            edgecolor="none",
        )

    region_colors = {
        "grass": "#b8d99b",
        "stump": "#d62728",
        "pit": "#1f77b4",
        "stairs": "#ff7f0e",
        "roughness": "#7a7a7a",
    }
    roughness_values = [float(region["value"]) for region in layout_meta["regions"] if region["type"] == "roughness"]
    if roughness_values:
        rough_vmin = min(roughness_values)
        rough_vmax = max(roughness_values)
        if np.isclose(rough_vmin, rough_vmax):
            rough_vmax = rough_vmin + 1e-6
        rough_norm = matplotlib.colors.Normalize(vmin=rough_vmin, vmax=rough_vmax)
        rough_cmap = matplotlib.cm.get_cmap("Greys")
    else:
        rough_norm = None
        rough_cmap = None

    region_draw_order = {"grass": 0, "roughness": 1, "pit": 2, "stairs": 3, "stump": 4}
    for region in sorted(layout_meta["regions"], key=lambda r: (region_draw_order.get(r["type"], 99), r["x_start"])):
        color = region_colors.get(region["type"], "#999999")
        label_text = region["type"]
        if region["type"] == "roughness" and rough_norm is not None and rough_cmap is not None:
            color = rough_cmap(0.45 + 0.40 * float(rough_norm(region["value"])))
            label_text = f"R {region['value']:.2f}"
        width = region["x_end"] - region["x_start"]
        edgecolor = "#444444"
        linewidth = 0.8
        alpha = 0.35
        y0 = 0.0
        height = 1.0
        if region["type"] == "grass":
            continue
        elif region["type"] == "roughness":
            alpha = 0.35
            y0 = 0.0
            height = 1.0
        ax_layout.add_patch(
            plt.Rectangle(
                (region["x_start"], y0),
                width,
                height,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )
    ax_layout.axvline(layout_meta["spawn_x"], color="#444444", linestyle="--", linewidth=1.2)
    ax_layout.axvline(layout_meta["target_x"], color="#2ca02c", linestyle="--", linewidth=1.6)

    ax_hist.set_ylim(0.0, 1.0)
    ax_hist.set_yticks([])
    ax_hist.set_ylabel("Coverage")
    ax_hist.set_title(title, fontsize=12, fontweight="bold")
    ax_hist.text(
        0.01,
        0.96,
        f"max visited x = {max_data_x:.2f}m | target x = {layout_meta['target_x']:.2f}m",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax_hist.set_xlim(0.0, plot_max_x)
    ax_layout.set_xlim(0.0, plot_max_x)
    ax_layout.set_ylim(0.0, 1.0)
    ax_layout.margins(x=0.0, y=0.0)
    ax_layout.set_yticks([])
    ax_layout.set_ylabel("Layout")
    ax_layout.set_xlabel("X Position (m)")
    ax_layout.hlines(0.50, 0.0, plot_max_x, colors="#9a9a9a", linewidth=1.2)
    handles = [
        plt.Line2D([0], [0], color="#444444", linestyle="--", label="spawn"),
        plt.Line2D([0], [0], color="#2ca02c", linestyle="--", label="target"),
        plt.Rectangle((0, 0), 1, 1, color=region_colors["stump"], alpha=0.35, label="stump"),
        plt.Rectangle((0, 0), 1, 1, color=region_colors["pit"], alpha=0.35, label="pit"),
        plt.Rectangle((0, 0), 1, 1, color=region_colors["stairs"], alpha=0.35, label="stairs"),
        plt.Rectangle((0, 0), 1, 1, color=region_colors["roughness"], alpha=0.35, label="roughness"),
    ]
    ax_layout.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bipedal 1D coverage saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_saved_dataset(data_path, save_path, fig_name, env_type=None, layout_path=None):
    '''
    Visualize the saved dataset coverage heatmap
    '''
    if os.path.exists(data_path):
        print(f"Visualizing coverage from dataset: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        if env_type == "bipedalwalker" or data["a"].ndim == 2:
            if layout_path is None:
                raise ValueError("layout_path is required for Bipedal 1D coverage visualization.")
            visualize_bipedal_coverage_1d(
                data,
                layout_source=layout_path,
                save_path=save_path,
                title=f"Bipedal 1D Coverage ({fig_name})",
            )
        else:
            visualize_agent_coverage(
                data,
                save_path=save_path,
                title=f"Agent Position Coverage ({fig_name})"
            )
        print(f"Coverage heatmap saved to {save_path}")

if __name__ == "__main__": 
    data_collect() 
