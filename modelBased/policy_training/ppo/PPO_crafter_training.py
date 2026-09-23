"""Crafter World-Model PPO training.

Mirrors PPO_world_training.py but adapted for Crafter:
  - Observation: (2, H, W) — obj channel + dir channel
  - Actions: 17 (noop … make_iron_sword)
  - Inventory: 16 discrete counts (health/food/drink/energy + 12 items)
  - Reward/done: native Crafter health + first-achievement reward and
    terminated/truncated semantics; collect_diamond is evaluation-only success
  - WM map and inventory output: categorical effects (KEEP/SET_TO), decoded
    against the current state into integer next-state values
"""
from __future__ import annotations

import random
import sys
import time
import json
from collections import deque
from pathlib import Path

WM_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(WM_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_PACKAGE_ROOT))

import hydra
import numpy as np
import torch
from datetime import datetime
from omegaconf import DictConfig

from modelBased.common import utils
from modelBased.common.utils import WM_VISUALIZATIONS_PATH
from modelBased.world_model.crafter_dynamics import (
    crafter_player_counts,
    imagined_crafter_step_batch as _shared_imagined_crafter_step_batch,
)
from modelBased.policy_training.ppo.PPO import PPO
from modelBased.policy_training.common.experiment_naming import (
    policy_checkpoint_path,
    policy_selection_paths,
    policy_wandb_identity,
)
from modelBased.common.artifacts import world_model_checkpoint_path
from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import ACHIEVEMENT_NAMES, native_reward_batch
from domain.crafter.crafter_support import (
    CrafterCoverageTracker,
    crafter_reconstruct_from_logits,
    inspect_crafter_planning_checkpoint,
    load_crafter_planning_model,
)
from modelBased.policy_training.common.crafter_vector_env import CrafterSubprocessVectorEnv

import wandb

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to:", torch.cuda.get_device_name(device))
else:
    print("Device set to: cpu")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRAFTER_OBJ_CLASSES = 20
CRAFTER_DIR_CLASSES = 5
CRAFTER_ACTION_COUNT = 17
CRAFTER_PLAYER_ID = 13
CRAFTER_ACHIEVEMENT_MASK_DIM = len(ACHIEVEMENT_NAMES)
CRAFTER_ACTION_NAMES = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_policy_training(seed: int) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_crafter_agent_position_torch(states: torch.Tensor) -> torch.Tensor:
    """Find player (ID=13) positions in a (B, 2, H, W) batch of Crafter states.

    Returns (B, 2) tensor of (y, x) positions.  Falls back to (0, 0) if not found.
    """
    obj_channel = states[:, 0]          # (B, H, W)
    B, H, W = obj_channel.shape
    player_mask = (obj_channel == CRAFTER_PLAYER_ID)  # (B, H, W)
    flat = player_mask.reshape(B, -1).float()
    # argmax gives index of first True; if not found flat stays 0.
    idx = torch.argmax(flat, dim=1)     # (B,)
    ys = idx // W
    xs = idx % W
    return torch.stack([ys, xs], dim=1)  # (B, 2)


def initialize_crafter_entity_hp(states: torch.Tensor) -> torch.Tensor:
    """Initialize hidden HP for stationary no-AI entities in a map batch."""
    obj = states[:, 0]
    hp = torch.full_like(obj, -1.0)
    hp = torch.where(obj == 14, torch.full_like(hp, 3.0), hp)  # cow
    hp = torch.where(obj == 15, torch.full_like(hp, 5.0), hp)  # zombie
    hp = torch.where(obj == 16, torch.full_like(hp, 3.0), hp)  # skeleton
    return hp


def find_success_tile_position(state_numpy: np.ndarray, success_obj_id: int):
    """Return (y, x) of the first success tile, or None if absent."""
    obj_layer = state_numpy[0]  # (H, W)
    yx = np.argwhere(obj_layer == success_obj_id)
    if len(yx) > 0:
        return tuple(yx[0])
    return None


def imagined_crafter_step_batch(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    inventory: torch.Tensor,
    attention_mask_size: int,
    inventory_target_mode: str,
    *,
    predict_survival: bool = True,
    inventory_value_mode: str = "categorical_absolute",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper around the shared symbolic WM step."""
    return _shared_imagined_crafter_step_batch(
        model,
        states,
        actions,
        inventory,
        attention_mask_size,
        inventory_target_mode,
        predict_survival=predict_survival,
        inventory_value_mode=inventory_value_mode,
    )


def crafter_inventory_target_mode(observation_schema) -> str:
    """Return the configured representation of the auxiliary inventory target."""
    for field in observation_schema:
        if (
            str(field.get("name", "")) == "inventory"
            and str(field.get("target_source", "")) == "inventory"
        ):
            return str(field.get("target_mode", "categorical_gate"))
    return "categorical_gate"


def crafter_checkpoint_output_mode(checkpoint_path: str | Path) -> str:
    """Compatibility helper; planning semantics come from the checkpoint spec."""
    return inspect_crafter_planning_checkpoint(checkpoint_path).output_mode


def build_policy_state(
    states: torch.Tensor,
    inventory: torch.Tensor,
    obs_norm_values,
    unlocked_achievements: torch.Tensor,
    inventory_max: float = 9.0,
) -> torch.Tensor:
    """Flatten map, inventory, and the reward tracker achievement history."""
    B = states.shape[0]
    n_channels = states.shape[1]
    # obs_norm_values may have 3 entries (legacy MiniGrid default); take only first n_channels.
    nv = list(obs_norm_values)[:n_channels]
    norm = utils.normalize_obs(states.clone().float(), nv)
    flat_obs = norm.reshape(B, -1)                         # (B, 2*H*W)
    norm_inv = (inventory.float().clamp(0) / inventory_max)  # (B, 16)
    unlocked = torch.as_tensor(
        unlocked_achievements, device=states.device, dtype=torch.float32
    )
    if unlocked.ndim != 2 or tuple(unlocked.shape) != (B, CRAFTER_ACHIEVEMENT_MASK_DIM):
        raise ValueError(
            "Crafter policy achievement mask must have shape "
            f"({B}, {CRAFTER_ACHIEVEMENT_MASK_DIM}), got {tuple(unlocked.shape)}"
        )
    return torch.cat([flat_obs, norm_inv, unlocked], dim=1)


def crafter_achievement_mask_from_names(
    newly_unlocked, *, device_: torch.device | None = None
) -> torch.Tensor:
    """Encode environment-reported first unlocks in the canonical 22-slot order."""
    mask = torch.zeros(
        CRAFTER_ACHIEVEMENT_MASK_DIM, device=device_, dtype=torch.bool
    )
    for name in newly_unlocked:
        try:
            mask[ACHIEVEMENT_NAMES.index(str(name))] = True
        except ValueError as exc:
            raise ValueError(f"Unknown Crafter achievement from environment: {name!r}") from exc
    return mask


def crafter_policy_selection_is_better(candidate: dict, incumbent: dict | None) -> bool:
    """Rank frozen-WM evaluations without using real-environment feedback."""
    if incumbent is None:
        return True
    candidate_key = (
        int(candidate["success_count"]),
        float(candidate["mean_achievement_count"]),
        float(candidate["mean_native_return"]),
        -float(candidate["mean_steps"]),
    )
    incumbent_key = (
        int(incumbent["success_count"]),
        float(incumbent["mean_achievement_count"]),
        float(incumbent["mean_native_return"]),
        -float(incumbent["mean_steps"]),
    )
    return candidate_key > incumbent_key


def evaluate_deterministic_crafter_wm_policy(
    ppo_agent: PPO,
    model,
    wm_spec,
    initial_states: list[torch.Tensor],
    initial_inventories: list[torch.Tensor],
    max_steps: int,
) -> dict:
    """Run fixed argmax episodes entirely inside the frozen Crafter WM."""
    outcomes = []
    diamond_index = ACHIEVEMENT_NAMES.index("collect_diamond")
    for initial_state, initial_inventory in zip(initial_states, initial_inventories):
        states = initial_state.unsqueeze(0).clone()
        inventory = initial_inventory.unsqueeze(0).clone()
        unlocked = torch.zeros((1, len(ACHIEVEMENT_NAMES)), device=device, dtype=torch.bool)
        life_state = {
            "hunger": torch.zeros(1, device=device), "thirst": torch.zeros(1, device=device),
            "fatigue": torch.zeros(1, device=device), "recover": torch.zeros(1, device=device),
            "sleeping": torch.zeros(1, device=device, dtype=torch.bool),
        }
        cow_hp = initialize_crafter_entity_hp(states)
        native_return = 0.0
        success = False
        for step in range(1, max_steps + 1):
            policy_state = build_policy_state(states, inventory, wm_spec.obs_norm_values, unlocked)
            with torch.no_grad():
                action = torch.argmax(ppo_agent.policy_old.actor(policy_state), dim=-1)
                previous_states, previous_inventory = states.clone(), inventory.clone()
                states, inventory = imagined_crafter_step_batch(
                    model, states, action, inventory, wm_spec.attention_mask_size,
                    wm_spec.inventory_output_mode, predict_survival=wm_spec.predict_survival,
                    inventory_value_mode=wm_spec.inventory_value_mode,
                )
                if bool(crafter_player_counts(states).ne(1).any()):
                    # Treat an invalid learned transition as an unsuccessful end,
                    # exactly as the training rollout guard does.
                    break
                rewards, newly_unlocked, unlocked, life_state, cow_hp, inventory = native_reward_batch(
                    previous_states, previous_inventory, action, states, inventory,
                    unlocked, life_state["sleeping"], cow_hp, life_state,
                )
            native_return += float(rewards.item())
            success |= bool(newly_unlocked[0, diamond_index].item())
            if bool(inventory[0, 0] <= 0):
                break
        outcomes.append((success, int(unlocked.sum().item()), native_return, step))
    return {
        "success_count": sum(int(row[0]) for row in outcomes),
        "mean_achievement_count": float(np.mean([row[1] for row in outcomes])),
        "mean_native_return": float(np.mean([row[2] for row in outcomes])),
        "mean_steps": float(np.mean([row[3] for row in outcomes])),
        "episodes": outcomes,
    }


def _init_crafter_wandb(cfg, default_project: str):
    ppo_cfg = cfg.PPO
    wandb_group, wandb_run_name, training_source = policy_wandb_identity(cfg)
    task_name = str(cfg.domains["crafter"].task_name)
    wandb_dir = Path(
        str(getattr(getattr(cfg, "paths", None), "wandb", utils.WM_OUTPUTS_PATH / "wandb"))
    ).expanduser().resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.login()
    init_kwargs = {
        "dir": str(wandb_dir),
        "project": str(getattr(ppo_cfg, "wandb_project", default_project)),
        "name": wandb_run_name,
        "group": wandb_group,
        "job_type": task_name,
        "tags": [training_source, task_name, "crafter"],
        "reinit": True,
        "config": {
            "domain": "crafter",
            "task_name": task_name,
            "layout_path": str(cfg.domains["crafter"].layout_path),
            "training_source": training_source,
            "policy_checkpoint": str(policy_checkpoint_path(cfg, domain="crafter")),
            "seed": int(getattr(ppo_cfg, "seed", 0)),
            "max_ep_len": int(ppo_cfg.max_ep_len),
            "rollout_steps": int(getattr(ppo_cfg, "rollout_steps", 4096)),
            "num_imagined_envs": int(getattr(ppo_cfg, "num_imagined_envs", 8)),
            "max_training_timesteps": int(ppo_cfg.max_training_timesteps),
        },
    }
    entity = getattr(ppo_cfg, "wandb_entity", None)
    if entity:
        init_kwargs["entity"] = str(entity)
    return wandb.init(**init_kwargs)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_ppo_crafter_wm(cfg: DictConfig):
    hparams = cfg
    hparams_wm = hparams.attention_model
    hparams_ppo = hparams.PPO
    domain_cfg = hparams.domains["crafter"]

    # ---- 1. Load World Model ----
    configured_wm_ckpt = getattr(hparams_ppo, "checkpoint_path_wm", None)
    wm_ckpt = str(
        Path(str(configured_wm_ckpt)).expanduser().resolve()
        if configured_wm_ckpt is not None and str(configured_wm_ckpt).strip() and str(configured_wm_ckpt).lower() != "null"
        else world_model_checkpoint_path(hparams, "crafter")
    )
    model, wm_spec = load_crafter_planning_model(wm_ckpt)
    model.to(device)
    print(f"[Crafter PPO] WM loaded from: {wm_ckpt} ({wm_spec})")

    # ---- 2. PPO hyperparameters ----
    seed                   = int(getattr(hparams_ppo, "seed", 0))
    lr_actor               = hparams_ppo.lr_actor
    lr_critic              = hparams_ppo.lr_critic
    gamma                  = hparams_ppo.gamma
    K_epochs               = hparams_ppo.K_epochs
    eps_clip               = hparams_ppo.eps_clip
    action_std             = hparams_ppo.action_std
    max_training_timesteps = int(hparams_ppo.max_training_timesteps)
    save_model_freq        = int(hparams_ppo.save_model_freq)
    max_ep_len             = int(hparams_ppo.max_ep_len)
    has_continuous         = hparams_ppo.has_continuous_action_space
    update_timestep        = int(getattr(hparams_ppo, "rollout_steps", 4096))
    num_imagined_envs      = int(getattr(hparams_ppo, "num_imagined_envs", 8))
    entropy_coef           = float(getattr(hparams_ppo, "entropy_coef", 0.01))
    normalize_advantages   = bool(getattr(hparams_ppo, "normalize_advantages", True))
    normalize_returns      = bool(getattr(hparams_ppo, "normalize_returns", False))
    max_grad_norm          = float(getattr(hparams_ppo, "max_grad_norm", 0.5))
    rolling_window         = int(getattr(hparams_ppo, "rolling_window_episodes", 50))
    step_penalty           = float(getattr(hparams_ppo, "step_penalty", 0.0))
    use_wandb              = bool(hparams_ppo.use_wandb)
    checkpoint_path        = str(policy_checkpoint_path(cfg, domain="crafter"))

    if update_timestep < 2:
        raise ValueError("PPO.rollout_steps must be at least 2")
    if num_imagined_envs < 1:
        raise ValueError("PPO.num_imagined_envs must be at least 1")
    if update_timestep % num_imagined_envs != 0:
        raise ValueError("PPO.rollout_steps must be divisible by PPO.num_imagined_envs")
    if max_training_timesteps % num_imagined_envs != 0:
        raise ValueError("PPO.max_training_timesteps must be divisible by PPO.num_imagined_envs")
    best_policy_selection_enabled = bool(
        getattr(hparams_ppo, "best_policy_selection_enabled", True)
    )
    selection_eval_freq_value = getattr(hparams_ppo, "best_policy_eval_freq", None)
    selection_eval_freq = (
        save_model_freq if selection_eval_freq_value is None else int(selection_eval_freq_value)
    )
    selection_episodes = int(getattr(hparams_ppo, "best_policy_eval_episodes", 8))
    selection_seed = int(getattr(hparams_ppo, "best_policy_eval_seed", seed))
    selection_max_steps_value = getattr(hparams_ppo, "best_policy_max_steps", None)
    selection_max_steps = (
        max_ep_len if selection_max_steps_value is None else int(selection_max_steps_value)
    )
    if best_policy_selection_enabled and (
        selection_eval_freq < 1 or selection_episodes < 1 or selection_max_steps < 1
    ):
        raise ValueError(
            "PPO best-policy evaluation frequency, episodes, and max_steps must be positive"
        )
    selection_paths = policy_selection_paths(checkpoint_path) if best_policy_selection_enabled else None

    # ---- 3. Crafter-specific dimensions ----
    inventory_dim   = int(getattr(domain_cfg, "inventory_dim", 16))
    obs_norm_values = list(wm_spec.obs_norm_values)
    inventory_target_mode = wm_spec.inventory_output_mode
    predict_survival = wm_spec.predict_survival
    if inventory_target_mode not in {"categorical_gate", "categorical_effect"}:
        raise ValueError(
            "Crafter WM planning requires categorical_gate or categorical_effect inventory, "
            f"got {inventory_target_mode!r}"
        )

    # ---- 4. Build real env (for resets only) ----
    seed_policy_training(seed)
    env_path = str(domain_cfg.layout_path)
    env = CustomCrafterEnv(
        txt_file_path=env_path,
        max_steps=max_ep_len,
        seed=seed,
    )
    # Determine map dimensions from one reset
    sample_obs, _ = env.reset()
    sample_grid = sample_obs["image"]        # (H, W, 2) → will transpose
    H, W = sample_grid.shape[0], sample_grid.shape[1]
    state_dim = 2 * H * W + inventory_dim + CRAFTER_ACHIEVEMENT_MASK_DIM

    # Success is an evaluation metric only. It is latched when the imagined
    # native-reward tracker first unlocks collect_diamond; it neither changes
    # reward nor terminates the episode.
    diamond_achievement_index = ACHIEVEMENT_NAMES.index("collect_diamond")

    # ---- 5. PPO agent ----
    seed_policy_training(seed)
    ppo_agent = PPO(
        state_dim,
        CRAFTER_ACTION_COUNT,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous,
        action_std,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages,
        normalize_returns=normalize_returns,
        max_grad_norm=max_grad_norm,
        minibatch_size=int(getattr(hparams_ppo, "minibatch_size", 0)),
    )

    # Fixed starts make best-policy ranking reproducible while keeping every
    # selection rollout entirely in the frozen WM.
    selection_states, selection_inventories = [], []
    if best_policy_selection_enabled:
        for episode_index in range(selection_episodes):
            selection_env = CustomCrafterEnv(
                txt_file_path=env_path,
                max_steps=max_ep_len,
                seed=selection_seed + episode_index,
            )
            selection_obs, _ = selection_env.reset()
            selection_states.append(torch.as_tensor(
                np.transpose(selection_obs["image"], (2, 0, 1)),
                device=device, dtype=torch.float32,
            ))
            selection_inventories.append(torch.as_tensor(
                selection_obs["inventory"], device=device, dtype=torch.float32
            ))
            selection_env.close()

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
                "Lexicographic: maximize collect_diamond success_count; then mean "
                "unlocked achievement count; then mean native imagined return; then "
                "minimize mean steps. Fixed reset seeds; no real-environment evaluation is used."
            ),
            "enabled": True,
            "seed": selection_seed,
            "episodes": selection_episodes,
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

    def evaluate_and_select_policy(timestep: int):
        nonlocal best_selection_score, last_selection_timestep
        if not best_policy_selection_enabled or last_selection_timestep == timestep:
            return
        metrics = evaluate_deterministic_crafter_wm_policy(
            ppo_agent, model, wm_spec, selection_states, selection_inventories,
            selection_max_steps,
        )
        candidate = {"timestep": int(timestep), **metrics}
        selection_history.append(candidate)
        last_selection_timestep = int(timestep)
        ppo_agent.save(selection_paths["last"])
        selected = crafter_policy_selection_is_better(candidate, best_selection_score)
        if selected:
            best_selection_score = candidate
            ppo_agent.save(selection_paths["best"])
            # Existing consumers use the canonical short name; it is always best.
            ppo_agent.save(selection_paths["canonical"])
        write_policy_selection_manifest()
        print(
            "[Crafter WM PPO best-policy] "
            f"t={timestep} success={candidate['success_count']}/{selection_episodes} "
            f"achievements={candidate['mean_achievement_count']:.2f} "
            f"return={candidate['mean_native_return']:.3f} "
            f"steps={candidate['mean_steps']:.1f} selected={selected}"
        )

    # ---- 6. WandB ----
    if use_wandb:
        sub_run = _init_crafter_wandb(cfg, default_project="crafter_policy_training")
    else:
        sub_run = None

    def log_ppo_update(metrics):
        if sub_run is not None and metrics.get("updated", False):
            sub_run.log(
                {
                    "ppo/loss": metrics["loss"],
                    "ppo/grad_norm": metrics["grad_norm"],
                    "ppo/parameter_delta": metrics["parameter_delta"],
                    "ppo/rollout_size": metrics["rollout_size"],
                    "ppo/update_count": metrics["update_count"],
                },
                step=time_step,
            )

    evaluate_and_select_policy(0)

    # ---- 7. Initialise imagined trajectories ----
    def reset_imagined_state():
        obs, _ = env.reset()
        grid = obs["image"]  # (H, W, 2)
        chw = np.transpose(grid, (2, 0, 1))  # (2, H, W)
        return (
            torch.as_tensor(chw, device=device, dtype=torch.float32),
            torch.as_tensor(obs["inventory"], device=device, dtype=torch.float32),
        )

    state_list, inv_list = zip(*[reset_imagined_state() for _ in range(num_imagined_envs)])
    states    = torch.stack(list(state_list), dim=0)   # (B, 2, H, W)
    inventory = torch.stack(list(inv_list), dim=0)     # (B, 16)
    # Entity HP is hidden from the symbolic grid, so track it per map cell.
    cow_hp = initialize_crafter_entity_hp(states)
    unlocked = torch.zeros(
        (num_imagined_envs, len(ACHIEVEMENT_NAMES)), device=device, dtype=torch.bool
    )
    life_state = {
        "hunger": torch.zeros(num_imagined_envs, device=device),
        "thirst": torch.zeros(num_imagined_envs, device=device),
        "fatigue": torch.zeros(num_imagined_envs, device=device),
        "recover": torch.zeros(num_imagined_envs, device=device),
        "sleeping": torch.zeros(
            num_imagined_envs, device=device, dtype=torch.bool
        ),
    }

    episode_rewards = torch.zeros(num_imagined_envs, device=device)
    episode_steps   = torch.zeros(num_imagined_envs, device=device, dtype=torch.long)
    episode_success = torch.zeros(
        num_imagined_envs, device=device, dtype=torch.bool
    )
    last_terminated = torch.zeros(num_imagined_envs, device=device, dtype=torch.bool)

    def reset_imagined_slots(slots: torch.Tensor) -> None:
        """Reset invalid or completed imagined environments without logging them."""
        if slots.numel() == 0:
            return
        reset_states, reset_invs = zip(
            *[reset_imagined_state() for _ in range(slots.numel())]
        )
        states[slots] = torch.stack(list(reset_states))
        inventory[slots] = torch.stack(list(reset_invs))
        cow_hp[slots] = initialize_crafter_entity_hp(states[slots])
        unlocked[slots] = False
        for value in life_state.values():
            value[slots] = 0
        episode_rewards[slots] = 0.0
        episode_steps[slots] = 0
        episode_success[slots] = False
        last_terminated[slots] = False

    recent_rewards   = deque(maxlen=rolling_window)
    recent_steps     = deque(maxlen=rolling_window)
    recent_successes = deque(maxlen=rolling_window)

    time_step              = 0
    i_episode              = 0
    print_freq             = 1000
    next_print_timestep    = print_freq
    next_save_timestep     = save_model_freq
    print_running_reward   = 0.0
    print_running_episodes = 0
    print_running_steps    = 0
    print_running_successes= 0
    print_running_achievements = 0
    invalid_agent_transitions = 0
    start_time             = datetime.now().replace(microsecond=0)
    final_norm_regret      = None

    print(
        f"[Crafter WM PPO] Parallel imagined envs: {num_imagined_envs}; "
        f"temporal rollout: {update_timestep // num_imagined_envs}; "
        f"transitions/update: {update_timestep}"
    )

    # ---- 8. Training loop ----
    while time_step < max_training_timesteps:
        # Match MiniGrid's invalid-transition guard before entering the local
        # pose decoder.  This is not an environment death or episode result.
        malformed_before_step = crafter_player_counts(states).ne(1)
        if bool(malformed_before_step.any()):
            malformed_slots = torch.nonzero(
                malformed_before_step, as_tuple=False
            ).reshape(-1)
            invalid_agent_transitions += int(malformed_slots.numel())
            reset_imagined_slots(malformed_slots)

        state_norm = build_policy_state(states, inventory, obs_norm_values, unlocked)
        (
            actions,
            state_buffer,
            action_buffer,
            action_logprobs,
            state_values,
        ) = ppo_agent.select_action_batch(state_norm)

        previous_states = states.clone()
        previous_inventory = inventory.clone()
        previous_unlocked = unlocked.clone()
        previous_cow_hp = cow_hp.clone()
        previous_life_state = {name: value.clone() for name, value in life_state.items()}
        with torch.no_grad():
            states, inventory = imagined_crafter_step_batch(
                model,
                states,
                actions,
                inventory,
                wm_spec.attention_mask_size,
                inventory_target_mode,
                predict_survival=predict_survival,
                inventory_value_mode=wm_spec.inventory_value_mode,
            )

        episode_steps += 1
        with torch.no_grad():
            (
                rewards,
                newly_unlocked,
                unlocked,
                life_state,
                cow_hp,
                inventory,
            ) = native_reward_batch(
                previous_states,
                previous_inventory,
                actions,
                states,
                inventory,
                unlocked,
                life_state["sleeping"],
                cow_hp,
                life_state,
            )
        invalid_transitions = crafter_player_counts(states).ne(1)
        valid_transitions = ~invalid_transitions
        # A malformed WM transition must not mutate reward-tracker state or
        # become a PPO sample.  Restore it before resetting that slot below.
        if bool(invalid_transitions.any()):
            invalid_agent_transitions += int(invalid_transitions.sum().item())
            states = torch.where(
                invalid_transitions[:, None, None, None], previous_states, states
            )
            inventory = torch.where(
                invalid_transitions[:, None], previous_inventory, inventory
            )
            unlocked = torch.where(
                invalid_transitions[:, None], previous_unlocked, unlocked
            )
            cow_hp = torch.where(
                invalid_transitions[:, None, None], previous_cow_hp, cow_hp
            )
            for name, value in life_state.items():
                life_state[name] = torch.where(
                    invalid_transitions, previous_life_state[name], value
                )
            newly_unlocked = newly_unlocked & valid_transitions[:, None]
        episode_success |= newly_unlocked[:, diamond_achievement_index]
        truncated = (episode_steps >= max_ep_len) & valid_transitions
        terminated = (inventory[:, 0] <= 0) & valid_transitions
        dones = terminated | truncated | invalid_transitions
        rewards = rewards + step_penalty
        rewards = torch.where(invalid_transitions, torch.zeros_like(rewards), rewards)
        episode_rewards += rewards

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
        last_terminated = terminated.clone()

        # PPO update
        if time_step % update_timestep == 0:
            next_norm = build_policy_state(states, inventory, obs_norm_values, unlocked)
            bootstrap_values = ppo_agent.estimate_old_values_batch(next_norm)
            # A time-limit truncation may bootstrap; only true death is a
            # terminal state for the native Crafter return.
            bootstrap_values[last_terminated.detach().cpu()] = 0.0
            update_metrics = ppo_agent.update(bootstrap_value=bootstrap_values)
            log_ppo_update(update_metrics)
            if best_policy_selection_enabled and time_step >= next_selection_timestep:
                evaluate_and_select_policy(time_step)
                while next_selection_timestep <= time_step:
                    next_selection_timestep += selection_eval_freq

        # Agent-invalid slots are reset, but deliberately not logged as
        # completed native episodes.
        completed = torch.nonzero(
            (terminated | truncated) & valid_transitions, as_tuple=False
        ).reshape(-1)
        if completed.numel() > 0:
            comp_rewards  = episode_rewards[completed].detach().cpu().tolist()
            comp_steps    = episode_steps[completed].detach().cpu().tolist()
            comp_successes = episode_success[completed].detach().cpu().int().tolist()
            comp_achievements = unlocked[completed].sum(dim=1).detach().cpu().tolist()
            for ep_r, ep_s, ep_ok, ep_ach in zip(
                comp_rewards, comp_steps, comp_successes, comp_achievements
            ):
                print_running_reward    += ep_r
                print_running_steps     += ep_s
                print_running_successes += ep_ok
                print_running_achievements += ep_ach
                print_running_episodes  += 1
                recent_rewards.append(ep_r)
                recent_steps.append(ep_s)
                recent_successes.append(ep_ok)
                i_episode += 1

            if sub_run is not None:
                sub_run.log(
                    {
                        "episode/reward": sum(comp_rewards) / len(comp_rewards),
                        "episode/success": sum(comp_successes) / len(comp_successes),
                        "episode/achievements": sum(comp_achievements) / len(comp_achievements),
                        "episode/steps": sum(comp_steps) / len(comp_steps),
                        "episode/completed_count": len(comp_rewards),
                        "episode/index": i_episode,
                        "rolling/average_reward": sum(recent_rewards) / len(recent_rewards),
                        "rolling/success_rate": sum(recent_successes) / len(recent_successes),
                        "rolling/average_episode_steps": sum(recent_steps) / len(recent_steps),
                        "rolling/window_episode_count": len(recent_rewards),
                    },
                    step=time_step,
                )

        reset_slots = torch.nonzero(dones, as_tuple=False).reshape(-1)
        reset_imagined_slots(reset_slots)

        if (
            sub_run is not None
            and invalid_agent_transitions
            and time_step % update_timestep == 0
        ):
            sub_run.log(
                {"wm/invalid_agent_transitions": invalid_agent_transitions},
                step=time_step,
            )

        # Print interval
        if time_step >= next_print_timestep and print_running_episodes > 0:
            rol_r  = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            rol_sr = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            rol_st = sum(recent_steps) / len(recent_steps) if recent_steps else 0.0
            print(
                f"Episode : {i_episode} \t Timestep : {time_step} \t "
                f"Interval Reward : {print_running_reward / print_running_episodes:.5f} \t "
                f"Interval Success : {print_running_successes / print_running_episodes:.1%} \t "
                f"Interval Achievements : {print_running_achievements / print_running_episodes:.2f} \t "
                f"Interval Steps : {print_running_steps / print_running_episodes:.1f} \t "
                f"Rolling({len(recent_rewards)}) Reward : {rol_r:.5f} \t "
                f"Success : {rol_sr:.1%} \t Steps : {rol_st:.1f}"
            )
            print_running_reward    = 0.0
            print_running_episodes  = 0
            print_running_steps     = 0
            print_running_successes = 0
            print_running_achievements = 0
            while next_print_timestep <= time_step:
                next_print_timestep += print_freq

        # Save interval
        if time_step >= next_save_timestep:
            save_path = selection_paths["last"] if best_policy_selection_enabled else checkpoint_path
            print(f"Saving policy at: {save_path}")
            ppo_agent.save(save_path)
            print(f"Elapsed: {datetime.now().replace(microsecond=0) - start_time}")
            while next_save_timestep <= time_step:
                next_save_timestep += save_model_freq

    # ---- 9. Final update on partial rollout ----
    if ppo_agent.buffer.transition_count() >= 2:
        next_norm = build_policy_state(states, inventory, obs_norm_values, unlocked)
        bootstrap_values = ppo_agent.estimate_old_values_batch(next_norm)
        bootstrap_values[last_terminated.detach().cpu()] = 0.0
        update_metrics = ppo_agent.update(bootstrap_value=bootstrap_values)
        log_ppo_update(update_metrics)
    else:
        ppo_agent.buffer.clear()

    if best_policy_selection_enabled:
        # The final optimizer state is retained separately; canonical stays best.
        ppo_agent.save(selection_paths["last"])
        if last_selection_timestep != time_step:
            evaluate_and_select_policy(time_step)
        print(f"[Crafter PPO] Best policy saved: {selection_paths['best']}")
        print(f"[Crafter PPO] Last policy saved: {selection_paths['last']}")
    else:
        ppo_agent.save(checkpoint_path)
        print(f"[Crafter PPO] Final policy saved: {checkpoint_path}")
    env.close()
    if use_wandb and sub_run is not None:
        sub_run.finish()
    return final_norm_regret


# ---------------------------------------------------------------------------
# Real-environment PPO baseline
# ---------------------------------------------------------------------------

def _build_real_policy_state_batch(obs, obs_norm_values, unlocked_achievements):
    """Convert a vector-env observation dict into one policy batch."""
    states = torch.as_tensor(
        np.transpose(obs["image"], (0, 3, 1, 2)),
        device=device,
        dtype=torch.float32,
    )
    inventory = torch.as_tensor(
        obs["inventory"], device=device, dtype=torch.float32
    )
    return build_policy_state(states, inventory, obs_norm_values, unlocked_achievements)


def _new_real_env_coverage_tracker(cfg):
    if not bool(getattr(cfg.PPO, "save_real_env_coverage", True)):
        return None
    return CrafterCoverageTracker()


def _save_real_env_coverage(cfg, tracker):
    if tracker is None or tracker.position_samples == 0:
        return None
    ppo_cfg = cfg.PPO
    task_name = str(cfg.domains["crafter"].task_name)
    seed = int(getattr(ppo_cfg, "seed", 0))
    save_dir = Path(
        str(
            getattr(
                ppo_cfg,
                "real_env_coverage_save_path",
                WM_VISUALIZATIONS_PATH / "real_env_coverage",
            )
        )
    ).expanduser().resolve()
    filename = str(
        getattr(
            ppo_cfg,
            "real_env_coverage_filename",
            f"{task_name}_realenv_seed{seed}_coverage.png",
        )
    )
    return tracker.save(
        save_dir / filename,
        title=(
            f"Crafter Real-Env Inventory Coverage "
            f"({task_name}, seed={seed}, transitions={tracker.position_samples})"
        ),
    )


def _run_ppo_crafter_real_parallel(cfg: DictConfig, num_real_envs: int):
    """Train real-env PPO with synchronous subprocess environment sampling."""
    ppo_cfg = cfg.PPO
    domain_cfg = cfg.domains["crafter"]
    seed = seed_policy_training(int(getattr(ppo_cfg, "seed", 0)))
    max_ep_len = int(ppo_cfg.max_ep_len)
    max_training_timesteps = int(ppo_cfg.max_training_timesteps)
    rollout_steps = int(getattr(ppo_cfg, "rollout_steps", 4096))
    if rollout_steps < 2 or rollout_steps % num_real_envs != 0:
        raise ValueError(
            "PPO.rollout_steps must be at least 2 and divisible by num_real_envs"
        )
    if max_training_timesteps % num_real_envs != 0:
        raise ValueError("PPO.max_training_timesteps must be divisible by num_real_envs")

    obs_norm_values = list(cfg.attention_model.obs_norm_values)
    checkpoint_path = str(policy_checkpoint_path(cfg, domain="crafter"))
    envs = CrafterSubprocessVectorEnv(
        layout_path=str(domain_cfg.layout_path),
        max_steps=max_ep_len,
        num_envs=num_real_envs,
        seed=seed,
        start_method=str(getattr(ppo_cfg, "real_env_start_method", "spawn")),
    )

    coverage_tracker = _new_real_env_coverage_tracker(cfg)
    try:
        obs, _ = envs.reset()
        grid_shape = np.transpose(obs["image"][0], (2, 0, 1)).shape
        inventory_dim = int(getattr(domain_cfg, "inventory_dim", 16))
        state_dim = int(np.prod(grid_shape) + inventory_dim + CRAFTER_ACHIEVEMENT_MASK_DIM)
        ppo_agent = PPO(
            state_dim, CRAFTER_ACTION_COUNT,
            ppo_cfg.lr_actor, ppo_cfg.lr_critic, ppo_cfg.gamma,
            ppo_cfg.K_epochs, ppo_cfg.eps_clip,
            ppo_cfg.has_continuous_action_space, ppo_cfg.action_std,
            entropy_coef=float(getattr(ppo_cfg, "entropy_coef", 0.01)),
            normalize_advantages=bool(getattr(ppo_cfg, "normalize_advantages", True)),
            normalize_returns=bool(getattr(ppo_cfg, "normalize_returns", False)),
            max_grad_norm=float(getattr(ppo_cfg, "max_grad_norm", 0.5)),
            minibatch_size=int(getattr(ppo_cfg, "minibatch_size", 0)),
        )
        use_wandb = bool(getattr(ppo_cfg, "use_wandb", False))
        sub_run = _init_crafter_wandb(cfg, "crafter_policy_training") if use_wandb else None
        step_penalty = float(getattr(ppo_cfg, "step_penalty", 0.0))
        save_model_freq = int(ppo_cfg.save_model_freq)
        next_save_timestep = save_model_freq
        recent_rewards = deque(maxlen=int(getattr(ppo_cfg, "rolling_window_episodes", 50)))
        episode_reward = np.zeros(num_real_envs, dtype=np.float32)
        episode_steps = np.zeros(num_real_envs, dtype=np.int64)
        episode_health_reward = np.zeros(num_real_envs, dtype=np.float32)
        episode_achievement_reward = np.zeros(num_real_envs, dtype=np.float32)
        episode_unlocks = [set() for _ in range(num_real_envs)]
        unlocked_masks = torch.zeros(
            (num_real_envs, CRAFTER_ACHIEVEMENT_MASK_DIM), device=device, dtype=torch.bool
        )
        episode_action_counts = np.zeros((num_real_envs, CRAFTER_ACTION_COUNT), dtype=np.int64)
        time_step = 0
        episode_index = 0
        start_time = time.perf_counter()

        print(
            f"[Crafter Real PPO] Parallel environments={num_real_envs}; "
            f"rollout={rollout_steps} transitions/update"
        )
        while time_step < max_training_timesteps:
            if coverage_tracker is not None:
                coverage_tracker.update(obs["image"], obs["inventory"])
            state = _build_real_policy_state_batch(obs, obs_norm_values, unlocked_masks)
            actions, state_buf, action_buf, logprob_buf, value_buf = \
                ppo_agent.select_action_batch(state)
            next_obs, rewards, terminated, truncated, infos = envs.step(
                actions.detach().cpu().numpy()
            )
            rewards = rewards + step_penalty
            dones = terminated | truncated
            ppo_agent.save_buffer_batch(
                state_buf, action_buf, logprob_buf, value_buf, rewards, dones
            )

            episode_reward += rewards
            episode_steps += 1
            time_step += num_real_envs
            for index, info in enumerate(infos):
                episode_health_reward[index] += float(info.get("health_reward", 0.0))
                newly_unlocked = info.get("newly_unlocked", [])
                unlocked_masks[index] |= crafter_achievement_mask_from_names(
                    newly_unlocked, device_=device
                )
                episode_achievement_reward[index] += float(bool(newly_unlocked))
                episode_unlocks[index].update(newly_unlocked)
                episode_action_counts[index, int(actions[index])] += 1

            if time_step % rollout_steps == 0:
                next_state = _build_real_policy_state_batch(
                    next_obs, obs_norm_values, unlocked_masks
                )
                bootstrap_values = ppo_agent.estimate_old_values_batch(next_state)
                bootstrap_values[torch.as_tensor(terminated)] = 0.0
                update_metrics = ppo_agent.update(bootstrap_value=bootstrap_values)
                if sub_run is not None and update_metrics.get("updated", False):
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    sub_run.log(
                        {
                            "env/num_real_envs": num_real_envs,
                            "env/steps_per_second": time_step / elapsed,
                            "env/episode_throughput": episode_index / elapsed,
                            "ppo/loss": update_metrics["loss"],
                            "ppo/grad_norm": update_metrics["grad_norm"],
                            "ppo/parameter_delta": update_metrics["parameter_delta"],
                            "ppo/update_count": update_metrics["update_count"],
                        },
                        step=time_step,
                    )

            done_indices = np.flatnonzero(dones)
            for index in done_indices:
                info = infos[index]
                unlock_names = sorted(episode_unlocks[index])
                final_inventory = info.get("inventory", {})
                episode_index += 1
                recent_rewards.append(float(episode_reward[index]))
                print(
                    f"[Crafter Real PPO] episode={episode_index} timestep={time_step} "
                    f"env={index} steps={episode_steps[index]} "
                    f"native_reward={episode_reward[index]:.4f} "
                    f"achievement_reward={episode_achievement_reward[index]:.1f} "
                    f"unlock_count={len(unlock_names)} "
                    f"health_reward={episode_health_reward[index]:.4f} "
                    f"terminated={bool(terminated[index])} "
                    f"truncated={bool(truncated[index])} unlocked={unlock_names}"
                )
                if sub_run is not None:
                    metrics = {
                        "episode/reward": float(episode_reward[index]),
                        "episode/achievement_reward": float(episode_achievement_reward[index]),
                        "episode/unlock_count": float(len(unlock_names)),
                        "episode/health_reward": float(episode_health_reward[index]),
                        "episode/steps": int(episode_steps[index]),
                        "episode/index": episode_index,
                        "episode/terminated": float(terminated[index]),
                        "episode/truncated": float(truncated[index]),
                        "episode/final_health": float(final_inventory.get("health", 0)),
                        "rolling/average_reward": sum(recent_rewards) / len(recent_rewards),
                        "env/num_real_envs": num_real_envs,
                    }
                    metrics.update({
                        f"achievement/{name}": float(name in episode_unlocks[index])
                        for name in ACHIEVEMENT_NAMES
                    })
                    sub_run.log(metrics, step=time_step)
                episode_reward[index] = 0.0
                episode_steps[index] = 0
                episode_health_reward[index] = 0.0
                episode_achievement_reward[index] = 0.0
                episode_unlocks[index].clear()
                unlocked_masks[index] = False
                episode_action_counts[index].fill(0)

            if done_indices.size:
                obs = next_obs
                reset_obs, _ = envs.reset_at(done_indices)
                obs["image"][done_indices] = reset_obs["image"]
                obs["inventory"][done_indices] = reset_obs["inventory"]
            else:
                obs = next_obs

            if time_step >= next_save_timestep:
                ppo_agent.save(checkpoint_path)
                _save_real_env_coverage(cfg, coverage_tracker)
                print(f"[Crafter Real PPO] checkpoint saved at timestep={time_step}: {checkpoint_path}")
                while next_save_timestep <= time_step:
                    next_save_timestep += save_model_freq

            if time_step and time_step % max(rollout_steps, 1000) == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                print(
                    f"[Crafter Real PPO] throughput={time_step / elapsed:.1f} "
                    f"transitions/s envs={num_real_envs}"
                )

        if ppo_agent.buffer.transition_count() >= num_real_envs:
            final_state = _build_real_policy_state_batch(obs, obs_norm_values, unlocked_masks)
            bootstrap_values = ppo_agent.estimate_old_values_batch(final_state)
            ppo_agent.update(bootstrap_value=bootstrap_values)
        ppo_agent.save(checkpoint_path)
        _save_real_env_coverage(cfg, coverage_tracker)
        print(f"[Crafter Real PPO] Final policy saved: {checkpoint_path}")
        if sub_run is not None:
            sub_run.finish()
    finally:
        envs.close()

def run_ppo_crafter_real(cfg: DictConfig):
    """Train Crafter PPO directly from native environment transitions."""
    num_real_envs = int(getattr(cfg.PPO, "num_real_envs", 1))
    if num_real_envs < 1:
        raise ValueError("PPO.num_real_envs must be at least 1")
    backend = str(getattr(cfg.PPO, "real_env_backend", "subprocess"))
    if num_real_envs > 1 and backend != "subprocess":
        raise ValueError(
            "Only PPO.real_env_backend=subprocess is supported for parallel real PPO"
        )
    if num_real_envs > 1:
        return _run_ppo_crafter_real_parallel(cfg, num_real_envs)

    ppo_cfg = cfg.PPO
    domain_cfg = cfg.domains["crafter"]
    seed = seed_policy_training(int(getattr(ppo_cfg, "seed", 0)))
    max_ep_len = int(ppo_cfg.max_ep_len)
    max_training_timesteps = int(ppo_cfg.max_training_timesteps)
    rollout_steps = int(getattr(ppo_cfg, "rollout_steps", 4096))
    if rollout_steps < 2:
        raise ValueError("PPO.rollout_steps must be at least 2")

    env = CustomCrafterEnv(
        txt_file_path=str(domain_cfg.layout_path), max_steps=max_ep_len, seed=seed
    )
    sample_obs, _ = env.reset()
    grid_shape = np.transpose(sample_obs["image"], (2, 0, 1)).shape
    inventory_dim = int(getattr(domain_cfg, "inventory_dim", 16))
    state_dim = int(np.prod(grid_shape) + inventory_dim + CRAFTER_ACHIEVEMENT_MASK_DIM)
    obs_norm_values = list(cfg.attention_model.obs_norm_values)
    checkpoint_path = str(policy_checkpoint_path(cfg, domain="crafter"))

    ppo_agent = PPO(
        state_dim,
        CRAFTER_ACTION_COUNT,
        ppo_cfg.lr_actor,
        ppo_cfg.lr_critic,
        ppo_cfg.gamma,
        ppo_cfg.K_epochs,
        ppo_cfg.eps_clip,
        ppo_cfg.has_continuous_action_space,
        ppo_cfg.action_std,
        entropy_coef=float(getattr(ppo_cfg, "entropy_coef", 0.01)),
        normalize_advantages=bool(getattr(ppo_cfg, "normalize_advantages", True)),
        normalize_returns=bool(getattr(ppo_cfg, "normalize_returns", False)),
        max_grad_norm=float(getattr(ppo_cfg, "max_grad_norm", 0.5)),
        minibatch_size=int(getattr(ppo_cfg, "minibatch_size", 0)),
    )
    use_wandb = bool(getattr(ppo_cfg, "use_wandb", False))
    sub_run = _init_crafter_wandb(cfg, "crafter_policy_training") if use_wandb else None
    step_penalty = float(getattr(ppo_cfg, "step_penalty", 0.0))
    time_step = 0
    episode_index = 0
    save_model_freq = int(ppo_cfg.save_model_freq)
    next_save_timestep = save_model_freq
    recent_rewards = deque(maxlen=int(getattr(ppo_cfg, "rolling_window_episodes", 50)))

    def policy_state(obs, unlocked_achievements):
        state = torch.as_tensor(
            np.transpose(obs["image"], (2, 0, 1)), device=device, dtype=torch.float32
        ).unsqueeze(0)
        inventory = torch.as_tensor(
            obs["inventory"], device=device, dtype=torch.float32
        ).unsqueeze(0)
        return build_policy_state(
            state, inventory, obs_norm_values, unlocked_achievements.unsqueeze(0)
        ).squeeze(0)

    obs, _ = env.reset()
    unlocked_mask = torch.zeros(
        CRAFTER_ACHIEVEMENT_MASK_DIM, device=device, dtype=torch.bool
    )
    coverage_tracker = _new_real_env_coverage_tracker(cfg)
    last_terminated = False
    while time_step < max_training_timesteps:
        episode_reward = 0.0
        episode_steps = 0
        terminated = False
        truncated = False
        episode_health_reward = 0.0
        episode_achievement_reward = 0.0
        episode_unlocks = set()
        episode_action_counts = np.zeros(CRAFTER_ACTION_COUNT, dtype=np.int64)

        while episode_steps < max_ep_len and time_step < max_training_timesteps:
            if coverage_tracker is not None:
                coverage_tracker.update(obs["image"], obs["inventory"])
            state = policy_state(obs, unlocked_mask)
            action, state_buf, action_buf, logprob_buf, value_buf = ppo_agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            reward = float(reward) + step_penalty
            done = bool(terminated or truncated)
            ppo_agent.save_buffer(
                state_buf, action_buf, logprob_buf, value_buf, reward, done
            )
            episode_reward += reward
            episode_health_reward += float(info.get("health_reward", 0.0))
            episode_achievement_reward += float(bool(info.get("newly_unlocked", [])))
            episode_unlocks.update(info.get("newly_unlocked", []))
            unlocked_mask |= crafter_achievement_mask_from_names(
                info.get("newly_unlocked", []), device_=device
            )
            episode_action_counts[int(action)] += 1
            episode_steps += 1
            time_step += 1
            obs = next_obs
            last_terminated = bool(terminated)

            if time_step % rollout_steps == 0 and ppo_agent.buffer.transition_count() >= 2:
                if terminated:
                    bootstrap = 0.0
                else:
                    # Bootstrap preprocessing is only needed at update
                    # boundaries; avoid rebuilding next_state every step.
                    bootstrap = ppo_agent.estimate_old_value(
                        policy_state(next_obs, unlocked_mask)
                    )
                ppo_agent.update(bootstrap_value=bootstrap)
            if time_step >= next_save_timestep:
                ppo_agent.save(checkpoint_path)
                _save_real_env_coverage(cfg, coverage_tracker)
                print(f"[Crafter Real PPO] checkpoint saved at timestep={time_step}: {checkpoint_path}")
                while next_save_timestep <= time_step:
                    next_save_timestep += save_model_freq
            if done:
                break

        episode_index += 1
        recent_rewards.append(episode_reward)
        unlock_names = sorted(episode_unlocks)
        final_inventory = info.get("inventory", {})
        print(
            f"[Crafter Real PPO] episode={episode_index} timestep={time_step} "
            f"steps={episode_steps} native_reward={episode_reward:.4f} "
            f"achievement_reward={episode_achievement_reward:.1f} "
            f"unlock_count={len(unlock_names)} "
            f"health_reward={episode_health_reward:.4f} "
            f"terminated={terminated} truncated={truncated} "
            f"unlocked={unlock_names} "
            f"final_status={{'health': {final_inventory.get('health', 0)}, "
            f"'food': {final_inventory.get('food', 0)}, "
            f"'drink': {final_inventory.get('drink', 0)}, "
            f"'energy': {final_inventory.get('energy', 0)}}}"
        )
        if sub_run is not None:
            episode_metrics = {
                "episode/reward": episode_reward,
                "episode/achievement_reward": episode_achievement_reward,
                "episode/unlock_count": float(len(unlock_names)),
                "episode/health_reward": episode_health_reward,
                "episode/steps": episode_steps,
                "episode/index": episode_index,
                "episode/terminated": float(terminated),
                "episode/truncated": float(truncated),
                "episode/final_health": float(final_inventory.get("health", 0)),
                "episode/final_food": float(final_inventory.get("food", 0)),
                "episode/final_drink": float(final_inventory.get("drink", 0)),
                "episode/final_energy": float(final_inventory.get("energy", 0)),
                "rolling/average_reward": sum(recent_rewards) / len(recent_rewards),
            }
            episode_metrics.update({
                f"achievement/{name}": float(name in episode_unlocks)
                for name in ACHIEVEMENT_NAMES
            })
            episode_metrics.update({
                f"action_fraction/{name}": float(episode_action_counts[index] / max(episode_steps, 1))
                for index, name in enumerate(CRAFTER_ACTION_NAMES)
            })
            sub_run.log(
                episode_metrics,
                step=time_step,
            )
        if terminated or truncated:
            obs, _ = env.reset()
            unlocked_mask.zero_()
    if ppo_agent.buffer.transition_count() >= 2:
        final_state = policy_state(obs, unlocked_mask)
        bootstrap = 0.0 if last_terminated else ppo_agent.estimate_old_value(final_state)
        ppo_agent.update(bootstrap_value=bootstrap)
    else:
        ppo_agent.buffer.clear()
    ppo_agent.save(checkpoint_path)
    _save_real_env_coverage(cfg, coverage_tracker)
    print(f"[Crafter Real PPO] Final policy saved: {checkpoint_path}")
    env.close()
    if sub_run is not None:
        sub_run.finish()


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(WM_PACKAGE_ROOT / "modelBased/config"),
    config_name="config",
)
def main(cfg: DictConfig):
    if bool(getattr(cfg.PPO, "train_in_real_env", False)):
        run_ppo_crafter_real(cfg)
    else:
        run_ppo_crafter_wm(cfg)


if __name__ == "__main__":
    main()
