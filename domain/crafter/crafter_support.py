import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from modelBased.world_model.crafter_dynamics import CRAFTER_CANONICAL_EVENT_CODEBOOK

PLAYER_ID = 13  # Updated: player ID in new CustomCrafterEnv mapping (was 10)

CRAFTER_INVENTORY_LABELS = (
    "Health", "Food", "Drink", "Energy",
    "Wood", "Stone", "Coal", "Iron", "Diamond", "Sapling",
    "Wood_Pickaxe", "Stone_Pickaxe", "Iron_Pickaxe",
    "Wood_Sword", "Stone_Sword", "Iron_Sword",
)


@dataclass(frozen=True)
class CrafterPlanningCheckpointSpec:
    """The complete Crafter WM contract required by imagined PPO rollouts."""

    data_type: str
    grid_shape: tuple[int, int, int]
    frame_stack: int
    attention_mask_size: int
    embed_dim: int
    num_heads: int
    output_mode: str
    inventory_output_mode: str
    inventory_classes: int
    inventory_value_mode: str
    inventory_architecture: str
    inventory_event_residual_enabled: bool
    inventory_event_residual_hidden_dims: tuple[int, ...]
    inventory_event_residual_action_embed_dim: int
    inventory_event_residual_change_bias: float
    pose_enabled: bool
    pose_mode: str
    predict_survival: bool
    obs_norm_values: tuple[float, ...]
    action_count: int


def _checkpoint_value(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read ordinary dicts and OmegaConf mappings without resolving config text."""
    try:
        return mapping.get(key, default)
    except AttributeError:
        return default


def _canonical_crafter_state_dict(raw_state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = {
        str(key)[6:] if str(key).startswith("model.") else str(key): value
        for key, value in raw_state.items()
    }
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Crafter checkpoint state_dict must contain tensor parameters")
    return state


def _single_head_width(state: Mapping[str, torch.Tensor], suffix: str) -> int:
    # Do not let ``crafter_inventory_action_embedding`` masquerade as the
    # shared ``action_embedding`` merely because their text suffix overlaps.
    widths = [int(value.shape[0]) for key, value in state.items()
              if (key == suffix or key.endswith("." + suffix)) and value.ndim == 2]
    if len(widths) != 1:
        raise ValueError(f"Crafter checkpoint must contain exactly one {suffix} tensor")
    return widths[0]


def _validate_inventory_architecture(state: Mapping[str, torch.Tensor], architecture: str) -> str:
    """Validate decoder keys so planning never guesses an inventory head."""
    architecture = str(architecture or "shared_pool_v1").lower()
    if architecture not in {"shared_pool_v1", "isolated_global_v1", "slot_attention_v1"}:
        raise ValueError(f"Unsupported Crafter inventory architecture {architecture!r}")
    if architecture == "shared_pool_v1":
        _single_head_width(state, "inv_head.2.weight")
    elif architecture == "isolated_global_v1":
        if _single_head_width(state, "crafter_inventory_head.2.weight") != 60:
            raise ValueError("isolated_global_v1 requires a 12x5 inventory head")
        if _single_head_width(state, "crafter_inventory_action_embedding.weight") != 17:
            raise ValueError("isolated_global_v1 requires a 17-action private embedding")
    else:
        if _single_head_width(state, "crafter_inventory_slot_head.2.weight") != 5:
            raise ValueError("slot_attention_v1 requires a 5-way per-slot head")
        if _single_head_width(state, "crafter_inventory_action_embedding.weight") != 17:
            raise ValueError("slot_attention_v1 requires a 17-action private embedding")
        required = {"crafter_inventory_value_embedding.weight", "crafter_inventory_slot_embedding"}
        if not required.issubset(state):
            raise ValueError("slot_attention_v1 checkpoint lacks inventory token parameters")
    return architecture


def _validate_event_residual(
    state: Mapping[str, torch.Tensor], inventory: Mapping[str, Any],
) -> tuple[bool, tuple[int, ...], int, float]:
    """Validate opt-in residual metadata and tensors; legacy is disabled."""
    residual = _checkpoint_value(inventory, "event_residual", {})
    enabled = bool(_checkpoint_value(residual, "enabled", False))
    if not enabled:
        if any(key.startswith("crafter_inventory_event_") for key in state):
            raise ValueError("Crafter checkpoint has event residual tensors but contract disables it")
        return False, (), 0, 0.0
    hidden = tuple(int(value) for value in _checkpoint_value(residual, "hidden_dims", ()))
    action_dim = int(_checkpoint_value(residual, "action_embed_dim", 0))
    bias = float(_checkpoint_value(residual, "change_bias", float("nan")))
    codebook = _checkpoint_value(residual, "codebook", None)
    if not hidden or any(value <= 0 for value in hidden) or action_dim <= 0 or not np.isfinite(bias):
        raise ValueError("Crafter event residual contract has invalid dimensions or change_bias")
    if not isinstance(codebook, list) or len(codebook) != 17 or any(len(row) != 12 for row in codebook):
        raise ValueError("Crafter event residual contract requires a [17,12] codebook")
    expected = torch.tensor(codebook, dtype=torch.long)
    if expected.tolist() != [list(row) for row in CRAFTER_CANONICAL_EVENT_CODEBOOK]:
        raise ValueError("Crafter event residual contract codebook is not canonical")
    buffer = state.get("crafter_inventory_event_codebook")
    if buffer is None or not torch.equal(buffer.long(), expected):
        raise ValueError("Crafter event residual codebook metadata does not match checkpoint")
    if _single_head_width(state, "crafter_inventory_event_action_embedding.weight") != 17:
        raise ValueError("Crafter event residual requires a 17-action embedding")
    final_weight = state.get("crafter_inventory_event_residual.%d.weight" % (2 * len(hidden)))
    if final_weight is None or tuple(final_weight.shape) != (17, hidden[-1]):
        raise ValueError("Crafter event residual final layer does not match contract")
    return True, hidden, action_dim, bias


def _infer_attention_mask_size(state: Mapping[str, torch.Tensor]) -> int:
    widths = [int(value.shape[1]) for key, value in state.items()
              if key.endswith("pos_embedding") and value.ndim == 3]
    if len(widths) != 1:
        raise ValueError("Crafter checkpoint must contain exactly one positional embedding")
    side = int(round(widths[0] ** 0.5))
    if side * side != widths[0]:
        raise ValueError(f"Crafter positional embedding has non-square token count {widths[0]}")
    return side


def _infer_embed_dim(state: Mapping[str, torch.Tensor]) -> int:
    tensors = [value for key, value in state.items()
               if key.endswith("pos_embedding") and value.ndim == 3]
    if len(tensors) != 1:
        raise ValueError("Crafter checkpoint must contain exactly one positional embedding")
    return int(tensors[0].shape[2])


def _crafter_contract_from_hparams(hparams: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Recover the pre-contract checkpoint format, validating every inferable field."""
    if str(_checkpoint_value(hparams, "env_type", "")) != "crafter":
        raise ValueError("Checkpoint metadata is not for env_type='crafter'")
    if str(_checkpoint_value(hparams, "data_type", "")) != "discrete":
        raise ValueError("Crafter planning requires data_type='discrete'")
    if str(_checkpoint_value(hparams, "model_type", "")).lower() != "attention":
        raise ValueError("Crafter imagined planning requires an Attention WM checkpoint")
    grid_shape = tuple(int(v) for v in _checkpoint_value(hparams, "grid_shape", ()))
    if len(grid_shape) != 3 or grid_shape[0] != 2:
        raise ValueError(f"Crafter checkpoint requires grid_shape=[2,H,W], got {grid_shape}")
    action_count = int(_checkpoint_value(hparams, "action_norm_values", 0))
    if action_count != 17 or _single_head_width(state, "action_embedding.weight") != 17:
        raise ValueError(f"Crafter planning requires 17 actions, got {action_count}")
    inventory_output_mode = str(
        _checkpoint_value(hparams, "crafter_inventory_output_mode", "categorical_gate")
    ).lower()
    value_mode = str(_checkpoint_value(hparams, "crafter_inventory_value_mode", "")).lower()
    inventory_architecture = _validate_inventory_architecture(
        state, _checkpoint_value(hparams, "crafter_inventory_architecture", "shared_pool_v1")
    )
    if inventory_architecture != "shared_pool_v1" and inventory_output_mode != "categorical_effect":
        raise ValueError(
            "Isolated Crafter inventory architectures require "
            "crafter_inventory_output_mode='categorical_effect'"
        )
    inventory_width = (
        _single_head_width(state, "inv_head.2.weight")
        if inventory_architecture == "shared_pool_v1" else 104
    )
    expected_width = {"categorical_absolute": 188, "categorical_delta": 116}
    if inventory_output_mode == "categorical_effect":
        valid_inventory = inventory_width == 104
    else:
        valid_inventory = value_mode in expected_width and inventory_width == expected_width[value_mode]
    if not valid_inventory:
        raise ValueError(
            "Crafter inventory metadata/head mismatch: "
            f"output_mode={inventory_output_mode!r}, value_mode={value_mode!r}, "
            f"output width={inventory_width}; expected 188 for categorical_absolute, "
            "116 for categorical_delta, or 104 for categorical_effect"
        )
    if inventory_architecture != "shared_pool_v1" and bool(_checkpoint_value(hparams, "predict_survival", True)):
        raise ValueError("Isolated Crafter inventory checkpoint requires predict_survival=false")
    pose_cfg = _checkpoint_value(hparams, "crafter_pose", {})
    pose_enabled = bool(_checkpoint_value(pose_cfg, "enabled", False))
    pose_mode = str(_checkpoint_value(pose_cfg, "mode", "learned_detached")).lower()
    pose_keys = [key for key in state if key.startswith("crafter_pose_head.")]
    if pose_enabled != bool(pose_keys):
        raise ValueError("Crafter pose metadata does not match checkpoint pose-head tensors")
    if pose_enabled and (pose_mode != "learned_detached" or _single_head_width(state, "crafter_pose_head.2.weight") != 20):
        raise ValueError("Crafter pose checkpoint must use a learned_detached 20-way pose head")
    output_mode = crafter_output_mode_from_state_dict(state)
    if output_mode != "effect":
        raise ValueError("Crafter imagined planning requires categorical effect map outputs")
    inferred_mask = _infer_attention_mask_size(state)
    inferred_embed_dim = _infer_embed_dim(state)
    if int(_checkpoint_value(hparams, "embed_dim", 0)) != inferred_embed_dim:
        raise ValueError("Crafter hparams embed_dim does not match positional embedding")
    saved_mask = _checkpoint_value(hparams, "attention_mask_size", None)
    if isinstance(saved_mask, (int, float)) and int(saved_mask) != inferred_mask:
        raise ValueError("Crafter hparams attention_mask_size does not match positional embedding")
    if "predict_survival" not in hparams:
        raise ValueError("Crafter legacy checkpoint lacks required predict_survival semantic metadata")
    return {
        "version": "crafter_planning_v1",
        "domain": "crafter",
        "data_type": str(_checkpoint_value(hparams, "data_type", "")),
        "grid_shape": list(grid_shape),
        "frame_stack": int(_checkpoint_value(hparams, "frame_stack", 1)),
        "attention_mask_size": inferred_mask,
        "embed_dim": inferred_embed_dim,
        "num_heads": int(_checkpoint_value(hparams, "num_heads", 0)),
        "output_mode": output_mode,
        "inventory": {
            "output_mode": inventory_output_mode, "classes": 10, "value_mode": value_mode,
            "architecture": inventory_architecture,
            "effect_values": [-4, -2, -1, 1] if inventory_output_mode == "categorical_effect" else None,
            "event_residual": {"enabled": False},
        },
        "pose": {"enabled": pose_enabled, "mode": pose_mode},
        "predict_survival": bool(_checkpoint_value(hparams, "predict_survival", True)),
        "obs_norm_values": list(_checkpoint_value(hparams, "obs_norm_values", ())),
        "action_count": action_count,
    }


def _spec_from_contract(contract: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> CrafterPlanningCheckpointSpec:
    if str(_checkpoint_value(contract, "domain", "")) != "crafter":
        raise ValueError("Checkpoint world_model_contract is not for Crafter")
    if "predict_survival" not in contract:
        raise ValueError("Crafter world_model_contract lacks predict_survival ownership")
    if str(_checkpoint_value(contract, "data_type", "")) != "discrete":
        raise ValueError("Crafter contract requires data_type='discrete'")
    inventory = _checkpoint_value(contract, "inventory", {})
    pose = _checkpoint_value(contract, "pose", {})
    inventory_output_mode = str(_checkpoint_value(inventory, "output_mode", ""))
    if inventory_output_mode not in {"categorical_gate", "categorical_effect"} or int(_checkpoint_value(inventory, "classes", 0)) != 10:
        raise ValueError("Crafter contract requires categorical_gate/effect inventory with 10 values")
    value_mode = str(_checkpoint_value(inventory, "value_mode", "")).lower()
    inventory_architecture = _validate_inventory_architecture(
        state, _checkpoint_value(inventory, "architecture", "shared_pool_v1")
    )
    residual_enabled, residual_hidden, residual_action_dim, residual_bias = _validate_event_residual(
        state, inventory
    )
    if inventory_architecture != "shared_pool_v1" and inventory_output_mode != "categorical_effect":
        raise ValueError(
            "Isolated Crafter inventory architectures require "
            "inventory.output_mode='categorical_effect'"
        )
    expected_width = {"categorical_absolute": 188, "categorical_delta": 116}
    if inventory_output_mode == "categorical_effect":
        if list(_checkpoint_value(inventory, "effect_values", ())) != [-4, -2, -1, 1]:
            raise ValueError("Crafter categorical_effect contract requires effect_values [-4, -2, -1, 1]")
        valid_inventory = (
            _single_head_width(state, "inv_head.2.weight") == 104
            if inventory_architecture == "shared_pool_v1" else True
        )
    else:
        valid_inventory = value_mode in expected_width and _single_head_width(state, "inv_head.2.weight") == expected_width[value_mode]
    if not valid_inventory:
        raise ValueError("Crafter world_model_contract inventory mode does not match inv_head shape")
    if inventory_architecture != "shared_pool_v1" and bool(_checkpoint_value(contract, "predict_survival", True)):
        raise ValueError("Isolated Crafter inventory contract requires predict_survival=false")
    pose_enabled = bool(_checkpoint_value(pose, "enabled", False))
    pose_keys = [key for key in state if key.startswith("crafter_pose_head.")]
    if pose_enabled != bool(pose_keys):
        raise ValueError("Crafter world_model_contract pose metadata does not match pose-head tensors")
    if pose_enabled and (_checkpoint_value(pose, "mode", "") != "learned_detached" or _single_head_width(state, "crafter_pose_head.2.weight") != 20):
        raise ValueError("Crafter pose contract requires learned_detached 20-way head")
    output_mode = str(_checkpoint_value(contract, "output_mode", ""))
    if output_mode != crafter_output_mode_from_state_dict(state) or output_mode != "effect":
        raise ValueError("Crafter world_model_contract output mode does not match fc head")
    action_count = int(_checkpoint_value(contract, "action_count", 0))
    if action_count != 17 or _single_head_width(state, "action_embedding.weight") != 17:
        raise ValueError("Crafter planning requires a 17-action checkpoint")
    grid_shape = tuple(int(v) for v in _checkpoint_value(contract, "grid_shape", ()))
    if len(grid_shape) != 3 or grid_shape[0] != 2:
        raise ValueError("Crafter contract requires grid_shape=[2,H,W]")
    mask_size = int(_checkpoint_value(contract, "attention_mask_size", 0))
    if mask_size != _infer_attention_mask_size(state):
        raise ValueError("Crafter contract attention_mask_size does not match positional embedding")
    obs_norm_values = tuple(float(v) for v in _checkpoint_value(contract, "obs_norm_values", ()))
    if len(obs_norm_values) < 2:
        raise ValueError("Crafter contract must provide at least two observation normalizers")
    embed_dim = int(_checkpoint_value(contract, "embed_dim", 0))
    num_heads = int(_checkpoint_value(contract, "num_heads", 0))
    if embed_dim != _infer_embed_dim(state) or embed_dim < 8 or num_heads < 1 or embed_dim % num_heads:
        raise ValueError("Crafter contract has invalid embed_dim/num_heads")
    return CrafterPlanningCheckpointSpec(
        data_type=str(_checkpoint_value(contract, "data_type", "")), grid_shape=grid_shape,
        frame_stack=int(_checkpoint_value(contract, "frame_stack", 1)), attention_mask_size=mask_size,
        embed_dim=embed_dim, num_heads=num_heads,
        output_mode=output_mode, inventory_output_mode=str(_checkpoint_value(inventory, "output_mode", "")),
        inventory_classes=int(_checkpoint_value(inventory, "classes", 0)), inventory_value_mode=value_mode,
        inventory_architecture=inventory_architecture,
        inventory_event_residual_enabled=residual_enabled,
        inventory_event_residual_hidden_dims=residual_hidden,
        inventory_event_residual_action_embed_dim=residual_action_dim,
        inventory_event_residual_change_bias=residual_bias,
        pose_enabled=pose_enabled, pose_mode=str(_checkpoint_value(pose, "mode", "")),
        predict_survival=bool(_checkpoint_value(contract, "predict_survival", True)), obs_norm_values=obs_norm_values,
        action_count=action_count,
    )


def inspect_crafter_planning_checkpoint(path: str | Path) -> CrafterPlanningCheckpointSpec:
    """Inspect a semantic Crafter checkpoint before constructing planning WM."""
    raw = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or "state_dict" not in raw:
        raise ValueError("Crafter planning refuses a pure state_dict without semantic metadata")
    state = _canonical_crafter_state_dict(raw["state_dict"])
    contract = raw.get("world_model_contract")
    if contract is None:
        hparams = raw.get("hyper_parameters")
        if hparams is None:
            raise ValueError("Crafter planning checkpoint lacks world_model_contract and Lightning hyper_parameters")
        contract = _crafter_contract_from_hparams(hparams, state)
    return _spec_from_contract(contract, state)


def load_crafter_planning_model(path: str | Path):
    """Strictly load the Attention WM selected by its checkpoint contract."""
    spec = inspect_crafter_planning_checkpoint(path)
    from modelBased.world_model.AttentionWM_support import AttentionModule

    model = AttentionModule(
        spec.data_type, spec.grid_shape, spec.attention_mask_size, spec.embed_dim, spec.num_heads,
        env_type="crafter", frame_stack=spec.frame_stack, crafter_output_mode=spec.output_mode,
        crafter_inventory_classes=spec.inventory_classes,
        crafter_inventory_output_mode=spec.inventory_output_mode,
        crafter_inventory_value_mode=spec.inventory_value_mode,
        crafter_inventory_architecture=spec.inventory_architecture,
        crafter_pose_enabled=spec.pose_enabled,
        crafter_inventory_event_residual_enabled=spec.inventory_event_residual_enabled,
        crafter_inventory_event_residual_hidden_dims=spec.inventory_event_residual_hidden_dims or (64,),
        crafter_inventory_event_residual_action_embed_dim=(
            spec.inventory_event_residual_action_embed_dim or 8
        ),
        crafter_inventory_event_residual_change_bias=spec.inventory_event_residual_change_bias,
    )
    raw = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    state = _canonical_crafter_state_dict(raw["state_dict"])
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, spec


def _to_nchw(obs: np.ndarray) -> np.ndarray:
    """Convert observations to (N, C, H, W)."""
    if obs.ndim != 4:
        raise ValueError(f"Expected 4D observations, got shape {obs.shape}")

    # Already (N, C, H, W)
    # Check if channel is obviously dim 1 (e.g. 2 or 3 channels vs H/W > 3, or explicitly fewer channels than width)
    if obs.shape[1] <= 3 or obs.shape[1] < obs.shape[-1]:
        return obs

    # Usually Crafter saved as (N, H, W, C)
    if obs.shape[-1] <= 8:
        return np.moveaxis(obs, -1, 1)

    raise ValueError(f"Cannot infer channel axis for shape {obs.shape}")

def interpret_env(terrain_map, cfg, inventory_vec=None):
    import torch
    import numpy as np
    
    # Handle inputs
    if isinstance(terrain_map, torch.Tensor):
        phys_map = terrain_map.cpu().numpy()
    else:
        phys_map = terrain_map

    # Handle shape (C, H, W) -> (H, W)
    if phys_map.ndim == 3:
        phys_map = phys_map[0]

    H_phys, W = phys_map.shape

    # Handle inventory vector (16 items)
    if inventory_vec is None:
        inventory_vec = np.zeros(16)
    elif isinstance(inventory_vec, torch.Tensor):
        inventory_vec = inventory_vec.cpu().numpy().flatten()
    
    # Ensure inventory is long enough
    if len(inventory_vec) < 16:
        pad = np.zeros(16 - len(inventory_vec))
        inventory_vec = np.concatenate([inventory_vec, pad])

    # Define mapping directly consistent with CustomCrafterEnv (0-19)
    # 0=None, 1=water, 2=grass, 3=stone, 4=path, 5=sand, 6=tree, 7=lava, 8=coal, 9=iron, 10=diamond, 11=table, 12=furnace, 
    # 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
    inv_obj_map = {
        0: 'none', 1: 'water', 2: 'grass', 3: 'stone', 4: 'path', 
        5: 'sand', 6: 'tree', 7: 'lava', 8: 'coal', 9: 'iron', 
        10: 'diamond', 11: 'table', 12: 'furnace', 13: 'agent',
        14: 'cow', 15: 'zombie', 16: 'skeleton', 17: 'arrow', 18: 'plant', 19: 'fence'
    }
    
    map_elem = cfg.training_generator.get('map_element_crafter', {
        "grass": "G", "water": "W", "tree": "T", "stone": "R", "coal": "C",
        "iron": "I", "lava": "L", "zombie": "Z", "table": "X", "furnace": "U", "agent": "A", "diamond": "O",
        "cow": "M", "skeleton": "K", "plant": "t", "fence": "F", "path": "P", "sand": "S"
    })
    
    lines = []
    for r in range(H_phys):
        row_str = ""
        for c in range(W):
            obj_idx = phys_map[r, c]
            obj_name = inv_obj_map.get(int(obj_idx), 'grass')
            char = map_elem.get(obj_name, 'G')
            row_str += char
        lines.append(row_str)
    
    layout_str = "\n".join(lines).strip()

    # Dynamic parsing of the inventory vector (16 items)
    # Define the 16-slot mapping
    CRAFTER_INV_SLOTS = [
        "health", "food", "drink", "energy",
        "wood", "stone", "coal", "iron", "diamond", "sapling",
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
        "wood_sword", "stone_sword", "iron_sword",
    ]
    
    stats_str = "\n\n# --- Initial Stats ---\n"
    for i, item_name in enumerate(CRAFTER_INV_SLOTS):
        if i < len(inventory_vec):
            val = int(inventory_vec[i])
            # For tools/swords, clamp to 1 if > 0 to maintain game balance
            if "pickaxe" in item_name or "sword" in item_name:
                val = 1 if val > 0 else 0
            # For health/food, if 0, default to 9
            if item_name in ["health", "food"] and val == 0:
                val = 9
            stats_str += f"{item_name}: {val}\n"
    
    final_env_source = layout_str + stats_str
    return final_env_source, ""



def extract_player_positions(obs: np.ndarray, player_id: int = PLAYER_ID) -> np.ndarray:
    """
    Return array of (y, x) positions for each frame.
    If player not found in a frame, returns (-1, -1) for that frame.
    """
    obs_nchw = _to_nchw(obs)
    obj_map = obs_nchw[:, 0, :, :]  # object-id channel

    positions = np.full((obj_map.shape[0], 2), -1, dtype=np.int32)
    for i in range(obj_map.shape[0]):
        hits = np.argwhere(obj_map[i] == player_id)
        if len(hits) > 0:
            positions[i] = hits[0]  # (y, x)

    return positions


class CrafterCoverageTracker:
    """Memory-bounded position and inventory coverage for online rollouts."""

    def __init__(self, player_id: int = PLAYER_ID):
        self.player_id = int(player_id)
        self.heatmap = None
        self.inventory_positive_counts = np.zeros(
            len(CRAFTER_INVENTORY_LABELS), dtype=np.int64
        )
        self.position_samples = 0
        self.inventory_samples = 0
        # Primary coverage tracks progression inventory only; physiology is a
        # separate survival signal and must not create artificial coverage.
        self.inventory_states: set[bytes] = set()
        self.inventory_state_history: list[int] = []
        self.inventory_gain_counts = np.zeros(
            len(CRAFTER_INVENTORY_LABELS), dtype=np.int64
        )

    def update(self, observations, inventories) -> None:
        observations = np.asarray(observations)
        if observations.ndim == 3:
            observations = observations[None, ...]
        obs_nchw = _to_nchw(observations)
        height, width = obs_nchw.shape[2:]
        if self.heatmap is None:
            self.heatmap = np.zeros((height, width), dtype=np.int64)
        elif self.heatmap.shape != (height, width):
            raise ValueError(
                f"Crafter coverage map changed from {self.heatmap.shape} "
                f"to {(height, width)}"
            )

        positions = extract_player_positions(obs_nchw, player_id=self.player_id)
        valid = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < height)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < width)
        )
        np.add.at(
            self.heatmap,
            (positions[valid, 0], positions[valid, 1]),
            1,
        )
        self.position_samples += int(len(positions))

        inventories = np.asarray(inventories)
        if inventories.ndim == 1:
            inventories = inventories[None, ...]
        inventories = inventories.reshape(inventories.shape[0], -1)
        expected_slots = len(CRAFTER_INVENTORY_LABELS)
        if inventories.shape[1] != expected_slots:
            raise ValueError(
                f"Crafter coverage expects {expected_slots} inventory slots, "
                f"got shape {inventories.shape}"
            )
        if inventories.shape[0] != obs_nchw.shape[0]:
            raise ValueError(
                "Crafter observation and inventory batch sizes differ: "
                f"{obs_nchw.shape[0]} != {inventories.shape[0]}"
            )
        self.inventory_positive_counts += np.count_nonzero(
            inventories > 0, axis=0
        )
        self.inventory_samples += int(inventories.shape[0])
        discrete = np.rint(inventories).astype(np.int16)
        for row in discrete[:, 4:]:
            self.inventory_states.add(row.tobytes())
        self.inventory_state_history.append(len(self.inventory_states))

    @property
    def inventory_presence_rate(self) -> np.ndarray:
        if self.inventory_samples == 0:
            return np.zeros(len(CRAFTER_INVENTORY_LABELS), dtype=np.float64)
        return self.inventory_positive_counts / self.inventory_samples * 100.0

    @property
    def unique_inventory_states(self) -> int:
        return len(self.inventory_states)

    def save(self, save_path: str | Path, title: str) -> Path:
        if self.inventory_samples == 0:
            raise ValueError("Cannot save Crafter coverage before recording observations")

        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_states, ax_inventory) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.35]}
        )
        history = self.inventory_state_history or [self.unique_inventory_states]
        ax_states.plot(np.arange(1, len(history) + 1), history,
                       color="#2c7fb8", linewidth=2)
        ax_states.set_title(f"Unique inventory states: {self.unique_inventory_states}")
        ax_states.set_xlabel("Collected observations")
        ax_states.set_ylabel("Cumulative unique states")
        ax_states.grid(axis="both", linestyle="--", alpha=0.3)
        rates = self.inventory_presence_rate[4:]
        colors = ["#2ecc71"] * 6 + ["#e74c3c"] * 3 + ["#f39c12"] * 3
        ax_inventory.bar(
            range(len(CRAFTER_INVENTORY_LABELS) - 4),
            rates,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax_inventory.set_title("Progression Inventory Coverage / Presence Rate")
        ax_inventory.set_ylabel("Presence Rate (%)")
        ax_inventory.set_ylim(0, 115)
        ax_inventory.set_xticks(range(len(CRAFTER_INVENTORY_LABELS) - 4))
        ax_inventory.set_xticklabels(
            CRAFTER_INVENTORY_LABELS[4:], rotation=45, ha="right", fontsize=9
        )
        ax_inventory.grid(axis="y", linestyle="--", alpha=0.3)
        for index, rate in enumerate(rates):
            color = "darkred" if rate >= 95 else "black"
            ax_inventory.text(
                index, rate + 2, f"{rate:.1f}", ha="center", fontsize=8,
                fontweight="bold", color=color,
            )

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Crafter real-env coverage saved to {out}")
        return out


def plot_crafter_coverage_from_npz(data_path: str, save_path: str | None = None, title: str = "Crafter Exploration Coverage"):
    data = np.load(data_path, allow_pickle=True)

    if "a" not in data:
        raise KeyError(f"Dataset {data_path} has no key 'a' for observations")

    obs = data["a"]
    obs_nchw = _to_nchw(obs)
    h, w = obs_nchw.shape[2], obs_nchw.shape[3]

    positions = extract_player_positions(obs)
    heatmap = np.zeros((h, w), dtype=np.int64)

    for y, x in positions:
        # Ignore boundary water (outermost row/column) for cleaner visualization
        if 1 <= y < h - 1 and 1 <= x < w - 1:
            heatmap[y, x] += 1

    plt.figure(figsize=(7, 7))
    plt.imshow(heatmap, cmap="viridis", origin="upper")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Visit Count")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Coverage saved to {out}")
    else:
        plt.show()

    plt.close()


# ============================================================
#  World Model Support Functions for Crafter (Categorical Effects)
# ============================================================
import torch.nn.functional as F

from modelBased.world_model.crafter_dynamics import (
    apply_categorical_effect,
    balanced_categorical_effect_loss,
    categorical_effect_target,
)

# CustomCrafterEnv Object IDs (matching original Crafter mat_ids + shifted entities):
# 0=None/empty, 1=water, 2=grass, 3=stone, 4=path, 5=sand, 6=tree,
# 7=lava, 8=coal, 9=iron, 10=diamond, 11=table, 12=furnace,
# 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
# Total: 20 classes (0-19)
# Direction IDs: 0=none, 1=up, 2=down, 3=left, 4=right  →  5 classes
CRAFTER_OBJ_CLASSES = 20
CRAFTER_DIR_CLASSES = 5
CRAFTER_OBJ_EFFECT_CLASSES = CRAFTER_OBJ_CLASSES + 1
CRAFTER_DIR_EFFECT_CLASSES = CRAFTER_DIR_CLASSES + 1
CRAFTER_EFFECT_CHANNELS = CRAFTER_OBJ_EFFECT_CLASSES + CRAFTER_DIR_EFFECT_CLASSES
CRAFTER_ABSOLUTE_CHANNELS = CRAFTER_OBJ_CLASSES + CRAFTER_DIR_CLASSES
CRAFTER_ACTION_NAMES = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep',
    'place_stone', 'place_table', 'place_furnace', 'place_plant',
    'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
    'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
]
_CLAMP_DEBUG_PRINTED = False


def crafter_clamp_targets(obj_true: torch.Tensor, dir_true: torch.Tensor):
    """
    Clamp object and direction targets to valid class ranges.
    Prints a one-time debug message if out-of-range values are detected.
    """
    global _CLAMP_DEBUG_PRINTED
    if not _CLAMP_DEBUG_PRINTED:
        obj_max = int(obj_true.max().item())
        dir_max = int(dir_true.max().item())
        obj_min = int(obj_true.min().item())
        dir_min = int(dir_true.min().item())
        if obj_max >= CRAFTER_OBJ_CLASSES or dir_max >= CRAFTER_DIR_CLASSES or obj_min < 0 or dir_min < 0:
            print(f"[CrafterClamp] WARNING: out-of-range targets detected! "
                  f"obj=[{obj_min},{obj_max}] (valid 0-{CRAFTER_OBJ_CLASSES-1}), "
                  f"dir=[{dir_min},{dir_max}] (valid 0-{CRAFTER_DIR_CLASSES-1}). Clamping.")
        else:
            print(f"[CrafterClamp] Targets in range: obj=[{obj_min},{obj_max}], dir=[{dir_min},{dir_max}]")
        _CLAMP_DEBUG_PRINTED = True
    obj_true = obj_true.clamp(0, CRAFTER_OBJ_CLASSES - 1)
    dir_true = dir_true.clamp(0, CRAFTER_DIR_CLASSES - 1)
    return obj_true, dir_true


def crafter_classification_loss(
    next_pred: torch.Tensor,
    next_true: torch.Tensor,
    reduction: str = "none",
    weighted: bool = False,
) -> torch.Tensor:
    """
    Compute per-location CrossEntropy loss for Crafter classification output.

    Args:
        next_pred: (B, 25, H, W) logits — first 20 = obj, last 5 = dir
        next_true: (B, 2, H, W)  ground truth — ch0=obj ID, ch1=dir ID
        reduction: 'none' => (B,H,W), 'mean' => scalar
        weighted: If True, use tiered class weighting (Minigrid-style)
    """
    obj_logits = next_pred[:, :CRAFTER_OBJ_CLASSES]   # (B, 20, H, W)
    dir_logits = next_pred[:, CRAFTER_OBJ_CLASSES:]   # (B, 5, H, W)

    obj_true = next_true[:, 0].long().to(obj_logits.device, non_blocking=True)
    dir_true = next_true[:, 1].long().to(dir_logits.device, non_blocking=True)

    obj_true, dir_true = crafter_clamp_targets(obj_true, dir_true)

    # --- Optional tiered weighting logic ---
    # obj_weights = None
    # if weighted:
    #     # Tiered Weights Map:
    #     # Grass=0.5, Standard=1.0, Tools/Animals=10.0, Progress=25.0, HolyGrail=50.0
    #     w = torch.ones(CRAFTER_OBJ_CLASSES, device=next_pred.device)
    #     w[2] = 0.5   # Grass (Suppress common background)
    #     w[8] = 2.0  # Coal
    #     w[11] = 2.0 # Table
    #     w[12] = 2.0 # Furnace
    #     w[14:20] = 3.0 # Mobs (Cow, Zombie, Skeleton, etc)
    #     w[9] = 3.0  # Iron
    #     w[13] = 4.0 # Player
    #     w[10] = 5.0 # Diamond (Highest priority)
    #     obj_weights = w.to(obj_logits.device, non_blocking=True)

    # # Add label_smoothing=0.1 to prevent model from being overconfident
    # loss_obj = F.cross_entropy(obj_logits, obj_true, weight=obj_weights, reduction=reduction, label_smoothing=0.1)
    loss_obj = F.cross_entropy(obj_logits, obj_true, reduction=reduction, label_smoothing=0.1)
    loss_dir = F.cross_entropy(dir_logits, dir_true, reduction=reduction, label_smoothing=0.1)

    total = loss_obj + loss_dir  # (B, H, W) or scalar

    return total


def crafter_effect_loss(
    effect_logits: torch.Tensor,
    current: torch.Tensor,
    following: torch.Tensor,
    *,
    reduction: str = "balanced_mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Loss for Crafter KEEP/SET_TO effects.

    The object and direction fields are balanced independently so their
    unchanged cells cannot drown out sparse transitions.
    """
    if effect_logits.shape[1] != CRAFTER_EFFECT_CHANNELS:
        raise ValueError(
            f"Crafter effect logits require {CRAFTER_EFFECT_CHANNELS} channels, "
            f"got {effect_logits.shape[1]}"
        )
    obj_logits = effect_logits[:, :CRAFTER_OBJ_EFFECT_CLASSES]
    dir_logits = effect_logits[:, CRAFTER_OBJ_EFFECT_CLASSES:]
    obj_target = categorical_effect_target(current[:, 0], following[:, 0])
    dir_target = categorical_effect_target(current[:, 1], following[:, 1])
    obj_loss = balanced_categorical_effect_loss(
        obj_logits,
        obj_target,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    dir_loss = balanced_categorical_effect_loss(
        dir_logits,
        dir_target,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    return (obj_loss + dir_loss) / 2.0


def crafter_reconstruct_from_logits(
    prediction: torch.Tensor,
    current: torch.Tensor | None = None,
    *,
    constrain_agent: bool = False,
    suppress_agent: bool = False,
) -> torch.Tensor:
    """
    Reconstruct Crafter channels from effect logits.

    Legacy 25-channel absolute checkpoints remain decodable when ``current``
    is omitted or supplied, which lets an already-running old pipeline finish
    without being confused with the new 27-channel effect artifacts.
    """
    channels = int(prediction.shape[1])
    if constrain_agent and suppress_agent:
        raise ValueError("Crafter decode cannot both constrain and suppress the player")
    if channels == CRAFTER_ABSOLUTE_CHANNELS:
        obj_logits = prediction[:, :CRAFTER_OBJ_CLASSES]
        obj_pred = obj_logits.argmax(dim=1)
        if constrain_agent:
            obj_pred = _decode_crafter_object_with_single_agent(
                obj_logits, current_obj=None
            )
        elif suppress_agent:
            obj_pred = _decode_crafter_object_without_agent(obj_logits, current_obj=None)
        dir_pred = prediction[:, CRAFTER_OBJ_CLASSES:].argmax(dim=1)
        return torch.stack([obj_pred, dir_pred], dim=1).float()
    if channels != CRAFTER_EFFECT_CHANNELS:
        raise ValueError(
            "Unsupported Crafter prediction width: "
            f"{channels}; expected {CRAFTER_EFFECT_CHANNELS} effect or "
            f"{CRAFTER_ABSOLUTE_CHANNELS} legacy absolute channels"
        )
    if current is None:
        raise ValueError("Current Crafter state is required to apply categorical effects")
    obj_effect_logits = prediction[:, :CRAFTER_OBJ_EFFECT_CLASSES]
    obj_effect = obj_effect_logits.argmax(dim=1)
    dir_effect = prediction[:, CRAFTER_OBJ_EFFECT_CLASSES:].argmax(dim=1)
    current_obj = current[:, 0]
    if constrain_agent:
        obj_pred = _decode_crafter_object_with_single_agent(
            obj_effect_logits, current_obj=current_obj
        )
    elif suppress_agent:
        obj_pred = _decode_crafter_object_without_agent(
            obj_effect_logits, current_obj=current_obj
        )
    else:
        obj_pred = apply_categorical_effect(current_obj, obj_effect)
    dir_pred = apply_categorical_effect(current[:, 1], dir_effect)
    return torch.stack([obj_pred, dir_pred], dim=1).float()


def _crafter_next_class_probabilities(
    logits: torch.Tensor,
    current_obj: torch.Tensor | None,
    classes: int,
) -> torch.Tensor:
    """Map effect logits to absolute next-object probabilities.

    Effect class zero means KEEP, so its probability mass is added to the
    current object class before applying any state invariant projection.
    """
    probabilities = logits.softmax(dim=1)
    if current_obj is None:
        return probabilities
    current_obj = current_obj.long().clamp(0, classes - 1)
    next_probabilities = probabilities[:, 1 : classes + 1].clone()
    next_probabilities.scatter_add_(
        1, current_obj.unsqueeze(1), probabilities[:, 0:1]
    )
    return next_probabilities


def _decode_crafter_object_with_single_agent(
    logits: torch.Tensor,
    current_obj: torch.Tensor | None,
) -> torch.Tensor:
    """Decode object probabilities with the Crafter one-player invariant."""
    is_effect = logits.shape[1] == CRAFTER_OBJ_EFFECT_CLASSES
    if not is_effect and logits.shape[1] != CRAFTER_OBJ_CLASSES:
        raise ValueError(
            "Crafter object logits must have 20 absolute or 21 effect classes, "
            f"got {logits.shape[1]}"
        )
    probabilities = _crafter_next_class_probabilities(
        logits, current_obj if is_effect else None, CRAFTER_OBJ_CLASSES
    )
    flat = probabilities.flatten(2)
    agent_probability = flat[:, PLAYER_ID]
    nonagent = flat.clone()
    nonagent[:, PLAYER_ID] = 0.0
    best_nonagent_probability, _ = nonagent.max(dim=1)
    choice_score = torch.log(agent_probability.clamp_min(1e-12)) - torch.log(
        best_nonagent_probability.clamp_min(1e-12)
    )
    chosen = choice_score.argmax(dim=1)
    decoded = nonagent.argmax(dim=1)
    decoded.scatter_(1, chosen.unsqueeze(1), torch.full_like(
        chosen.unsqueeze(1), PLAYER_ID
    ))
    return decoded.reshape_as(probabilities[:, 0])


def _decode_crafter_object_without_agent(
    logits: torch.Tensor,
    current_obj: torch.Tensor | None,
) -> torch.Tensor:
    """Decode each cell's strongest non-player next class.

    Used only by the learned-pose planner path: its separate pose readout
    writes the one player afterwards, while this preserves the spatial WM's
    best terrain/object prediction under every candidate player location.
    """
    is_effect = logits.shape[1] == CRAFTER_OBJ_EFFECT_CLASSES
    if not is_effect and logits.shape[1] != CRAFTER_OBJ_CLASSES:
        raise ValueError(
            "Crafter object logits must have 20 absolute or 21 effect classes, "
            f"got {logits.shape[1]}"
        )
    probabilities = _crafter_next_class_probabilities(
        logits, current_obj if is_effect else None, CRAFTER_OBJ_CLASSES
    ).clone()
    probabilities[:, PLAYER_ID] = 0.0
    return probabilities.argmax(dim=1)


def crafter_output_mode_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Infer legacy absolute versus categorical-effect output from FC width."""
    for key, value in state_dict.items():
        if key.endswith("fc.weight") and value.ndim == 2:
            width = int(value.shape[0])
            if width == CRAFTER_EFFECT_CHANNELS:
                return "effect"
            if width == CRAFTER_ABSOLUTE_CHANNELS:
                return "absolute"
    raise ValueError("Checkpoint does not contain a recognized Crafter fc.weight")
    
def visualize_crafter_wm(
    obs_masked: torch.Tensor, 
    obs_next_masked: torch.Tensor, 
    obs_pred_logits: torch.Tensor,
    action: int,
    step: int,
    save_dir: str | None = None,
    full_map_size: tuple = (64, 64),
    agent_pos: tuple = (32, 32),
    inv: np.ndarray = None,
    inv_next: np.ndarray = None
):
    """
    Visualize WM prediction for Crafter with 5 panels.
    """
    import os
    if save_dir is None:
        wm_root = Path(os.environ.get("WM_ROOT", Path(__file__).resolve().parents[2]))
        save_dir = str(wm_root / "outputs" / "visualizations" / "world_model")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle batch dimension if present
    if obs_masked.ndim == 4: obs_masked = obs_masked[0]
    if obs_next_masked.ndim == 4: obs_next_masked = obs_next_masked[0]
    if obs_pred_logits.ndim == 4: obs_pred_logits = obs_pred_logits[0]
        
    # 1. Reconstruct the next object map. New checkpoints emit effects; keep
    # legacy absolute logits readable for old artifacts and visual comparisons.
    if obs_pred_logits.shape[0] == CRAFTER_EFFECT_CHANNELS:
        obj_logits = obs_pred_logits[:CRAFTER_OBJ_EFFECT_CLASSES]
        obj_probs = F.softmax(obj_logits, dim=0)
        obj_effect = obj_probs.argmax(dim=0)
        obj_pred = apply_categorical_effect(
            obs_masked[0], obj_effect
        ).detach().cpu().numpy()
    elif obs_pred_logits.shape[0] == CRAFTER_ABSOLUTE_CHANNELS:
        obj_logits = obs_pred_logits[:CRAFTER_OBJ_CLASSES]
        obj_probs = F.softmax(obj_logits, dim=0)
        obj_pred = obj_probs.argmax(dim=0).detach().cpu().numpy()
    else:
        raise ValueError(
            f"Unsupported Crafter visualization output width {obs_pred_logits.shape[0]}"
        )
    confidence = obj_probs.max(dim=0)[0].detach().cpu().numpy()
    uncertainty = 1.0 - confidence
    
    # 2. Read the ground-truth observation.
    obj_curr = obs_masked[0].detach().cpu().numpy()
    obj_next = obs_next_masked[0].detach().cpu().numpy()
    
    # 3. Compute the error map before cropping.
    error_map = (obj_pred != obj_next).astype(np.float32)

    # --- Crop precisely using the map size and agent position ---
    mask_size = obs_masked.shape[-1]
    half = mask_size // 2
    # Ensure integer coordinates.
    ay, ax = int(agent_pos[0]), int(agent_pos[1])
    H, W = int(full_map_size[0]), int(full_map_size[1])
    
    y_start = half - ay
    y_end = y_start + H
    x_start = half - ax
    x_end = x_start + W
    
    # Keep slice bounds within the valid region.
    y_start, y_end = max(0, y_start), min(mask_size, y_end)
    x_start, x_end = max(0, x_start), min(mask_size, x_end)

    obj_curr = obj_curr[y_start:y_end, x_start:x_end]
    obj_pred = obj_pred[y_start:y_end, x_start:x_end]
    obj_next = obj_next[y_start:y_end, x_start:x_end]
    uncertainty = uncertainty[y_start:y_end, x_start:x_end]
    error_map = error_map[y_start:y_end, x_start:x_end]
    
    h_crop, w_crop = obj_curr.shape
        
    # --- Define an intuitive Crafter color palette ---
    # 0=Empty, 1=Water, 2=Grass, 3=Stone, 4=Path, 5=Sand, 6=Tree, 7=Lava, 8=Coal, 9=Iron, 10=Diamond, 11=Table, 12=Furnace
    # 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
    from matplotlib.colors import ListedColormap
    colors = [
        '#000000', # 0: None (Black)
        '#1E90FF', # 1: Water (Dodger Blue)
        '#32CD32', # 2: Grass (Lime Green)
        '#888888', # 3: Stone (Grey)
        '#964B00', # 4: Path (Brown)
        '#FFFFBB', # 5: Sand (Yellowish)
        '#006400', # 6: Tree (Dark Green)
        '#FFA500', # 7: Lava (Orange-Yellow)
        '#333333', # 8: Coal (Grey/Black)
        '#CCCCCC', # 9: Iron (Light Grey)
        '#00FFFF', # 10: Diamond (Cyan)
        '#774400', # 11: Table (Wood)
        '#331100', # 12: Furnace (Darker)
        '#FF0000', # 13: Player (Red)
        '#FFFFFF', # 14: Cow (White)
        '#9400D3', # 15: Zombie (Dark Violet)
        '#EEEEEE', # 16: Skeleton (Bone)
        '#FFFF00', # 17: Arrow (Yellow)
        '#ADFF2F', # 18: Plant (Green Yellow)
        '#442200', # 19: Fence (Wood)
    ]
    # Extend the palette if the class count exceeds the predefined colors.
    while len(colors) < CRAFTER_OBJ_CLASSES: colors.append('#333333')
    cmap = ListedColormap(colors)
    
    # Create a 2x3 panel layout.
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    # Panel 0: Current State
    axes[0].imshow(obj_curr, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[0].set_title(f"Current Observed (T)")
    
    # Panel 1: Prediction (Argmax)
    act_name = CRAFTER_ACTION_NAMES[action] if action < len(CRAFTER_ACTION_NAMES) else f"act_{action}"
    axes[1].imshow(obj_pred, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[1].set_title(f"WM Prediction (T+1) | {act_name}")
    
    # Panel 2: Ground Truth Next
    axes[2].imshow(obj_next, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[2].set_title(f"True Observed (T+1)")
    
    # Panel 3: Error Map (Correctness)
    # Use a binary map to highlight incorrect vs. correct predictions.
    im3 = axes[3].imshow(error_map, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(f"Error Map (Yellow=Wrong)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Panel 4: Uncertainty Heatmap
    im4 = axes[4].imshow(uncertainty, cmap="hot", vmin=0, vmax=1.0)
    axes[4].set_title(f"Uncertainty (Entropy/Confidence)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    # Panel 5: Legend & Detail Stats (Text)
    axes[5].axis('off')
    mean_uncer = uncertainty.mean()
    total_errors = int(error_map.sum())
    acc = (1.0 - (total_errors / (h_crop * w_crop))) * 100 if (h_crop * w_crop) > 0 else 0
    
    inv_text = ""
    if inv is not None:
        inv_labels = ['health', 'food', 'drink', 'energy', 'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'w_pick', 's_pick', 'i_pick', 'w_sword', 's_sword', 'i_sword']
        inv_text = "Inventory (Now -> Next):\n"
        for i, val in enumerate(inv):
            if i >= len(inv_labels): break
            next_val = inv_next[i] if inv_next is not None else val
            if val > 0 or next_val > 0 or i < 4:
                diff_sym = "↑" if next_val > val else ("↓" if next_val < val else " ")
                inv_text += f"{inv_labels[i][:7]:7s}:{int(val):1d}->{int(next_val):1d}{diff_sym} "
                if i % 2 == 1: inv_text += "\n"

    stats_text = (
        f"Step: {step} | Act: {action} ({act_name})\n"
        f"Acc: {acc:.1f}% | Unc: {mean_uncer:.3f}\n"
        f"Mismatch: {total_errors}\n"
        f"{inv_text}"
    )
    axes[5].text(0.0, 1.0, stats_text, fontsize=11, family='monospace', verticalalignment='top')
    
    # --- Add Color Legend (More compact) ---
    import matplotlib.patches as mpatches
    legend_items = [
        ("Water", colors[1]), ("Grass", colors[2]), ("Stone", colors[3]),
        ("Tree", colors[6]), ("Coal", colors[8]), ("Iron", colors[9]),
        ("Table", colors[11]), ("Diam", colors[10]), ("Plant", colors[18])
    ]
    for i, (label, color) in enumerate(legend_items):
        y_pos = 0.35 - (i // 3) * 0.08
        x_pos = (i % 3) * 0.33
        axes[5].add_patch(mpatches.Rectangle((x_pos, y_pos), 0.05, 0.08, color=color, transform=axes[5].transAxes))
        axes[5].text(x_pos + 0.07, y_pos + 0.02, label, fontsize=10, transform=axes[5].transAxes)
    
    # Add grid lines for a clearer layout view.
    for i in range(5):
        axes[i].set_xticks(np.arange(-.5, w_crop, 1), minor=True)
        axes[i].set_yticks(np.arange(-.5, h_crop, 1), minor=True)
        axes[i].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        # Hide ticks and labels without triggering Matplotlib warnings.
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].tick_params(which='both', size=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"step_{step:06d}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Crafter NPZ exploration coverage.")
    parser.add_argument("data_path", help="Path to collected .npz dataset")
    parser.add_argument("--save", dest="save_path", default=None, help="Output image path (png)")
    parser.add_argument("--title", default="Crafter Exploration Coverage", help="Plot title")
    args = parser.parse_args()

    plot_crafter_coverage_from_npz(
        data_path=args.data_path,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()
