import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLAYER_ID = 13  # Updated: player ID in new CustomCrafterEnv mapping (was 10)


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
    from generator.crafter_env_designer import CRAFTER_OBJ_MAP
    
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
#  World Model Support Functions for Crafter (Classification)
# ============================================================
import torch
import torch.nn.functional as F

# CustomCrafterEnv Object IDs (matching original Crafter mat_ids + shifted entities):
# 0=None/empty, 1=water, 2=grass, 3=stone, 4=path, 5=sand, 6=tree,
# 7=lava, 8=coal, 9=iron, 10=diamond, 11=table, 12=furnace,
# 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
# Total: 20 classes (0-19)
# Direction IDs: 0=none, 1=up, 2=down, 3=left, 4=right  →  5 classes
CRAFTER_OBJ_CLASSES = 20
CRAFTER_DIR_CLASSES = 5
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


def crafter_reconstruct_from_logits(next_pred: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct integer observation channels from logits using argmax.
    Used for pseudo-MSE metric logging.

    Args:
        next_pred: (B, 22, H, W)

    Returns:
        (B, 2, H, W) float tensor: [obj_id, dir_id]
    """
    obj_pred = next_pred[:, :CRAFTER_OBJ_CLASSES].argmax(dim=1).float()  # (B, H, W)
    dir_pred = next_pred[:, CRAFTER_OBJ_CLASSES:].argmax(dim=1).float()  # (B, H, W)
    return torch.stack([obj_pred, dir_pred], dim=1)  # (B, 2, H, W)
    
def visualize_crafter_wm(
    obs_masked: torch.Tensor, 
    obs_next_masked: torch.Tensor, 
    obs_pred_logits: torch.Tensor,
    action: int,
    step: int,
    save_dir: str = "trainer/logs/wm_visual",
    full_map_size: tuple = (64, 64),
    agent_pos: tuple = (32, 32),
    inv: np.ndarray = None,
    inv_next: np.ndarray = None
):
    """
    Visualize WM prediction for Crafter with 5 panels.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle batch dimension if present
    if obs_masked.ndim == 4: obs_masked = obs_masked[0]
    if obs_next_masked.ndim == 4: obs_next_masked = obs_next_masked[0]
    if obs_pred_logits.ndim == 4: obs_pred_logits = obs_pred_logits[0]
        
    # 1. Extract predicted IDs and uncertainty.
    obj_logits = obs_pred_logits[:CRAFTER_OBJ_CLASSES]  
    obj_probs = F.softmax(obj_logits, dim=0)
    obj_pred = obj_probs.argmax(dim=0).detach().cpu().numpy()
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
