import os
import sys
from pathlib import Path
from typing import Optional
import dotenv
from matplotlib import pyplot as plt
import numpy as np
import torch
from . import utilis_support
from typing import Dict
from matplotlib.colors import LinearSegmentedColormap
from generator.common.utils import generate_color_map, layout_to_string, combine_maps, add_outer_wall
import omegaconf
import torch.serialization

def replace_values(arr, old_values, new_values):
    assert arr.ndim >= 2 and len(old_values) == len(new_values)
    mapping = np.arange(256, dtype=arr.dtype)
    mapping[np.array(old_values, dtype=arr.dtype)] = np.array(new_values, dtype=arr.dtype)
    arr[:, :] = np.take(mapping, arr[:, :])
    return arr

def create_mask(state_shape, agent_position, mask_size):
    mask = np.zeros((state_shape[0], state_shape[1]), dtype=bool)
    y, x = agent_position
    half_size = mask_size // 2
    y_start, y_end = max(0, y - half_size), min(state_shape[0], y + half_size + 1)
    x_start, x_end = max(0, x - half_size), min(state_shape[1], x + half_size + 1)
    mask[y_start:y_end, x_start:x_end] = True
    return mask

def ColRowCanl_to_CanlRowCol(state):
    if len(state.shape) == 3:
        dims = (2, 1 ,0)
    elif len(state.shape) == 4:
        dims = (0, 3, 2, 1)
    else:
        raise ValueError("Input must be a 3D or 4D array.")
    
    transpose_func = getattr(state, "permute", None) or getattr(state, "transpose", None)  # 如果有permute方法就用permute，否则用transpose
    if transpose_func:
        return transpose_func(*dims)
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")

def get_agent_position(state):
    """
    Extract the (y, x) position of the agent from a MiniGrid state tensor.

    Parameters
    ----------
    state : np.ndarray or torch.Tensor
        MiniGrid observation tensor, expected in one of the following formats:
        - (C, H, W): single observation (channels-first)
        - (B, C, H, W): batch of observations
        The first channel (index 0) corresponds to object IDs in the grid.
        The agent is represented by object ID == 10.

    Returns
    -------
    np.ndarray
        - If input is (C, H, W): returns np.array([y, x]) for the agent position.
        - If input is (B, C, H, W): returns np.ndarray of shape (B, 2),
          where each row is [y, x] for one sample.
        If the agent is not found, returns [-1, -1] for that sample.
    """
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    if len(state.shape) == 3:
        channel, row, col = state.shape
        agent_position_index = np.argmax(state[0, :, :])
        agent_position_yx = np.unravel_index(agent_position_index, (row, col))
        return agent_position_yx

    elif len(state.shape) == 4:
        B, channel, row, col = state.shape
        agent_position_index = np.argmax(state[:, 0, :, :].reshape(B, -1), axis=1)
        agent_position_yx_batch = np.stack(np.unravel_index(agent_position_index, (row, col)), axis=1)
        return agent_position_yx_batch
    else:
        raise ValueError("Input must be a 3D or 4D array.")

def extract_masked_state(state, mask_size, agent_position_yx):
    tensor_flag = False
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
        tensor_flag = True

    #区分带batch 和不带batch的情况
    if len(state.shape) == 3:
        state_masked = utilis_support.extract_masked_state_support(state, agent_position_yx , mask_size)

    if len(state.shape) == 4:
        B, channel, row, col = state.shape
        
        state_masked = np.zeros((B, channel, mask_size, mask_size), dtype=state.dtype) 
        for i in range(B):
            state_masked[i, :, :, :] = utilis_support.extract_masked_state_support(state[i], agent_position_yx[i], mask_size)

    if tensor_flag:
        state_masked = torch.from_numpy(state_masked).cuda()
    return state_masked

def put_back_masked_state(state_masked, orginal_state, mask_size, agent_position_yx):
    tensor_flag = False
    if isinstance(state_masked, torch.Tensor):
        state_masked = state_masked.detach().cpu().numpy()
        tensor_flag = True
    
    if isinstance(orginal_state, torch.Tensor):
        orginal_state = orginal_state.detach().cpu().numpy()
        tensor_flag = True

    if len(state_masked.shape) == 3:
        channels, rows, cols = orginal_state.shape
        y, x = agent_position_yx
        half = mask_size // 2

        src_slice_y = slice(max(y - half, 0), min(y + half + 1, rows))
        src_slice_x = slice(max(x - half, 0), min(x + half + 1, cols))

        dest_slice_y = slice(max(0, half - y), max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)))
        dest_slice_x = slice(max(0, half - x), max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)))
        orginal_state[:, src_slice_y, src_slice_x] = state_masked[:, dest_slice_y, dest_slice_x]



    if tensor_flag:
        orginal_state = torch.from_numpy(orginal_state).cuda()
    return orginal_state

def load_model_weight(model, weight_path, freeze=True):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

        with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
            checkpoint = torch.load(weight_path, weights_only=False, map_location=device)


        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # 是纯 state_dict
            state_dict = checkpoint
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(cleaned_state_dict, strict=False)

        model.to(device)
        model.eval()

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        print(f"[Load] Model weights loaded from: {weight_path}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Weight file not found at {weight_path}")

    except KeyError as e:
        raise KeyError(f"Error: missing key in checkpoint -> {e}")

    except RuntimeError as e:
        raise RuntimeError(
            f"Model loading RuntimeError: {e}\n"
            f"If this is due to PyTorch safe-unpickle restrictions, "
            f"ensure add_safe_globals was applied correctly."
        )

    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}")

def normalize_obs(x, obs_norm_values):
    if isinstance(x, np.ndarray):
        # NumPy array case
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        # PyTorch tensor case
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")
    
    # Normalize vector observations (e.g. BipedalWalker).
    if x.ndim == 1:
        if obs_norm_values is None or len(obs_norm_values) not in (1, x.shape[0]):
            raise ValueError("Normalization values must be scalar or match the feature dimension.")
        if len(obs_norm_values) == 1:
            if obs_norm_values[0] != 0:
                x /= obs_norm_values[0]
        else:
            for i, value in enumerate(obs_norm_values):
                if value != 0:
                    x[i] /= value
    elif x.ndim == 2:
        if obs_norm_values is None or len(obs_norm_values) not in (1, x.shape[1]):
            raise ValueError("Normalization values must be scalar or match the feature dimension.")
        if len(obs_norm_values) == 1:
            if obs_norm_values[0] != 0:
                x /= obs_norm_values[0]
        else:
            for i, value in enumerate(obs_norm_values):
                if value != 0:
                    x[:, i] /= value
    elif x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (H, W, C)
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:  # Avoid division by zero
                x[i, :, :] /= max_val
        # Flatten the data into a 1D vector and convert to a torch tensor
        x = x.reshape(channel, row, col)
        
    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (B, H, W, C)
        B, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] /= max_val
        # Flatten each observation in the batch
        x = x.reshape(B, channel, row, col)
    else:
        raise ValueError("Input must be a 1D, 2D, 3D or 4D array.")
    
    return x

def denormalize_obj(x, obs_norm_values):    
    # Denormalize vector observations (e.g. BipedalWalker).
    if x.ndim == 1:
        if obs_norm_values is None or len(obs_norm_values) not in (1, x.shape[0]):
            raise ValueError("Normalization values must be scalar or match the feature dimension.")
        if len(obs_norm_values) == 1:
            if obs_norm_values[0] != 0:
                x *= obs_norm_values[0]
        else:
            for i, value in enumerate(obs_norm_values):
                if value != 0:
                    x[i] *= value
    elif x.ndim == 2:
        if obs_norm_values is None or len(obs_norm_values) not in (1, x.shape[1]):
            raise ValueError("Normalization values must be scalar or match the feature dimension.")
        if len(obs_norm_values) == 1:
            if obs_norm_values[0] != 0:
                x *= obs_norm_values[0]
        else:
            for i, value in enumerate(obs_norm_values):
                if value != 0:
                    x[:, i] *= value
    elif x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (H, W, C)
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:  # Avoid division by zero
                x[i, :, :] *= max_val
        # Flatten the data into a 1D vector and convert to a torch tensor
        x = x.reshape(channel, row, col)
        
    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (B, H, W, C)
        B, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] *= max_val
        # Flatten each observation in the batch
        x = x.reshape(B, channel, row, col)
    else:
        raise ValueError("Input must be a 1D, 2D, 3D or 4D array.")
    return x
    
def map_obs_to_nearest_value(obs_denorm, obj_values, color_values, state_values):
    obs_denorm[0, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[0, :, :], obj_values)
    obs_denorm[1, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[1, :, :], color_values)
    obs_denorm[2, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[2, :, :], state_values)
    return obs_denorm

class Visualization:
    def __init__(self, config=''):
        self.cfg = config
        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)
    

    def compare_states(self, obs, obs_next, act, step_counter=0, saveImage=False, size=(10, 4), shrink=0.5):
        if isinstance(obs, np.ndarray): 
            obs = torch.from_numpy(obs).cuda()
        if isinstance(obs_next, np.ndarray): 
            obs_next = torch.from_numpy(obs_next).cuda()
        plt.close()
        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[0, :, :].detach().cpu().numpy() * obj_ratio
        direction = self.cfg.direction_map[round(obs[2, :, :].detach().cpu().numpy().max() * dir_ratio)]

        state_image_next = obs_next[0, :, :].detach().cpu().numpy() * obj_ratio
        direction_next = self.cfg.direction_map[round(obs_next[2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        if act is None:
            action = "None"
        else:
            action = self.cfg.action_map[round(act * act_ratio)]
    
        num_colors = 11
        custom_cmap = plt.cm.get_cmap('jet', num_colors)
        self._plot_subplot(1, 2, 1, state_image, custom_cmap, 'State', f"Dir: {direction}  Action: {action}", shrink)
        self._plot_subplot(1, 2, 2, state_image_next, custom_cmap, 'State Pre', f"Dir: {direction_next}", shrink)
        plt.tight_layout()
        if saveImage:
            save_file = os.path.join(self.cfg.save_path, f"Compare_{step_counter}.png")
            plt.savefig(save_file)
            plt.close()
        else:
            plt.show()

    def visualize_single_state(self, obs, act=None, info=None, ep=1, index=1,save_flag=False,shrink=1):
        act = 4 if act == 5 else act
        if info is not None and 'carrying_key' in info:
            key = info['carrying_key']
        else:
            key = "None"

        if isinstance(obs, np.ndarray): 
            obs = torch.from_numpy(obs).cuda()

        plt.close()
        obs = obs.detach().cpu().numpy()  # Convert tensor to numpy

        state_image = obs[:, :, 0]

        if act is None:
            action = "None"
        else:
            action = self.cfg.action_map[int(act)]
    
        color_list = [
            "#440154",  # 紫
            "#3b528b",  # 深蓝
            "#21918c",  # 蓝绿
            "#5ec962",  # 绿
            "#fde725",  # 黄
            "#f98400",  # 橘
            "#d00000",  # 红
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom7", color_list, N=7)
        plt.imshow(state_image, cmap=custom_cmap, interpolation='nearest')
        plt.title(f"Act: {action}, key:{key}")
        plt.colorbar(shrink=shrink)
        if save_flag:
            if not os.path.exists(self.cfg.save_path):
                os.mkdir(self.cfg.save_path)
            save_file = os.path.join(self.cfg.save_path, f"colect data_{ep}_{index}.png")
            plt.savefig(save_file)
        else:
            plt.show()


    def visualize_data(self, obs_all, obs_next_all, act, obs, obs_next, info=None, step_counter='', pos_xy=[], size=(14, 10), shrink=1):
        if info is not None and 'carrying_key' in info:
            key = info['carrying_key'][-1].item()
        else:
            key = "None"

        def convert_ny(data):
            data = data.detach().cpu().numpy() 
            return data
        obs_all = convert_ny(obs_all)
        obs_next_all = convert_ny(obs_next_all)
        obs = convert_ny(obs)
        obs_next = convert_ny(obs_next)

        mask_size = self.cfg.attention_mask_size


        obs_mask = obs[:, 0, :, :]  # Convert tensor to numpy
        all_obs = obs_all[:, 0, :, :]  # Convert tensor to numpy
        all_obs_next = obs_next_all[:, 0, :, :]   # Convert tensor to numpy
        direction = [self.cfg.direction_map[int(x)] for x in np.round(obs[:, 2, mask_size//2, mask_size//2])]
        action = [self.cfg.action_map[int(x)] for x in act[:]]
        next_direction = [self.cfg.direction_map[int(x)] for x in np.round(obs_next[:, 2, mask_size//2, mask_size//2])]
        obs_next_mask = obs_next[:, 0, :, :]  # Convert tensor to numpy

        color_list = [
            "#440154",  # 紫
            "#3b528b",  # 深蓝
            "#21918c",  # 蓝绿
            "#5ec962",  # 绿
            "#fde725",  # 黄
            "#f98400",  # 橘
            "#d00000",  # 红
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom7", color_list, N=7)
        def show(cur_all_obs, cur_all_next, cur_obs, cur_obs_next, cur_direction, cur_action, cur_next_direction, key, step_counter, index):
            plt.figure(figsize=size)
            self._plot_subplot(2, 2, 1, cur_all_obs, custom_cmap, 'whole map', "", shrink)
            self._plot_subplot(2, 2, 2, cur_all_next, custom_cmap, 'whole map next', "", shrink)
            self._plot_subplot(2, 2, 3, cur_obs, custom_cmap, 'mask obs', f"Dir: {cur_direction}  Action: {cur_action}, key:{key}", shrink)
            self._plot_subplot(2, 2, 4, cur_obs_next, custom_cmap, 'mask obs next', f"Dir:{cur_next_direction}", shrink)
            plt.tight_layout()

            # Save plot
            if not os.path.exists(self.cfg.save_path):
                os.mkdir(self.cfg.save_path)
            save_file = os.path.join(self.cfg.save_path, f"Source_data_{step_counter}_{index}.png")
            plt.savefig(save_file)
            plt.close()
        
        for i in range(len(act)):
            show(all_obs[i], all_obs_next[i], obs_mask[i], obs_next_mask[i], direction[i], action[i], next_direction[i], key, step_counter, i)
        

    def visualize_attention(self, obs, act, attentionWeight, obs_next, obs_pred, step_counter, info=None, size=(14, 10), shrink=1):
        if info is not None and 'carrying_key' in info:
            key = info['carrying_key'][-1].item()
        else:
            key = "None"
        mask_size = self.cfg.attention_mask_size
        channel, row, col = 3, mask_size, mask_size
        # Preprocess data
        obs_next_temp = obs_next.view(obs_next.shape[0], channel, row, col) 

        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[-1, 0, :, :].detach().cpu().numpy() * obj_ratio  # Convert tensor to numpy
        direction = self.cfg.direction_map[round(obs[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        action = self.cfg.action_map[round(act[-1].item() * act_ratio)]
        next_direction = self.cfg.direction_map[round(obs_next_temp[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * obj_ratio  # Convert tensor to numpy
        pred_direction_idx = round(obs_pred[-1, :].reshape(channel, row, col)[2, :, :].detach().cpu().numpy().max() * dir_ratio)
        obs_pred = np.round(obs_pred[-1, :].reshape(channel, row, col)[0, :, :].detach().cpu().numpy() * obj_ratio)  # Convert tensor to nump
        if len(attentionWeight.shape) == 2:
            heat_map = attentionWeight[-1, :].reshape(row, col).detach().cpu().numpy()  # Convert tensor to numpy
        elif len(attentionWeight.shape) == 3:
            heat_map = attentionWeight[-1, (mask_size**2)//2, :].reshape(row, col).detach().cpu().numpy()
        else:
            raise ValueError("Attention weight shape is not supported.")
        pre_direction = self.cfg.direction_map.get(pred_direction_idx, "Unknown")

        
        # Set up colormap for states
        num_colors = 13
        custom_cmap = plt.cm.get_cmap('jet', num_colors)
        plt.figure(figsize=size)
        self._plot_subplot(2, 2, 1, state_image, custom_cmap, 'State', f"State  Dir: {direction}  Action: {action}, key:{key}", shrink)
        self._plot_subplot(2, 2, 3, heat_map, 'viridis', 'Attention', "Attention Heatmap", shrink)
        self._plot_subplot(2, 2, 2, obs_next, custom_cmap, 'Next State', f"Next State  Dir:{next_direction}", shrink)
        self._plot_subplot(2, 2, 4, obs_pred, custom_cmap, 'Predicted', f"Pre State  Dir: {pre_direction}", shrink)
        plt.tight_layout()

        # Save plot
        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)
        save_file = os.path.join(self.cfg.save_path, f"Attention_{step_counter}.png")
        plt.savefig(save_file)
        plt.close()

    def _plot_subplot(self, row, col, position, data, cmap, colorbar_label, title, shrink):
        plt.subplot(row, col, position)
        im = plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, shrink=shrink, label=colorbar_label)
        plt.title(title)
    
    def _plot(self, data, cmap, title, shrink):
        plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(shrink=shrink, label=title)
        plt.title(title)
        plt.show()

def get_env(env_name: str, default: Optional[str] = None) -> str:
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value

def load_envs(env_file: Optional[str] = '.env') -> None:
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def merge_data_dicts(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = {}
    for key in d1.keys():
        if key in d2:
            merged[key] = np.concatenate([d1[key], d2[key]], axis=0)
        else:
            merged[key] = d1[key]
    return merged


def extract_unique_patches(layout_str: str, patch_size: int):
    """
    Extract all unique local patches (substrings) of size patch_size x patch_size
    from a string-based 2D map. Split the target task into mini tasks based on unique local patterns.

    This function:
        - Treats the input as a pure character grid (no agents, no assumptions)
        - Slides a fixed-size square window across the map
        - Extracts each window as a patch
        - Converts the patch into a multiline string
        - Removes duplicates
        - Returns all unique patch strings

    Arguments:
        layout_str (str): Multi-line string representing the map.
                          Each character corresponds to a tile (e.g., '#', '.', 'D', 'K').
        patch_size (int): The dimension of the square patch.
                          Typically equal to your model's mask_size.
                          Must be an odd number (3, 5, 7, ...).

    Returns:
        List[str]: A list of unique patches, each represented as a multiline string.
                   Each patch is of size patch_size x patch_size.
    """

    # ---------------------------------------------------------
    # 1. Convert the string map into a 2D grid (NumPy array)
    # ---------------------------------------------------------
    # Split the input into lines and convert each line into a list of characters.
    lines = [list(line) for line in layout_str.strip().split("\n")]
    H, W = len(lines), len(lines[0])  # map height and width
    grid = np.array(lines)

    # ---------------------------------------------------------
    # 2. Slide a window across the map and extract patches
    # ---------------------------------------------------------
    r = patch_size // 2  # half-size of the patch window
    unique_set = set()   # used for deduplication
    unique_patches = []  # final list of patch strings

    for i in range(H):
        for j in range(W):
            
            # Skip positions where the patch would go out of bounds
            if i - r < 0 or i + r >= H or j - r < 0 or j + r >= W:
                continue

            # Extract the patch as a (patch_size x patch_size) slice of the grid
            patch = grid[i - r : i + r + 1, j - r : j + r + 1]

            # --- NEW: Check center cell is walkable ---
            center = patch[r, r]
            if center != 'E':       # only keep patches centered on empty floor
                continue

            # Convert the patch into a string representation
            # Each row becomes a line in the final string
            patch_str = "\n".join("".join(row) for row in patch)

            # Add to the list only if it is unique
            if patch_str not in unique_set:
                unique_set.add(patch_str)
                unique_patches.append(patch_str)

    return unique_patches

def patch_to_array(patch_str):
    """string patch → numpy array"""
    lines = patch_str.split("\n")
    return np.array([list(row) for row in lines])

def array_to_patch(arr):
    """numpy array → string patch"""
    return "\n".join("".join(row) for row in arr)


# ============================================================
# 3. Combine 4 patches into 1 minitask (2×2 grid)
# ============================================================

def combine_patches_2x2(patches):
    """
    patches: list of 4 patch strings
    return: combined minitask layout string
    """
    assert len(patches) == 4

    A = patch_to_array(patches[0])
    B = patch_to_array(patches[1])
    C = patch_to_array(patches[2])
    D = patch_to_array(patches[3])

    K = A.shape[0]
    big = np.full((2*K, 2*K), 'E', dtype=str)

    # 2×2 placement
    big[0*K:1*K, 0*K:1*K] = A
    big[0*K:1*K, 1*K:2*K] = B
    big[1*K:2*K, 0*K:1*K] = C
    big[1*K:2*K, 1*K:2*K] = D

    return array_to_patch(big)


# ============================================================
# 4. Utility: check if selected patches contain ANY new patch
# ============================================================

def minitask_has_new_patch(selected, covered_set):
    for p in selected:
        if p not in covered_set:
            return True
    return False


def combine_patches_1x2(patches):
    """
    patches: list of 2 patch strings
    return: combined minitask layout string
    """
    assert len(patches) == 2

    A = patch_to_array(patches[0])
    B = patch_to_array(patches[1])

    K = A.shape[0]

    # Create 1 x 2 grid
    big = np.full((K, 2 * K), 'E', dtype=str)
    big[:, 0*K:1*K] = A
    big[:, 1*K:2*K] = B

    # Convert back to string (same format as original)
    return array_to_patch(big)



# ============================================================
# 5. Master generator: keep building minitasks until all patches covered
# ============================================================

def generate_minitasks_until_covered(
    all_patches, 
    patch_size, 
    patches_per_minitask
):
    """
    Generate minitasks until all unique target patches are covered.

    Coverage criterion: all_patches ⊆ covered_patches
    """

    import random

    covered = set()      
    minitasks = []       
    iteration = 0

    all_set = set(all_patches)

    if patches_per_minitask == 1:
        while not all_set.issubset(covered):
            iteration += 1

            remaining = list(all_set - covered)

            # pick exactly 1 patch
            selected = [random.choice(remaining)]
            mt_layout = selected[0]                   # use single patch directly
            mt_layout = add_outer_wall(mt_layout)     # add padding wall

            minitasks.append(mt_layout)

            # update coverage
            covered.add(selected[0])
            # print(f"[{iteration}] Accepted → covered "
            #     f"{len(covered)}/{len(all_patches)}")

        # After covering all patches: convert to combined maps
        minitask_set = []
        for layout_map in minitasks:
            color_map = generate_color_map(layout_map)
            combined_map = combine_maps(layout_map, color_map, None)
            minitask_set.append(combined_map)

        # print("\nAll patches covered!")
        return minitask_set

    # -------------- correct condition --------------
    while not all_set.issubset(covered):
        iteration += 1

        remaining = list(all_set - covered)

        # Case 1: enough remaining
        if len(remaining) >= patches_per_minitask:
            selected = random.sample(remaining, patches_per_minitask)

        # Case 2: need padding
        else:
            selected = remaining.copy()
            missing = patches_per_minitask - len(selected)

            if len(covered) == 0:
                pad = random.sample(all_patches, missing)
            else:
                pad = random.sample(list(covered), missing)

            selected += pad

        # Skip if no new patch
        if not minitask_has_new_patch(selected, covered):
            print(f"[{iteration}] Skipped (no new patch). "
                  f"{len(covered)}/{len(all_patches)} covered.")
            continue

        # Combine → layout
        if patches_per_minitask == 2:
            mt_layout = combine_patches_1x2(selected)
        else:
            mt_layout = combine_patches_2x2(selected)

        # Add walls
        mt_layout = add_outer_wall(mt_layout)

        minitasks.append(mt_layout)

        # Extract all patches in this minitask
        mt_patches = extract_unique_patches(mt_layout, patch_size)

        # Update covered
        covered.update(mt_patches)

        # print(f"[{iteration}] Accepted → covered "
        #       f"{len(covered)}/{len(all_patches)}")


    # Final safety check
    if not all_set.issubset(covered):
        print("WARNING: Not all target patches are covered!")

    # Convert to full map (layout + color)
    minitask_set = []
    for layout_map in minitasks:
        color_map = generate_color_map(layout_map)
        combined_map = combine_maps(layout_map, color_map, None)
        minitask_set.append(combined_map)

    return minitask_set


load_envs()
PROJECT_ROOT : Path = Path(get_env("PROJECT_ROOT"))
GENERATOR_PATH : Path = Path(get_env("GENERATOR_PATH"))
TRAINER_PATH : Path = Path(get_env("TRAINER_PATH"))
WORLD_MODEL_PATH = Path(get_env("WORLD_MODEL_PATH"))
sys.path.append(str(PROJECT_ROOT.resolve()))	



