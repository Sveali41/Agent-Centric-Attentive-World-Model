import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from minigrid.wrappers import FullyObsWrapper

from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from generator.common.utils import (
    add_outer_wall,
    combine_maps,
    generate_color_map,
    generate_obj_map,
    interpret_color_map,
    
    
)
from generator.data.env_dataset_support import generate_envs_dataset, is_reachable
from modelBased.common import utils


def _get_env_visualize(cfg) -> bool:
    """Read optional env.visualize safely for OmegaConf struct configs."""
    try:
        from omegaconf import OmegaConf
        return bool(OmegaConf.select(cfg, "env.visualize", default=False))
    except Exception:
        return False


def wrap_env(env_layout, cfg):
    render_mode = "human" if _get_env_visualize(cfg) else None
    layout_string = generate_obj_map(env_layout, cfg.training_generator.map_element)
    color_string = generate_color_map(layout_string)
    print("layout_string: ", layout_string)
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            layout_str=layout_string,
            color_str=color_string,
            custom_mission="Navigate to the start position.",
            render_mode=render_mode,
        )
    )
    return env


def wrap_env_from_text(file_path, max_steps, cfg):
    render_mode = "human" if _get_env_visualize(cfg) else None
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=file_path,
            custom_mission="Navigate to the start position.",
            max_steps=max_steps,
            render_mode=render_mode,
        )
    )
    return env


def interpret_env(env, cfg, color_array=None):
    layout_string = generate_obj_map(env, cfg.training_generator.map_element)
    if color_array is None:
        color_string = generate_color_map(layout_string)
    else:
        color_string = interpret_color_map(color_array, cfg.training_generator.color_element)
    print("layout_string: ", layout_string)
    return layout_string, color_string


def generate_final_task(
    rows,
    cols,
    num_maps,
    cfg,
    save=True,
    wall_p_range=(0.1, 0.5),
    door_p_range=(0.075, 0.1),
    key_p_range=(0.1, 0.15),
    max_len=1e7,
    random_gen_max=3e4,
):
    final_task_dict = generate_envs_dataset(
        rows,
        cols,
        num_maps,
        wall_p_range=wall_p_range,
        door_p_range=door_p_range,
        key_p_range=key_p_range,
        max_len=max_len,
        random_gen_max=random_gen_max,
    )

    file_names = []
    if save:
        from modelBased.common.utils import TRAINER_PATH

        for idx, key in enumerate(final_task_dict):
            map_data = final_task_dict[key]
            map_tensor = torch.tensor(map_data).unsqueeze(0)
            layout_string = generate_obj_map(map_tensor, cfg.training_generator.map_element)
            color_string = generate_color_map(layout_string)
            save_path = os.path.join(
                TRAINER_PATH,
                "level",
                "final_task",
                f"gen_final_task_{idx}.txt",
            )
            combine_maps(layout_string, color_string, save_path)
            file_names.append(save_path)

    return final_task_dict


def env_editor(env, dynamic_object, cfg, flip_ratio=0.15, max_attempts=20000):
    key_door = 4 or 5 in dynamic_object

    if flip_ratio <= 0.03:
        env_original_layout = env.copy()
        env = wrap_env(torch.tensor(env_original_layout).unsqueeze(0), cfg)
        return env, env_original_layout

    for _ in range(max_attempts):
        env = env.copy()
        h, w = env.shape

        inner_coords = [
            (i, j)
            for i in range(1, h - 1)
            for j in range(1, w - 1)
            if env[i, j] in (1, 2)
        ]
        num_flips = int(len(inner_coords) * flip_ratio)
        flip_coords = random.sample(inner_coords, num_flips)

        for i, j in flip_coords:
            env[i, j] = 2 if env[i, j] == 1 else 1

        movable_coords = [
            (i, j)
            for i in range(1, h - 1)
            for j in range(1, w - 1)
            if env[i, j] in dynamic_object
        ]
        empty_inner_coords = [
            (i, j)
            for i in range(1, h - 1)
            for j in range(1, w - 1)
            if env[i, j] not in dynamic_object and (i, j) not in flip_coords
        ]

        for i, j in movable_coords:
            val = env[i, j]
            env[i, j] = 1
            new_i, new_j = random.choice(empty_inner_coords)
            env[new_i, new_j] = val
            empty_inner_coords.remove((new_i, new_j))

        if is_reachable(env, key_door=key_door):
            env_layout = env
            env = wrap_env(torch.tensor(env).unsqueeze(0), cfg)
            return env, env_layout

    return env_editor(env, dynamic_object, cfg, flip_ratio - 0.02, max_attempts)


def ColRowCanl_to_CanlRowCol(state):
    if len(state.shape) == 3:
        dims = (2, 1, 0)
    elif len(state.shape) == 4:
        dims = (0, 3, 2, 1)
    else:
        raise ValueError("Input must be a 3D or 4D array.")

    transpose_func = getattr(state, "permute", None) or getattr(state, "transpose", None)
    if transpose_func:
        return transpose_func(*dims)
    raise TypeError("Input must be a PyTorch tensor or a NumPy array.")


def get_agent_position(state, player_id: int = 10):
    """
    Find the agent position in the observation.
    player_id: 10 for MiniGrid, 13 for Crafter.
    """
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()

    def _find_agent_yx(obs_chw):
        obj_map = obs_chw[0]
        hits = np.argwhere(obj_map == player_id)
        if len(hits) > 0:
            return hits[0][0], hits[0][1]  # (y, x)
        # Fallback: argmax
        return np.unravel_index(np.argmax(obj_map), obj_map.shape)

    if len(state.shape) == 3:
        return _find_agent_yx(state)

    if len(state.shape) == 4:
        bsz = state.shape[0]
        positions = np.zeros((bsz, 2), dtype=np.int64)
        for i in range(bsz):
            positions[i] = _find_agent_yx(state[i])
        return positions

    raise ValueError("Input must be a 3D or 4D array.")


def extract_masked_state(state, mask_size, agent_position_yx):
    tensor_flag = False
    tensor_device = None
    if isinstance(state, torch.Tensor):
        tensor_device = state.device
        state = state.detach().cpu().numpy()
        tensor_flag = True

    # --- Environment-Specific Padding ---
    # Crafter (2 channels) -> Pad with Grass (ID 2)
    # MiniGrid (3 channels) -> Pad with Zero (ID 0)
    pad_val = 2 if state.shape[-3] == 2 else 0
    
    if len(state.shape) == 3:
        state_masked = utils.extract_masked_state_support(
            state, agent_position_yx, mask_size, pad_value=pad_val
        )
    elif len(state.shape) == 4:
        bsz, channel, _, _ = state.shape
        state_masked = np.zeros((bsz, channel, mask_size, mask_size), dtype=state.dtype)
        for i in range(bsz):
            state_masked[i, :, :, :] = utils.extract_masked_state_support(
                state[i], agent_position_yx[i], mask_size, pad_value=pad_val
            )
    else:
        raise ValueError("Input must be a 3D or 4D array.")

    if tensor_flag:
        if tensor_device is None:
            tensor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_masked = torch.from_numpy(state_masked).to(tensor_device)
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

        dest_slice_y = slice(
            max(0, half - y),
            max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)),
        )
        dest_slice_x = slice(
            max(0, half - x),
            max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)),
        )
        orginal_state[:, src_slice_y, src_slice_x] = state_masked[
            :, dest_slice_y, dest_slice_x
        ]

    if tensor_flag:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orginal_state = torch.from_numpy(orginal_state).to(device)
    return orginal_state


def map_obs_to_nearest_value(obs_denorm, obj_values, color_values, state_values):
    obs_denorm[0, :, :] = utils.map_to_nearest_value_support(
        obs_denorm[0, :, :], obj_values
    )
    obs_denorm[1, :, :] = utils.map_to_nearest_value_support(
        obs_denorm[1, :, :], color_values
    )
    obs_denorm[2, :, :] = utils.map_to_nearest_value_support(
        obs_denorm[2, :, :], state_values
    )
    return obs_denorm


class Visualization:
    def __init__(self, config=""):
        self.cfg = config
        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)

    def compare_states(
        self,
        obs,
        obs_next,
        act,
        step_counter=0,
        saveImage=False,
        size=(10, 4),
        shrink=0.5,
    ):
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
        direction = self.cfg.direction_map[
            round(obs[2, :, :].detach().cpu().numpy().max() * dir_ratio)
        ]

        state_image_next = obs_next[0, :, :].detach().cpu().numpy() * obj_ratio
        direction_next = self.cfg.direction_map[
            round(obs_next[2, :, :].detach().cpu().numpy().max() * dir_ratio)
        ]
        action = "None" if act is None else self.cfg.action_map[round(act * act_ratio)]

        num_colors = 11
        custom_cmap = plt.cm.get_cmap("jet", num_colors)
        self._plot_subplot(
            1,
            2,
            1,
            state_image,
            custom_cmap,
            "State",
            f"Dir: {direction}  Action: {action}",
            shrink,
        )
        self._plot_subplot(
            1,
            2,
            2,
            state_image_next,
            custom_cmap,
            "State Pre",
            f"Dir: {direction_next}",
            shrink,
        )
        plt.tight_layout()
        if saveImage:
            save_file = os.path.join(self.cfg.save_path, f"Compare_{step_counter}.png")
            plt.savefig(save_file)
            plt.close()
        else:
            plt.show()

    def visualize_single_state(
        self, obs, act=None, info=None, ep=1, index=1, save_flag=False, shrink=1
    ):
        act = 4 if act == 5 else act
        key = info["carrying_key"] if info is not None and "carrying_key" in info else "None"

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).cuda()

        plt.close()
        obs = obs.detach().cpu().numpy()
        state_image = obs[:, :, 0]
        action = "None" if act is None else self.cfg.action_map[int(act)]

        color_list = [
            "#440154",
            "#3b528b",
            "#21918c",
            "#5ec962",
            "#fde725",
            "#f98400",
            "#d00000",
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom7", color_list, N=7)
        plt.imshow(state_image, cmap=custom_cmap, interpolation="nearest")
        plt.title(f"Act: {action}, key:{key}")
        plt.colorbar(shrink=shrink)
        if save_flag:
            if not os.path.exists(self.cfg.save_path):
                os.mkdir(self.cfg.save_path)
            save_file = os.path.join(self.cfg.save_path, f"colect data_{ep}_{index}.png")
            plt.savefig(save_file)
        else:
            plt.show()

    def visualize_data(
        self,
        obs_all,
        obs_next_all,
        act,
        obs,
        obs_next,
        info=None,
        step_counter="",
        pos_xy=[],
        size=(14, 10),
        shrink=1,
    ):
        key = info["carrying_key"][-1].item() if info is not None and "carrying_key" in info else "None"

        def convert_ny(data):
            return data.detach().cpu().numpy()

        obs_all = convert_ny(obs_all)
        obs_next_all = convert_ny(obs_next_all)
        obs = convert_ny(obs)
        obs_next = convert_ny(obs_next)

        mask_size = self.cfg.attention_mask_size
        obs_mask = obs[:, 0, :, :]
        all_obs = obs_all[:, 0, :, :]
        all_obs_next = obs_next_all[:, 0, :, :]
        direction = [
            self.cfg.direction_map[int(x)]
            for x in np.round(obs[:, 2, mask_size // 2, mask_size // 2])
        ]
        action = [self.cfg.action_map[int(x)] for x in act[:]]
        next_direction = [
            self.cfg.direction_map[int(x)]
            for x in np.round(obs_next[:, 2, mask_size // 2, mask_size // 2])
        ]
        obs_next_mask = obs_next[:, 0, :, :]

        color_list = [
            "#440154",
            "#3b528b",
            "#21918c",
            "#5ec962",
            "#fde725",
            "#f98400",
            "#d00000",
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom7", color_list, N=7)

        def show(
            cur_all_obs,
            cur_all_next,
            cur_obs,
            cur_obs_next,
            cur_direction,
            cur_action,
            cur_next_direction,
            cur_key,
            cur_step_counter,
            index,
        ):
            plt.figure(figsize=size)
            self._plot_subplot(2, 2, 1, cur_all_obs, custom_cmap, "whole map", "", shrink)
            self._plot_subplot(
                2, 2, 2, cur_all_next, custom_cmap, "whole map next", "", shrink
            )
            self._plot_subplot(
                2,
                2,
                3,
                cur_obs,
                custom_cmap,
                "mask obs",
                f"Dir: {cur_direction}  Action: {cur_action}, key:{cur_key}",
                shrink,
            )
            self._plot_subplot(
                2,
                2,
                4,
                cur_obs_next,
                custom_cmap,
                "mask obs next",
                f"Dir:{cur_next_direction}",
                shrink,
            )
            plt.tight_layout()

            if not os.path.exists(self.cfg.save_path):
                os.mkdir(self.cfg.save_path)
            save_file = os.path.join(
                self.cfg.save_path, f"Source_data_{cur_step_counter}_{index}.png"
            )
            plt.savefig(save_file)
            plt.close()

        for i in range(len(act)):
            show(
                all_obs[i],
                all_obs_next[i],
                obs_mask[i],
                obs_next_mask[i],
                direction[i],
                action[i],
                next_direction[i],
                key,
                step_counter,
                i,
            )

    def visualize_attention(
        self,
        obs,
        act,
        attentionWeight,
        obs_next,
        obs_pred,
        step_counter,
        info=None,
        size=(14, 10),
        shrink=1,
    ):
        key = info["carrying_key"][-1].item() if info is not None and "carrying_key" in info else "None"
        mask_size = self.cfg.attention_mask_size
        channel, row, col = 3, mask_size, mask_size
        obs_next_temp = obs_next.view(obs_next.shape[0], channel, row, col)

        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[-1, 0, :, :].detach().cpu().numpy() * obj_ratio
        direction = self.cfg.direction_map[
            round(obs[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)
        ]
        action = self.cfg.action_map[round(act[-1].item() * act_ratio)]
        next_direction = self.cfg.direction_map[
            round(obs_next_temp[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)
        ]
        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * obj_ratio
        pred_direction_idx = round(
            obs_pred[-1, :].reshape(channel, row, col)[2, :, :].detach().cpu().numpy().max()
            * dir_ratio
        )
        obs_pred = np.round(
            obs_pred[-1, :].reshape(channel, row, col)[0, :, :].detach().cpu().numpy()
            * obj_ratio
        )
        if len(attentionWeight.shape) == 2:
            heat_map = attentionWeight[-1, :].reshape(row, col).detach().cpu().numpy()
        elif len(attentionWeight.shape) == 3:
            heat_map = (
                attentionWeight[-1, (mask_size**2) // 2, :]
                .reshape(row, col)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            raise ValueError("Attention weight shape is not supported.")
        pre_direction = self.cfg.direction_map.get(pred_direction_idx, "Unknown")

        num_colors = 13
        custom_cmap = plt.cm.get_cmap("jet", num_colors)
        plt.figure(figsize=size)
        self._plot_subplot(
            2,
            2,
            1,
            state_image,
            custom_cmap,
            "State",
            f"State  Dir: {direction}  Action: {action}, key:{key}",
            shrink,
        )
        self._plot_subplot(2, 2, 3, heat_map, "viridis", "Attention", "Attention Heatmap", shrink)
        self._plot_subplot(
            2,
            2,
            2,
            obs_next,
            custom_cmap,
            "Next State",
            f"Next State  Dir:{next_direction}",
            shrink,
        )
        self._plot_subplot(
            2,
            2,
            4,
            obs_pred,
            custom_cmap,
            "Predicted",
            f"Pre State  Dir: {pre_direction}",
            shrink,
        )
        plt.tight_layout()

        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)
        save_file = os.path.join(self.cfg.save_path, f"Attention_{step_counter}.png")
        plt.savefig(save_file)
        plt.close()

    def _plot_subplot(self, row, col, position, data, cmap, colorbar_label, title, shrink):
        plt.subplot(row, col, position)
        im = plt.imshow(data, cmap=cmap, interpolation="nearest")
        plt.colorbar(im, shrink=shrink, label=colorbar_label)
        plt.title(title)

    def _plot(self, data, cmap, title, shrink):
        plt.imshow(data, cmap=cmap, interpolation="nearest")
        plt.colorbar(shrink=shrink, label=title)
        plt.title(title)
        plt.show()


def extract_unique_patches(layout_str: str, patch_size: int):
    lines = [list(line) for line in layout_str.strip().split("\n")]
    h, w = len(lines), len(lines[0])
    grid = np.array(lines)

    r = patch_size // 2
    unique_set = set()
    unique_patches = []

    for i in range(h):
        for j in range(w):
            if i - r < 0 or i + r >= h or j - r < 0 or j + r >= w:
                continue

            patch = grid[i - r : i + r + 1, j - r : j + r + 1]
            center = patch[r, r]
            if center != "E":
                continue

            patch_str = "\n".join("".join(row) for row in patch)
            if patch_str not in unique_set:
                unique_set.add(patch_str)
                unique_patches.append(patch_str)

    return unique_patches


def patch_to_array(patch_str):
    lines = patch_str.split("\n")
    return np.array([list(row) for row in lines])


def array_to_patch(arr):
    return "\n".join("".join(row) for row in arr)


def combine_patches_2x2(patches):
    assert len(patches) == 4
    a = patch_to_array(patches[0])
    b = patch_to_array(patches[1])
    c = patch_to_array(patches[2])
    d = patch_to_array(patches[3])

    k = a.shape[0]
    big = np.full((2 * k, 2 * k), "E", dtype=str)
    big[0 * k : 1 * k, 0 * k : 1 * k] = a
    big[0 * k : 1 * k, 1 * k : 2 * k] = b
    big[1 * k : 2 * k, 0 * k : 1 * k] = c
    big[1 * k : 2 * k, 1 * k : 2 * k] = d
    return array_to_patch(big)


def minitask_has_new_patch(selected, covered_set):
    for p in selected:
        if p not in covered_set:
            return True
    return False


def combine_patches_1x2(patches):
    assert len(patches) == 2
    a = patch_to_array(patches[0])
    b = patch_to_array(patches[1])

    k = a.shape[0]
    big = np.full((k, 2 * k), "E", dtype=str)
    big[:, 0 * k : 1 * k] = a
    big[:, 1 * k : 2 * k] = b
    return array_to_patch(big)


def generate_minitasks_until_covered(
    all_patches, patch_size, patches_per_minitask, add_agent_start=False
):
    covered = set()
    minitasks = []
    all_set = set(all_patches)

    if patches_per_minitask == 1:
        while not all_set.issubset(covered):
            remaining = list(all_set - covered)
            selected = [random.choice(remaining)]
            mt_layout = add_outer_wall(selected[0])
            minitasks.append(mt_layout)
            covered.add(selected[0])

        minitask_set = []
        for layout_map in minitasks:
            color_map = generate_color_map(layout_map)
            combined_map = combine_maps(layout_map, color_map, None)
            minitask_set.append(combined_map)
        return minitask_set

    while not all_set.issubset(covered):
        remaining = list(all_set - covered)

        if len(remaining) >= patches_per_minitask:
            selected = random.sample(remaining, patches_per_minitask)
        else:
            selected = remaining.copy()
            missing = patches_per_minitask - len(selected)
            if len(covered) == 0:
                pad = random.sample(all_patches, missing)
            else:
                pad = random.sample(list(covered), missing)
            selected += pad

        if not minitask_has_new_patch(selected, covered):
            continue

        if patches_per_minitask == 2:
            mt_layout = combine_patches_1x2(selected)
        else:
            mt_layout = combine_patches_2x2(selected)

        mt_layout = add_outer_wall(mt_layout)
        minitasks.append(mt_layout)

        mt_patches = extract_unique_patches(mt_layout, patch_size)
        covered.update(mt_patches)

    if not all_set.issubset(covered):
        print("WARNING: Not all target patches are covered!")

    minitask_set = []
    for layout_map in minitasks:
        color_map = generate_color_map(layout_map)
        combined_map = combine_maps(layout_map, color_map, None)
        minitask_set.append(combined_map)

    if add_agent_start:
        for i in range(len(minitask_set)):
            layout_map = minitask_set[i]
            layout_lines = layout_map.strip().split("\n")
            empty_positions = []
            for r, line in enumerate(layout_lines):
                for c, char in enumerate(line):
                    if char == "E":
                        empty_positions.append((r, c))

            if not empty_positions:
                raise ValueError("No empty cell found to place the agent.")

            start_r, start_c = random.choice(empty_positions)
            row_list = list(layout_lines[start_r])
            row_list[start_c] = "S"
            layout_lines[start_r] = "".join(row_list)
            updated_layout = "\n".join(layout_lines)
            minitask_set[i] = updated_layout

    return minitask_set
