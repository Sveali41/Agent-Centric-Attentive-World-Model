import torch
import numpy as np

def map_to_nearest_value_support(tensor, valid_values):
    valid_values = torch.tensor(valid_values, dtype=torch.float32).to(tensor.device)
    tensor = tensor.unsqueeze(-1)  # Add a dimension to compare with valid values
    differences = torch.abs(tensor - valid_values)  # Calculate differences
    indices = torch.argmin(differences, dim=-1)  # Get index of nearest value
    nearest_values = valid_values[indices]  # Get nearest values using indices
    return nearest_values

def extract_masked_state_support(state, agent_position_yx, mask_size, pad_value=None):
    """
    state dimensions: (channels, rows, cols)
    """
    channels, rows, cols = state.shape
    y, x = agent_position_yx
    half = mask_size // 2
    if pad_value is None:
        margin_data = state[:, 0, 0]
    else:
        margin_data = np.full(channels, pad_value, dtype=state.dtype)
    region = np.tile(margin_data.reshape(channels, 1, 1),
                     (1, mask_size, mask_size))

    src_slice_y = slice(max(y - half, 0), min(y + half + 1, rows))
    src_slice_x = slice(max(x - half, 0), min(x + half + 1, cols))

    dest_slice_y = slice(max(0, half - y), max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)))
    dest_slice_x = slice(max(0, half - x), max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)))

    # 将 state 中的有效区域复制到预填充区域中
    region[:, dest_slice_y, dest_slice_x] = state[:, src_slice_y, src_slice_x]
    return region

