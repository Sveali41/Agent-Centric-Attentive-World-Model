"""Small, dependency-light helpers for MiniGrid map representations."""

import random
import re

import torch
from minigrid.core.constants import IDX_TO_COLOR


def generate_obj_map(layout, map_dict):
    reverse_map_dict = {v: k for k, v in map_dict.items()}
    return "\n".join(
        "".join(reverse_map_dict.get(value.item()) for value in row)
        for row in layout[0]
    )


def interpret_color_map(color_layout, color_map):
    if hasattr(color_layout, "detach"):
        color_layout = color_layout.detach().cpu()
    color_name_to_char = {
        "red": "R", "green": "G", "blue": "B",
        "purple": "M", "yellow": "Y", "grey": "W",
    }
    color_name_to_char.update({
        value: key for key, value in color_map.items()
        if value not in color_name_to_char
    })
    height, width = color_layout.shape
    return "\n".join(
        "".join(color_name_to_char.get(IDX_TO_COLOR[int(idx)], "?") for idx in color_layout[row])
        for row in range(height)
    )


def generate_color_map(layout_string):
    object_to_color_map = {
        "W": "W", "E": "E", "G": "G", "S": "E",
        "K": "Y", "D": "Y", "L": "L", "O": "Y",
    }
    return layout_string.translate(str.maketrans(object_to_color_map))


def layout_to_string(layout):
    return "\n".join("".join(row) for row in layout)


def combine_maps(layout, color, file_path=None):
    combined = layout.strip() + "\n\n" + color.strip()
    if file_path is not None:
        with open(file_path, "w") as file:
            file.write(combined)
    return combined


def add_outer_wall(layout_string):
    rows = layout_string.strip().split("\n")
    wall_row = "W" * (len(rows[0]) + 2)
    return "\n".join([wall_row, *("W" + row + "W" for row in rows), wall_row])
