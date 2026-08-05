import os
from typing import Optional, Tuple
import numpy as np

from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Floor, Ball, Key, Box, Door, Goal, Lava
from minigrid.manual_control import ManualControl

from custom_objects import Teleporter, WorldObj
from teleport_2 import TeleportBaseEnv

def char_to_color(char: str) -> Optional[str]:
    """
    Maps a single character to a color name supported by MiniGrid objects.
    """
    color_map = {'R': 'red', 'G': 'green', 'B': 'blue',
                 'Y': 'yellow', 'M': 'magenta', 'C': 'cyan'}
    return color_map.get(char.upper(), None)

def char_to_object_teleport(char: str, color: Optional[str] = None) -> Optional[WorldObj]:
    """
    Maps a character (and its associated color) to a MiniGrid object, with Teleport Support.
    'T' is an active teleporter.
    'U' is an inactive teleporter (often used as destination).
    """
    obj_map = {
        'W': lambda: Wall(),
        'F': lambda: Floor(),
        'B': lambda: Ball(color),
        'K': lambda: Key(color),
        'X': lambda: Box(color),
        'D': lambda: Door(color, is_locked=True),
        'G': lambda: Goal(),
        'L': lambda: Lava(),
        'O': lambda: Door(color, is_locked=False),
        'T': lambda: Teleporter(active=True),
        'U': lambda: Teleporter(active=False)
    }
    constructor = obj_map.get(char.upper(), None)
    return constructor() if constructor else None

class CustomTeleportEnv(TeleportBaseEnv):
    """
    A custom Teleport environment that can load layout from text files/strings.
    Similar to CustomMiniGridEnv but inherits from TeleportBaseEnv to natively support
    Teleporter step mechanics.
    """
    def __init__(
            self,
            txt_file_path: Optional[str] = None,
            layout_str: Optional[str] = None,
            color_str: Optional[str] = None,
            teleporter_configs: Optional[dict] = None, # Configuration dict for active teleporters
            size: Optional[int] = None,
            agent_start_pos: Optional[tuple[int, int]] = None,
            agent_start_dir: Optional[int] = None,
            custom_mission: str = "Explore and use teleporters.",
            max_steps: Optional[int] = None,
            **kwargs,
    ) -> None:
        
        self.txt_file_path = txt_file_path
        self.layout_str = layout_str
        self.color_str = color_str
        self.teleporter_configs = teleporter_configs or {}
        self.s_positions = []

        if size is None:
            if txt_file_path:
                self.height, self.width = self.determine_layout_size_from_file()
            elif layout_str and color_str:
                self.height, self.width = self.determine_layout_size_from_strings()
            else:
                raise ValueError("Either 'txt_file_path' or both 'layout_str' and 'color_str' must be provided.")
        else:
            self.height, self.width = size, size

        # Ensure max_steps is set
        if max_steps is None:
            max_steps = 4 * self.width ** 2

        self.rand_agent_start_pos = agent_start_pos is None
        self.agent_start_pos = agent_start_pos
        self.rand_agent_start_dir = agent_start_dir is None
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=lambda: custom_mission)

        # Call TeleportBaseEnv constructor
        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )
        
        self.mission = custom_mission

    def determine_layout_size_from_file(self) -> Tuple[int, int]:
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            layout_lines = sections[0].strip().split('\n')
            height = len(layout_lines)
            width = max(len(line) for line in layout_lines)
            return height, width

    def determine_layout_size_from_strings(self) -> Tuple[int, int]:
        layout_lines = self.layout_str.strip().split('\n')
        height = len(layout_lines)
        width = max(len(line) for line in layout_lines)
        return height, width

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        
        if self.txt_file_path:
            self.read_layout_from_file()
        else:
            self.read_layout_from_strings()

        # Place agent based on 'S' tag or randomly on empty floor
        if self.s_positions:
            self.agent_start_pos = self.s_positions[0]
            self.agent_dir = self.agent_start_dir if not self.rand_agent_start_dir else np.random.randint(0, 4)
            self.agent_pos = self.agent_start_pos
        else:
            empty_positions = [(x, y) for x in range(self.width) for y in range(self.height) 
                            if self.grid.get(x, y) is None]
            if not empty_positions:
                raise ValueError("No empty position found marked with 'E' or equivalent.")

            self.agent_start_pos = empty_positions[np.random.randint(0, len(empty_positions))]
            if self.rand_agent_start_dir:
                self.agent_start_dir = np.random.randint(0, 4)

            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
            
        self.start_pos = self.agent_start_pos
        self.start_dir = self.agent_start_dir

        # Apply specific configs to Teleporters (destinations and probabilities)
        for (x, y), config in self.teleporter_configs.items():
            obj = self.grid.get(x, y)
            if obj is not None and obj.type == "teleporter" and obj.is_active:
                if "end_locations" in config:
                    obj.end_locations = config["end_locations"]
                if "end_probabilities" in config:
                    obj.end_probabilities = config["end_probabilities"]

    def read_layout_from_file(self) -> None:
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            self.layout_str = sections[0].strip()
            self.color_str = sections[1].strip()

            layout_lines = self.layout_str.split('\n')
            color_lines = self.color_str.split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and color matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    if char.upper() == 'E':
                        continue # leave None
                    if char.upper() == 'S':
                        self.s_positions.append((x, y))
                        color = char_to_color(color_char)
                        obj = Floor(color) if color else Floor()
                        self.put_obj(obj, x, y)
                        continue
                    
                    color = char_to_color(color_char)
                    obj = char_to_object_teleport(char, color)
                    if obj:
                        self.put_obj(obj, x, y)

    def read_layout_from_strings(self) -> None:
        original_layout_str = self.layout_str.strip()
        original_color_str = self.color_str.strip()

        layout_lines = original_layout_str.split('\n')
        color_lines = original_color_str.split('\n')

        self.layout_str = original_layout_str
        self.color_str = original_color_str

        if len(layout_lines) != len(color_lines):
            raise ValueError("Layout and color strings must have the same number of lines.")

        for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
            if len(layout_line) != len(color_line):
                raise ValueError("Each layout line must correspond to a color line of the same length.")
            for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                if char.upper() == 'E':
                    continue # leave None
                if char.upper() == 'S':
                    self.s_positions.append((x, y))
                    color = char_to_color(color_char)
                    obj = Floor(color) if color else Floor()
                    self.put_obj(obj, x, y)
                    continue
                
                color = char_to_color(color_char)
                obj = char_to_object_teleport(char, color)
                if obj:
                    self.put_obj(obj, x, y)


if __name__ == "__main__":
    import textwrap

    layout_string = textwrap.dedent("""
        WWWWWWWWW
        WSFEEEEEW
        WEWTWEWEW
        WEFEEEEGW
        WWWWWWWWW
    """).strip()

    color_string = textwrap.dedent("""
        WWWWWWWWW
        WGGEEEEEW
        WEWBWEWEW
        WEFEEEEGW
        WWWWWWWWW
    """).strip()

    # Define behavior for the Teleporter at coordinate (x=3, y=2)
    # Target coordinate to teleport to: (7, 3), which is adjacent to the Goal.
    teleporter_configs = {
        (3, 2): {
            "end_locations": [(7, 3)],
            "end_probabilities": [1.0]
        }
    }

    env = CustomTeleportEnv(
        layout_str=layout_string,
        color_str=color_string,
        teleporter_configs=teleporter_configs,
        custom_mission="Test custom teleporter map.",
        render_mode="human"
    )

    env.reset()
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
