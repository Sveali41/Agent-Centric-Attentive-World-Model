from typing import Optional, Tuple
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from PIL import Image, ImageDraw
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import numpy as np  # Ensure numpy is imported
import textwrap
from pathlib import Path
from gymnasium import spaces


# Internal outcome marker for a dropped action. This is not a MiniGrid/native
# action and is never exposed through the policy action space.
INTERNAL_NOOP_ACTION = -1


def char_to_color(char: str) -> Optional[str]:
    """
    Maps a single character to a color name supported by MiniGrid objects.

    Args:
        char (str): A character representing a color.

    Returns:
        Optional[str]: The name of the color, or None if the character is not recognized.
    """
    color_map = {
        'R': 'red', 'G': 'green', 'B': 'blue', 'Y': 'yellow', 
        'M': 'purple', 'W': 'grey', 'L': 'red', 'E': 'grey', 'S': 'grey'
    }
    return color_map.get(char.upper(), None)


def char_to_object(char: str, color: Optional[str] = None) -> Optional[WorldObj]:
    """
    Maps a character (and its associated color) to a MiniGrid object.

    Args:
        char (str): A character representing an object type.
        color (Optional[str]): The color of the object.

    Returns:
        Optional[WorldObj]: The MiniGrid object corresponding to the character and color, or None if unrecognized.
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
        'O': lambda: Door(color, is_locked=False)
    }
    constructor = obj_map.get(char.upper(), None)
    return constructor() if constructor else None


class CustomMiniGridEnv(MiniGridEnv):
    """
    A custom MiniGrid environment that can load its layout and object properties from a text file or directly from strings.

    Attributes:
        txt_file_path (Optional[str]): Path to the text file containing the environment layout.
        layout_str (Optional[str]): String representing the environment layout.
        color_str (Optional[str]): String representing the environment colors.
        layout_size (int): The size of the environment, either specified or determined from the input.
        agent_start_pos (tuple[int, int]): Starting position of the agent.
        agent_start_dir (int): Initial direction the agent is facing. None for random direction
        mission (str): Custom mission description.
    """

    def __init__(
            self,
            txt_file_path: Optional[str] = None,
            layout_str: Optional[str] = None,
            color_str: Optional[str] = None,
            size: Optional[int] = None,
            agent_start_pos: Optional[tuple[int, int]] = None,  # Allow None for random initialization
            agent_start_dir: Optional[int] = None,  # Allow None for random initialization
            replace_start_with_empty: bool = False,
            initial_carrying_key_color: Optional[str] = None,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = None,
            stochastic_enabled: bool = False,
            move_failure_prob: float = 0.2,
            **kwargs,
    ) -> None:
        """
        Initializes the custom environment.

        If 'txt_file_path' is provided, it reads the layout from the file.
        Otherwise, it uses 'layout_str' and 'color_str' for layout and colors.

        Args:
            txt_file_path (Optional[str]): Path to the text file containing the environment layout.
            layout_str (Optional[str]): String representing the environment layout.
            color_str (Optional[str]): String representing the environment colors.
            size (Optional[int]): The size of the environment grid. If not provided, determined from input.
            agent_start_pos (Optional[tuple[int, int]]): Starting position of the agent.
            agent_start_dir (Optional[int]): Initial direction the agent is facing. If None, random direction.
            replace_start_with_empty (bool): Treat layout ``S`` cells as ``E``.
                Intended for random data collection; policy environments keep
                the default and therefore still start from ``S``.
            initial_carrying_key_color (Optional[str]): Restore this one carried
                key on every reset; ``None`` starts with an empty carrying slot.
            custom_mission (str): Custom mission description.
            max_steps (Optional[int]): Maximum number of steps in an episode.
            stochastic_enabled (bool): Whether navigation actions can fail.
            move_failure_prob (float): Probability that a left, right, or
                forward action becomes a no-op.
            **kwargs: Additional keyword arguments for MiniGridEnv.
        """
        self.txt_file_path = txt_file_path
        self.layout_str = layout_str
        self.color_str = color_str
        # A policy rollout resets this same environment many times.  Keep the
        # immutable text layout in memory so reset does not repeatedly read and
        # split the target file; objects are still reconstructed on every
        # reset, preserving MiniGrid's per-episode object state semantics.
        self._cached_file_sections = None
        self.s_positions = []  # List to store positions of 'S'
        self.replace_start_with_empty = bool(replace_start_with_empty)
        self.initial_carrying_key_color = initial_carrying_key_color
        self.stochastic_enabled = bool(stochastic_enabled)
        try:
            self.move_failure_prob = float(move_failure_prob)
        except (TypeError, ValueError) as exc:
            raise ValueError("move_failure_prob must be a number in [0, 1].") from exc
        if not 0.0 <= self.move_failure_prob <= 1.0:
            raise ValueError("move_failure_prob must be in [0, 1].")

        # Determine the size of the environment if not provided
        if size is None:
            if txt_file_path:
                self.height, self.width = self.determine_layout_size_from_file()
            elif layout_str and color_str:
                self.height, self.width = self.determine_layout_size_from_strings()
            else:
                raise ValueError("Either 'txt_file_path' or both 'layout_str' and 'color_str' must be provided.")
        else:
            self.height, self.width = size, size  # Assume square grid if size is provided

        # Initialize the MiniGrid environment with the determined size
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            see_through_walls=False,
            max_steps=max_steps or 4 * self.width ** 2,
            width=self.width,
            height=self.height,
            **kwargs,
        )
        # The project contract exposes exactly six MiniGrid actions. Native
        # ``done`` remains unavailable to callers; stochastic failures are
        # represented as unchanged transitions inside ``step``.
        self.action_space = spaces.Discrete(6)

        # Determine the starting position and direction of the agent
        self.rand_agent_start_pos = agent_start_pos is None
        self.agent_start_pos = agent_start_pos
        self.rand_agent_start_dir = agent_start_dir is None
        self.agent_start_dir = agent_start_dir

        # Mission or objects within the environment
        self.mission = custom_mission

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.carrying = (
            Key(self.initial_carrying_key_color)
            if self.initial_carrying_key_color is not None
            else None
        )
        # The symbolic grid does not encode carrying, but regenerating keeps
        # this reset contract correct for wrappers that extend gen_obs().
        return self.gen_obs(), info

    def step(self, action):
        """Execute one action, optionally dropping navigation actions.

        A dropped action follows the environment's private no-op path without
        invoking MiniGrid's native ``done`` action.
        """
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError(
                f"Invalid MiniGrid action {action_id}; expected action ids 0..5"
            )
        requested_action = self.actions(action_id)
        executed_action = requested_action
        action_failed = False
        navigation_actions = {
            self.actions.left,
            self.actions.right,
            self.actions.forward,
        }
        if (
            self.stochastic_enabled
            and requested_action in navigation_actions
            and self.np_random.random() < self.move_failure_prob
        ):
            executed_action = INTERNAL_NOOP_ACTION
            action_failed = True

        if action_failed:
            obs, reward, terminated, truncated, info = self._step_noop()
        else:
            obs, reward, terminated, truncated, info = super().step(executed_action)
        info = dict(info)
        info.update(
            requested_action=int(requested_action),
            executed_action=int(executed_action),
            action_failed=action_failed,
            stochastic_enabled=self.stochastic_enabled,
            move_failure_prob=self.move_failure_prob,
        )
        return obs, reward, terminated, truncated, info

    def _step_noop(self):
        """Advance time and observe without invoking a native action."""
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = self.step_count >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self.gen_obs(), reward, terminated, truncated, {}

    def determine_layout_size_from_file(self) -> Tuple[int, int]:
        """
        Reads the layout from the file to determine the environment's size based on its width and height.

        Returns:
            Tuple[int, int]: The height and width of the layout.
        """
        layout_str, _ = self._file_layout_sections()
        layout_lines = layout_str.split('\n')
        # Set the environment's width and height based on the layout
        height = len(layout_lines)
        width = max(len(line) for line in layout_lines)
        return height, width

    def _file_layout_sections(self) -> Tuple[str, str]:
        """Load and cache the immutable layout text for this environment."""
        if self._cached_file_sections is None:
            with open(self.txt_file_path, 'r') as file:
                sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")
            self._cached_file_sections = (sections[0].strip(), sections[1].strip())
        return self._cached_file_sections

    def determine_layout_size_from_strings(self) -> Tuple[int, int]:
        """
        Determines the environment's size based on layout and color strings.

        Returns:
            Tuple[int, int]: The height and width of the layout.
        """
        layout_lines = self.layout_str.strip().split('\n')
        color_lines = self.color_str.strip().split('\n')
        height = len(layout_lines)
        width = max(len(line) for line in layout_lines)
        return height, width

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generates the grid for the environment based on the layout specified in the file or strings.
        """
        self.grid = Grid(width, height)
        # ``_gen_grid`` runs on every reset.  Keep only the spawn markers for
        # the current grid; accumulating the same ``S`` position on each reset
        # needlessly grows memory and makes the cached-reset path slower.
        self.s_positions.clear()
        if self.txt_file_path:
            self.read_layout_from_file()
        else:
            self.read_layout_from_strings()

        # An explicit start position always takes precedence.  Otherwise an S
        # marker is required outside collection mode; only uniform collection
        # (replace_start_with_empty=True) samples a random empty start.
        if not self.rand_agent_start_pos:
            start_pos = self.agent_start_pos
        elif self.s_positions:
            start_pos = self.s_positions[0]
        elif self.replace_start_with_empty:
            # Find all 'E' positions (empty spots)
            empty_positions = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if self.grid.get(x, y) is None
            ]
            if not empty_positions:
                raise ValueError("No empty position found marked with 'E'.")
            start_pos = empty_positions[self.np_random.integers(len(empty_positions))]
        else:
            raise ValueError(
                "No start position: provide agent_start_pos or include an 'S' "
                "marker (random starts require replace_start_with_empty=True)."
            )

        start_dir = (
            int(self.np_random.integers(0, 4))
            if self.rand_agent_start_dir
            else self.agent_start_dir
        )
        self.agent_pos = start_pos
        self.agent_dir = start_dir
        self.start_pos = start_pos
        self.start_dir = start_dir

    def find_empty_position_in_area(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        """
        Finds an empty position in the defined area (x1, y1) to (x2, y2).

        Args:
            x1 (int): Top-left x-coordinate.
            y1 (int): Top-left y-coordinate.
            x2 (int): Bottom-right x-coordinate.
            y2 (int): Bottom-right y-coordinate.

        Returns:
            Tuple[int, int]: Coordinates of an empty position.
        """
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            # Randomly pick a position within the area
            x = self.np_random.integers(x1, x2 + 1)
            y = self.np_random.integers(y1, y2 + 1)

            if self.grid.get(x, y) is None:
                return (x, y)
            attempts += 1
        raise ValueError("No empty position found in the specified area after multiple attempts.")

    def read_layout_from_file(self) -> None:
        """
        Parses the text file specified by 'txt_file_path' to set the objects in the environment's grid.
        """
        layout_str, color_str = self._file_layout_sections()
        # Save both the original string and the split lines.
        self.layout_str = layout_str
        self.color_str = color_str
        layout_lines = self.layout_str.split('\n')
        color_lines = self.color_str.split('\n')

        if len(layout_lines) != len(color_lines) or any(
                len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
            raise ValueError("Object and color matrices must have the same size.")

        for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
            for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                if char.upper() == 'S':
                    if self.replace_start_with_empty:
                        # Collection-only mode: S behaves exactly like E.
                        # No object and no fixed spawn marker are created.
                        continue
                    # Record 'S' position as agent's start position
                    self.s_positions.append((x, y))
                    # S is a spawn marker, not a grid object. Keeping its
                    # cell empty makes the physical layout identical to
                    # collection mode, where S is parsed as E.
                    continue
                color = char_to_color(color_char)
                obj = char_to_object(char, color)
                if obj:
                    self.grid.set(x, y, obj)  # Place the object on the grid

    def read_layout_from_strings(self) -> None:
        """
        Parses the provided layout and color strings to set the objects in the environment's grid.
        """

        # --- KEEP the original multiline strings ---
        original_layout_str = self.layout_str.strip()
        original_color_str = self.color_str.strip()

        # --- Convert to lists of lines for internal processing ---
        layout_lines = original_layout_str.split('\n')
        color_lines = original_color_str.split('\n')

        # --- Store the original strings back into class variables ---
        self.layout_str = original_layout_str
        self.color_str = original_color_str

        if len(layout_lines) != len(color_lines):
            raise ValueError("Layout and color strings must have the same number of lines.")

        for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
            if len(layout_line) != len(color_line):
                raise ValueError("Each layout line must correspond to a color line of the same length.")
            for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                if char.upper() == 'S':
                    if self.replace_start_with_empty:
                        continue
                    # Record 'S' position as agent's start position
                    self.s_positions.append((x, y))
                    # S fixes the spawn but otherwise represents an empty cell.
                    continue
                color = char_to_color(color_char)
                obj = char_to_object(char, color)
                if obj:
                    self.grid.set(x, y, obj)  # Place the object on the grid

    def find_empty_position(self) -> Tuple[int, int]:
        """
        Finds an empty position on the grid where there is no object.

        Returns:
            Tuple[int, int]: The coordinates of an empty position.
        """
        empty_positions = [(x, y) for x in range(self.width) for y in range(self.height)
                           if self.grid.get(x, y) is None]
        if not empty_positions:
            raise ValueError("No empty position available on the grid.")
        index = self.np_random.integers(len(empty_positions))
        return empty_positions[index]
    
    def get_agent_position(self, obs=None):
        """
        Return the exact (y, x) position of the player in the MiniGrid environment.
        We return (y, x) so that row corresponds to y and col corresponds to x.
        """
        if getattr(self, "agent_pos", None) is not None:
            # agent_pos is typically (col, row) or (x, y). We return (y, x).
            x, y = self.agent_pos
            return np.array([y, x])
        return np.array([-1, -1])
    



if __name__ == "__main__":
    ## 1. generate env from string
    # # Example usage of the CustomMiniGridEnv class with file input
    # # env_file_based = FullyObsWrapper(CustomMiniGridEnv(
    # #     txt_file_path=path.LEVEL_FILE_Rmax,
    # #     custom_mission="Find the key and open the door.",
    # #     render_mode="human"
    # # ))

    # # Example usage of the CustomMiniGridEnv class with direct string input
    # layout_string = textwrap.dedent("""
    #     WWWWWWWWW
    #     WEEEEEEOW
    #     WEWEWEWGW
    #     WEEEEEEEW
    #     WWWWWWWWW
    # """).strip()

    # color_string = textwrap.dedent("""
    #     WWWWWWWWW
    #     WEEEEEEYW
    #     WEWEWEWEW
    #     WEEEEEEEW
    #     WWWWWWWWW
    # """).strip()

    # env_string_based = FullyObsWrapper(CustomMiniGridEnv(
    #     layout_str=layout_string,
    #     color_str=color_string,
    #     custom_mission="Navigate to the start position.",
    #     render_mode="human"
    # ))
 
    # # Choose which environment to run
    # selected_env = env_string_based  # Change to env_string_based to use string input

    # # selected_env.reset()
    # obs = selected_env.reset()[0]
    # obs_next, reward, done, trunc, _ = selected_env.step(0)
    # manual_control = ManualControl(selected_env)  # Allows manual control for testing and visualization
    # manual_control.start()  # Start the manual control interface



    # 2. generate env from text file
    project_root = Path(__file__).resolve().parents[2]
    level_path = project_root / "level" / "minigrid" / "env1_keydoor.txt"
    env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=level_path, 
                                        custom_mission="Find the key and open the door.",
                                        max_steps=5000, render_mode='human'))
    env.reset()
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
