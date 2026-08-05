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
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = None,
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
            custom_mission (str): Custom mission description.
            max_steps (Optional[int]): Maximum number of steps in an episode.
            **kwargs: Additional keyword arguments for MiniGridEnv.
        """
        self.txt_file_path = txt_file_path
        self.layout_str = layout_str
        self.color_str = color_str
        self.s_positions = []  # List to store positions of 'S'

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

        # Determine the starting position and direction of the agent
        self.rand_agent_start_pos = agent_start_pos is None
        self.agent_start_pos = agent_start_pos
        self.rand_agent_start_dir = agent_start_dir is None
        self.agent_start_dir = agent_start_dir

        # Mission or objects within the environment
        self.mission = custom_mission

    def determine_layout_size_from_file(self) -> Tuple[int, int]:
        """
        Reads the layout from the file to determine the environment's size based on its width and height.

        Returns:
            Tuple[int, int]: The height and width of the layout.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            layout_lines = sections[0].strip().split('\n')
            # Set the environment's width and height based on the layout
            height = len(layout_lines)
            width = max(len(line) for line in layout_lines)
            return height, width

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
        if self.txt_file_path:
            self.read_layout_from_file()
        else:
            self.read_layout_from_strings()

        # If 'S' positions are specified in the layout, use them as the agent's starting positions
        if self.s_positions:
            # For simplicity, use the first 'S' position as the agent's start
            self.agent_start_pos = self.s_positions[0]
            self.agent_dir = self.agent_start_dir if not self.rand_agent_start_dir else np.random.randint(0, 4)
            self.agent_pos = self.agent_start_pos
        else:
            # Find all 'E' positions (empty spots)
            empty_positions = [(x, y) for x in range(self.width) for y in range(self.height) 
                            if self.grid.get(x, y) is None]
            
            if not empty_positions:
                raise ValueError("No empty position found marked with 'E'.")

            # Randomly assign a starting position within the 'E' positions
            self.agent_start_pos = empty_positions[np.random.randint(0, len(empty_positions))]

            # Set direction (could be random or predefined)
            if self.rand_agent_start_dir:
                self.agent_start_dir = np.random.randint(0, 4)

            # Set the agent's position and direction
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
            
        self.start_pos = self.agent_start_pos
        self.start_dir = self.agent_start_dir

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
            x = np.random.randint(x1, x2 + 1)
            y = np.random.randint(y1, y2 + 1)

            if self.grid.get(x, y) is None:
                return (x, y)
            attempts += 1
        raise ValueError("No empty position found in the specified area after multiple attempts.")

    def read_layout_from_file(self) -> None:
        """
        Parses the text file specified by 'txt_file_path' to set the objects in the environment's grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            layout_lines = sections[0].strip().split('\n')
            color_lines = sections[1].strip().split('\n')

            # Save both the original string and the split lines
            self.layout_str = sections[0].strip()       # <-- the real multiline string
            self.color_str = sections[1].strip()

            layout_lines = self.layout_str.split('\n')  # for internal parsing
            color_lines = self.color_str.split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and color matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    if char.upper() == 'S':
                        # Record 'S' position as agent's start position
                        self.s_positions.append((x, y))
                        # Set 'S' as Floor with specified color
                        color = char_to_color(color_char)
                        obj = Floor()
                        self.grid.set(x, y, obj)
                        continue  # 'S' is handled
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
                    # Record 'S' position as agent's start position
                    self.s_positions.append((x, y))
                    # Set 'S' as Floor with specified color
                    color = char_to_color(color_char)
                    obj = Floor()
                    self.grid.set(x, y, obj)
                    continue  # 'S' is handled
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
        # Use random.randint to select an index and then retrieve the position
        index = np.random.randint(0, len(empty_positions))
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
    # path = Paths()
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    level_path = os.path.join(BASE_DIR, "trainer", "level", "env1_keydoor.txt")
    env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=level_path, 
                                        custom_mission="Find the key and open the door.",
                                        max_steps=5000, render_mode='human'))
    env.reset()
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
    