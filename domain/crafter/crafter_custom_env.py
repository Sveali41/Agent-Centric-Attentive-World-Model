"""
CrafterSymbolicEnv (Tensor + Render)
-----------------------------------
Outputs symbolic (H, W, 2) grid plus Crafter's RGB render.
- C0 = object ID
- C1 = direction ID (0–4)
"""

import numpy as np
import gym
from gym import spaces


# ---------------------------------------------------------------------
# 1. Object / Tile IDs
# ---------------------------------------------------------------------
# Materials (tile IDs 0-12), matching original Crafter engine._mat_ids:
# 0=None/empty, 1=water, 2=grass, 3=stone, 4=path, 5=sand, 6=tree,
# 7=lava, 8=coal, 9=iron, 10=diamond, 11=table, 12=furnace
# Entities (IDs 13-19), starting after materials:
# 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
# Total object classes = 20 (0-19)

ENTITY_ID = {
    "player":   13,
    "cow":      14,
    "zombie":   15,
    "skeleton": 16,
    "arrow":    17,
    "plant":    18,
    "fence":    19,
}

DIR_TO_ID = {
    (0, -1): 1,   # up
    (0,  1): 2,   # down
    (-1, 0): 3,   # left
    (1,  0): 4,   # right
}


# ---------------------------------------------------------------------
# 2. Adapter for old/new Crafter versions
# ---------------------------------------------------------------------

def get_engine(env):
    inner = getattr(env, "_env", env)
    if hasattr(inner, "engine"):
        return inner.engine
    if hasattr(inner, "_world"):
        class LegacyAdapter:
            def __init__(self, env):
                self._env = env
                self._world = env._world
            @property
            def tile_map(self): return self._world._mat_map
            @property
            def entities(self): return self._world.objects
            @property
            def player(self): return self._env._player
            @property
            def world_shape(self): return self._world._mat_map.shape
        return LegacyAdapter(inner)
    raise AttributeError("Unsupported Crafter API version.")


# ---------------------------------------------------------------------
# 3. Extract (H, W, 2) symbolic tensor
# ---------------------------------------------------------------------

def extract_tensor_grid(env):
    engine = get_engine(env)
    # Crafter engine returns (Width, Height) i.e. (Col, Row)
    W, H = engine.world_shape
    # Return (Row, Col, Chan) standard format
    grid = np.zeros((H, W, 2), dtype=np.int32)

    mat_map = engine.tile_map
    for x in range(W):
        for y in range(H):
            # mat_map is indexed (x, y) where x=col, y=row
            tile_val = int(mat_map[x, y]) if x < mat_map.shape[0] and y < mat_map.shape[1] else 0
            grid[y, x, 0] = max(0, min(tile_val, 12))

    for ent in getattr(engine, "entities", []):
        ex, ey = int(ent.pos[0]), int(ent.pos[1])
        if not (0 <= ex < W and 0 <= ey < H):
            continue
        kind = type(ent).__name__.lower()
        # grid[row, col] -> grid[ey, ex]
        grid[ey, ex, 0] = ENTITY_ID.get(kind, 0)
        if hasattr(ent, "facing"):
            dir_id = DIR_TO_ID.get(tuple(int(v) for v in ent.facing), 0)
            grid[ey, ex, 1] = dir_id

    px, py = map(int, engine.player.pos)
    # grid[row, col] -> grid[py, px]
    grid[py, px, 0] = ENTITY_ID["player"]
    if hasattr(engine.player, "facing"):
        dir_id = DIR_TO_ID.get(tuple(int(v) for v in engine.player.facing), 0)
        grid[py, px, 1] = dir_id

    return grid



# ---------------------------------------------------------------------
# 4. CrafterSymbolicEnv
# ---------------------------------------------------------------------

class CrafterSymbolicEnv(gym.Env):
    """Crafter environment returning both symbolic tensor and RGB render."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, reward=False, seed=0):
        super().__init__()
        import crafter
        try:
            self.env = crafter.Env(reward=reward, seed=seed, render_mode="rgb_array")
        except TypeError:
            self.env = crafter.Env(reward=reward, seed=seed)

        engine = get_engine(self.env)
        H, W = engine.world_shape

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=20, shape=(H, W, 2), dtype=np.int32),  # max ID=19
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "info": spaces.Dict({
                "health": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "food": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "drink": spaces.Box(0, 10, shape=(), dtype=np.float32),
                "energy": spaces.Box(0, 10, shape=(), dtype=np.float32),
            }),
        })
        self.action_space = self.env.action_space

    # -----------------------------------------------------------
    # Observation extraction
    # -----------------------------------------------------------
    def _extract_obs(self):
        grid = extract_tensor_grid(self.env)
        try:
            rgb = self.env.render(mode="rgb_array")
        except TypeError:
            rgb = self.env.render()
        engine = get_engine(self.env)
        player = engine.player
        info = {
            "health": float(player.health),
            "food": float(player.inventory.get("food", 0)),
            "drink": float(player.inventory.get("drink", 0)),
            "energy": float(player.inventory.get("energy", 0)),
        }
        return {"grid": grid, "rgb": rgb, "info": info}

    def reset(self, **kwargs):
        self.env.reset()
        return self._extract_obs()

    def step(self, action):
        _, reward, done, info_env = self.env.step(action)
        obs = self._extract_obs()
        info_env.update(obs["info"])
        return obs, reward, done, info_env

    def render(self, mode="human"):
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


import numpy as np
import gym
from gym import spaces
import crafter
from crafter.engine import World
from crafter import constants, objects


# ------------------------------------------------------------
# 1. Character → Material / Entity mapping
# ------------------------------------------------------------
CHAR_TO_TILE = {
    '.': 'grass',
    'G': 'grass',
    'W': 'water',
    'T': 'tree',
    'R': 'stone',
    'C': 'coal',
    'I': 'iron',
    'O': 'diamond',  # use diamond as placeholder for gold
    'L': 'lava',
    'P': 'path',
    'S': 'sand',
    'X': 'table',
    'U': 'furnace',
}

CHAR_TO_ENTITY = {
    'A': objects.Player,
    'M': objects.Cow,
    'Z': objects.Zombie,
    'K': objects.Skeleton,
    't': objects.Plant,
    'F': objects.Fence,
}


# ------------------------------------------------------------
# 2. Build Crafter world from character grid (safe version)
# ------------------------------------------------------------
def make_world_from_chars(char_grid, seed=0):
    """Safe version: Create a Crafter world from a character grid, ensuring material ID alignment."""
    H, W = char_grid.shape

    # ---- 1. Create a temporary world to read official material mappings ----
    # Crafter automatically adds [None] + materials to align ID=0 as empty
    from crafter.engine import World
    from crafter import constants
    dummy_world = World(area=(1, 1), materials=list(constants.materials), chunk_size=(12, 12))

    # Official internal mapping (includes None)
    MATERIAL_NAME_TO_ID = dummy_world._mat_ids
    # e.g. {None: 0, 'water': 1, 'grass': 2, 'stone': 3, ...}

    # ---- 2. Create the actual world with the same structure ----
    # area should be (Width, Height) -> (Cols, Rows)
    world = World(area=(W, H), materials=list(constants.materials), chunk_size=(12, 12))
    world.daylight = 1.0

    # ---- 3. Fill the material map from character layout ----
    for y in range(H):    # Row index
        for x in range(W): # Column index
            ch = char_grid[y, x]
            mat_name = CHAR_TO_TILE.get(ch, 'grass')
            mat_id = MATERIAL_NAME_TO_ID.get(mat_name, MATERIAL_NAME_TO_ID['grass'])
            world._mat_map[x, y] = mat_id


    # ---- 4. Place the player ----
    player = None
    for y in range(H):
        for x in range(W):
            if char_grid[y, x] == 'A':
                player = objects.Player(world, (x, y))
                world.add(player)
                break
        if player:
            break
    if player is None:
        raise ValueError("No player 'A' found in the layout.")

    # ---- 5. Place other entities (cow, zombie, etc.) ----
    for y in range(H):
        for x in range(W):
            ch = char_grid[y, x]
            if ch in CHAR_TO_ENTITY and ch != 'A':
                cls = CHAR_TO_ENTITY[ch]
                # Zombies and Skeletons need a player reference
                if cls.__name__ in ["Zombie", "Skeleton"]:
                    obj = cls(world, (x, y), player)
                else:
                    obj = cls(world, (x, y))
                world.add(obj)

    # ---- 6. Debug output (recommended to verify material correctness) ----
    # print(">>> Unique mat IDs:", np.unique(world._mat_map))
    # print(">>> Material table:", world._mat_ids)

    return world, player


# ------------------------------------------------------------
# 3. Custom Crafter Environment (string map + native rendering)
# ------------------------------------------------------------
class CustomCrafterEnv(gym.Env):
    """Crafter environment with string-defined maps and native renderer."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
            self,
            txt_file_path=None,
            layout_str=None,
            color_str=None,
            size=None,
            agent_start_pos=None,
            agent_start_dir=None,
            custom_mission="Explore and craft.",
            max_steps=None,
            seed=0,
            ai_enabled=False,
            slippery_prob=0.0,   # Probability that the agent's action is replaced by a random movement
            **kwargs,
    ):
        super().__init__()
        import crafter.worldgen
        # Disable default random world generation
        crafter.worldgen.generate_world = lambda world, player: None

        self.seed = seed
        self.txt_file_path = txt_file_path
        self.layout_str = layout_str
        self.max_steps = max_steps or 10000
        self.current_step = 0
        self.initial_inventory = kwargs.get('initial_inventory', {}) or {}
        
        # ========== Alignment: Parse Layout Input ==========
        if self.txt_file_path:
            with open(self.txt_file_path, 'r') as file:
                sections = file.read().strip().split('\n\n')
                self.layout_str = sections[0].strip()
                # Parse initial inventory if a second block is provided
                if len(sections) > 1:
                    inv_lines = sections[1].strip().split('\n')
                    for line in inv_lines:
                        line = line.strip()
                        if not line or line.startswith('#'): continue
                        if ':' in line or '=' in line:
                            sep = ':' if ':' in line else '='
                            key, val = line.split(sep, 1)
                            self.initial_inventory[key.strip().lower()] = float(val.strip())
        elif self.layout_str:
            # Parsing from a direct string input containing \n\n
            if '\n\n' in self.layout_str:
                sections = self.layout_str.strip().split('\n\n')
                self.layout_str = sections[0].strip()
                if len(sections) > 1:
                    inv_lines = sections[1].strip().split('\n')
                    for line in inv_lines:
                        line = line.strip()
                        if not line or line.startswith('#'): continue
                        if ':' in line or '=' in line:
                            sep = ':' if ':' in line else '='
                            key, val = line.split(sep, 1)
                            self.initial_inventory[key.strip().lower()] = float(val.strip())
                print(f"[CrafterCustomEnv] Parsed initial inventory: {self.initial_inventory}")
        elif not self.layout_str:
            # Fallback tiny map if nothing is provided
            self.layout_str = "GGGGGGG\nGGGGGGG\nGGGPGGG\nGGGGGGG\nGGGGGGG"
            
        # Filter lines to strictly include only map characters, exclude stats lines (e.g., 'health: 9')
        map_lines = []
        for line in self.layout_str.strip().split("\n"):
            s = line.strip()
            if not s or s.startswith('#') or ":" in s:
                continue
            map_lines.append(list(s))
        self.char_grid = np.array(map_lines)

        # Initialize native Crafter environment
        self.env = crafter.Env(reward=False, seed=seed)

        # Inject custom world
        world, player = make_world_from_chars(self.char_grid, seed)
        self.env._world = world
        self.env._player = player

        # Define action/observation space
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 2), dtype=np.int32),
            "inventory": spaces.Box(low=0.0, high=100.0, shape=(16,), dtype=np.float32)
        })
        self.ai_enabled = ai_enabled
        # Slippery: with this probability, a movement action (1-4) is replaced by a random movement
        self.slippery_prob = float(slippery_prob)
        # Crafter movement action IDs: 1=move_left, 2=move_right, 3=move_up, 4=move_down
        self._move_actions = [1, 2, 3, 4]

    def _extract_obs(self):
        # Replace the RGB view with the native symbolic extraction `(H, W, 2)`.
        symbolic_grid = extract_tensor_grid(self.env)
        player = self.env._player
        
        # Extract the four status attributes and twelve inventory counts (16 dims total).
        inv_list = [
            float(player.health), 
            float(player.inventory.get('food', 0)), 
            float(player.inventory.get('drink', 0)), 
            float(player.inventory.get('energy', 0)),
            float(player.inventory.get('wood', 0)), float(player.inventory.get('stone', 0)),
            float(player.inventory.get('coal', 0)), float(player.inventory.get('iron', 0)),
            float(player.inventory.get('diamond', 0)), float(player.inventory.get('sapling', 0)),
            float(player.inventory.get('wood_pickaxe', 0)), float(player.inventory.get('stone_pickaxe', 0)),
            float(player.inventory.get('iron_pickaxe', 0)), float(player.inventory.get('wood_sword', 0)),
            float(player.inventory.get('stone_sword', 0)), float(player.inventory.get('iron_sword', 0))
        ]
        
        return {
            "image": symbolic_grid,
            "inventory": np.array(inv_list, dtype=np.float32)
        }

    def get_agent_position(self, obs=None):
        """
        Return the exact (y, x) position of the player in the Crafter environment.
        In Crafter, coordinates are usually (x, y) but we return (y, x) to match 
        the standard MiniGrid array convention.
        """
        if getattr(self, "env", None) and getattr(self.env, "_player", None):
            pos = self.env._player.pos
            # Crafter pos is typically [x, y], we return (y, x)
            return np.array([int(pos[1]), int(pos[0])])
        return np.array([-1, -1])

    # --------------------------------------------------------
    # Reset / Step / Render
    # --------------------------------------------------------
    def reset(self, **kwargs):
        self.env.reset()

        # Build custom map
        world, player = make_world_from_chars(self.char_grid, seed=self.seed)
        
        # --- Handle custom position/direction for uniform sampling ---
        target_pos = kwargs.get('agent_pos', None)
        target_dir = kwargs.get('agent_dir', None)
        
        if target_pos is not None:
             tx, ty = int(target_pos[0]), int(target_pos[1])
             # Robust material ID lookup would be better, but based on Crafter constants:
             # 2=grass, 3=stone, 4=path, 6=tree, 8=coal, 9=iron, 10=diamond
             if 0 <= tx < world.area[0] and 0 <= ty < world.area[1]:
                 current_mat = world._mat_map[tx, ty] # x,y indexing
                 if current_mat == 6: # Tree -> Grass
                     world._mat_map[tx, ty] = 2
                 elif current_mat in [3, 8, 9, 10]: # Stone, Coal, Iron, Diamond -> Path
                     world._mat_map[tx, ty] = 4
             
             # Use engine-native move to handle chunking and object tracking
             try:
                 world.move(player, np.array(target_pos, dtype=np.int32))
             except Exception as e:
                 print(f"[Warning] Failed to teleport to {target_pos}: {e}. Keeping default spawn.")




        
        if target_dir is not None:
             player.facing = np.array(target_dir, dtype=np.float32)

        self.env._world = world
        self.env._player = player

        # Critical fix: reload textures and rebuild view pipeline
        from crafter import engine, constants
        self.env._textures = engine.Textures(constants.root / "assets")
        view_h, view_w = self.env._view
        item_rows = int(np.ceil(len(constants.items) / view_h))
        self.env._local_view = engine.LocalView(
            self.env._world, self.env._textures, [view_h, view_w - item_rows]
        )
        self.env._item_view = engine.ItemView(
            self.env._textures, [view_h, item_rows]
        )

        self.current_step = 0
        
        # Inject custom initial inventory
        if hasattr(self, 'initial_inventory') and self.initial_inventory:
            for item, amount in self.initial_inventory.items():
                if item in ['health', 'food', 'drink', 'energy']:
                    setattr(self.env._player, item, max(0, min(9, amount)))
                else:
                    self.env._player.inventory[item] = amount

        return self._extract_obs(), {}


    def step(self, action):
        """
        Deterministic or semi-deterministic step function for Crafter.
        - Only updates the player by default (for World Model training).
        - Optionally updates other entities (cow, zombie, etc.) if AI is enabled.
        """

        from crafter import constants

        # --- 1. Apply slippery (only for movement actions) ---
        if self.slippery_prob > 0.0 and action in self._move_actions:
            if np.random.random() < self.slippery_prob:
                # Replace with a random movement action (could be same or different)
                action = np.random.choice(self._move_actions)

        # --- 2. Apply player action ---
        # Map the discrete action ID to the Crafter action constant
        # and update the player’s state accordingly.
        self.env._player.action = constants.actions[action]
        self.env._player.update()

        # --- 3. Optionally update other entities (if AI is enabled) ---
        # This allows switching between deterministic and full simulation modes.
        if self.ai_enabled:
            for obj in self.env._world.objects:
                if obj is not self.env._player:
                    obj.update()

        # --- 3. Update environment time/daylight cycle ---
        # Keep the world visually consistent even if other entities are frozen.
        if hasattr(self.env, "_update_time"):
            self.env._update_time()

        # --- 4. Get the new observation ---
        obs = self._extract_obs()

        # --- 5. Return standard Gym-style outputs ---
        self.current_step += 1
        
        # Episode terminates when the step budget is exhausted or the player dies.
        reward = 0.0
        done = (self.current_step >= self.max_steps) or (self.env._player.health <= 0)
        info = {}

        return obs, reward, done, False, info


    def render(self, mode="rgb_array"):
        return self.env.render(size=(128, 128))

    def render_global(self, unit=64):
        """
        Renders the entire Crafter map globally instead of just the agent-centric local view.
        Returns a full RGB frame.
        """
        from crafter import engine
        world = self.env._world
        textures = self.env._textures
        
        W, H = world.area
        # Create buffer with (Width, Height) to match engine._draw column-first indexing
        canvas = np.zeros((W * unit, H * unit, 3), np.uint8) + 127
        
        # 1. Render all tiles
        for x in range(W):
            for y in range(H):
                material, _ = world[(x, y)]
                if material is not None:
                    texture = textures.get(material, (unit, unit))
                    engine._draw(canvas, (np.array([x, y]) * unit).astype(np.int32), texture)
                    
        # 2. Render all objects on top
        for obj in world.objects:
            pos = obj.pos
            texture = textures.get(obj.texture, (unit, unit))
            engine._draw_alpha(canvas, (pos * unit).astype(np.int32), texture)
            
        # Optional: time-of-day lighting
        # For a clean full map view, we just return the daylight canvas
        return canvas.transpose((1, 0, 2))

    def close(self):
        self.env.close()


# ------------------------------------------------------------
# 4. Example test run
# ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from crafter import constants

    # actually the string shape need to be square

    layout_str = """
    GGIIGGG
    GGAGXGW
    GKGGGtG
    RGOPGGI
    GWGUGGG
    GKGGGTG
    GWGGGGG
    """

    # --- 1. Create a custom environment from the layout string ---
    base_env = CustomCrafterEnv(layout_str=layout_str, seed=0)
    base_env.ai_enabled = False  # keep deterministic (disable Cow/Zombie auto-movement)

    # --- 2. Reset the environment ---
    obs, _ = base_env.reset()

    # --- Handle action mapping (compatible with both list and dict) ---
    # Available actions:
    # 0: noop
    # 1: move_left
    # 2: move_right
    # 3: move_up
    # 4: move_down
    # 5: do
    # 6: sleep
    # 7: place_stone
    # 8: place_table
    # 9: place_furnace
    # 10: place_plant
    # 11: make_wood_pickaxe
    # 12: make_stone_pickaxe
    # 13: make_iron_pickaxe
    # 14: make_wood_sword
    # 15: make_stone_sword
    # 16: make_iron_sword
    if isinstance(constants.actions, dict):
        action_names = list(constants.actions.keys())
    else:
        action_names = list(constants.actions)

    plt.ion()
    for i in range(100):
        # --- Execute a random action ---
        action_id = base_env.action_space.sample()
        action_name = action_names[action_id] if action_id < len(action_names) else str(action_id)

        obs, reward, done, trunc, info = base_env.step(action_id)

        # --- Extract symbolic grid representation ---
        symbolic_obs = obs['image']
        inv_obs = obs['inventory']

        rgb = base_env.render(mode="rgb_array")

        # --- Print debug info ---
        print(f"\n=== Step {i} ===")
        print(f"Action ID: {action_id}  →  {action_name}")
        print("Object layer:", symbolic_obs[..., 0])
        print("Direction layer:", symbolic_obs[..., 1])
        print("Symbolic obs shape:", symbolic_obs.shape)
        print("Inventory obs shape & content:", inv_obs.shape, "\n", inv_obs)

        # --- Visualize RGB frame ---
        plt.imshow(rgb)
        plt.title(f"Step {i} - Action: {action_name}")
        plt.axis("off")
        plt.pause(0.5)

    plt.ioff()
    plt.show()
