import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from typing import List
import pickle
import torch
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional

def save_dic(dict_obj, save_path='dict.pkl'):
    folder = os.path.dirname(save_path)

    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Directory created: {folder}")

    with open(save_path, 'wb') as f:
        pickle.dump(dict_obj, f)
    
    print(f"Dictionary saved to: {save_path}")
        
def load_dic(save_path):
    with open(save_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def replace_vector_value(grid, src_type='chart'):
    if src_type == 'value':
        mapping = {
            2: 'W',
            1: 'E',
            5: 'K',
            4: 'D',
            8: 'G'
        }
    else:
        mapping = {
            'W': 2,
            'E': 1,
            'K': 5,
            'D': 4,
            'G': 8
        }
    vectorized_replace = np.vectorize(lambda x: mapping.get(x, x)) 
    return vectorized_replace(grid)


def add_start_next_to_key(
    grid: np.ndarray,
    start_value: int = 3,
    key_value: int = 5,
    floor_value: int = 1
) -> Tuple[np.ndarray, bool]:
    """
    Attempts to place a start tile 'S' (start_value) next to any key tile (key_value).
    
    The function iterates over every key found on the map. For each one, it checks
    the four adjacent tiles (up/down/left/right). If at least one key has an empty
    floor tile next to it, the function places 'S' there and returns True.
    Otherwise, it leaves the grid unchanged and returns False.

    Args:
        grid (np.ndarray): 2D array representing the map.
        start_value (int): Numeric code to denote the start S (default = 3).
        key_value (int): Numeric code for key K (default = 5).
        floor_value (int): Numeric code for empty floor (default = 1).

    Returns:
        Tuple[np.ndarray, bool]:
          - Modified grid with an 'S' placed (if successful).
          - Boolean flag: True if placement succeeded; False otherwise.
    """
    height, width = grid.shape
    
    # Find all key positions on the grid
    keys = [tuple(pos) for pos in np.argwhere(grid == key_value)]
    if not keys:
        # No keys present; cannot place start
        return grid, False

    # Shuffle keys to avoid bias in deterministic maps
    np.random.shuffle(keys)

    # For each key, check its four neighbors
    for (ky, kx) in keys:
        for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sy, sx = ky + dy, kx + dx
            # Ensure neighbor is within bounds and is empty floor
            if 0 <= sy < height and 0 <= sx < width and grid[sy, sx] == floor_value:
                new_grid = grid.copy()
                new_grid[sy, sx] = start_value  # Place the start tile
                return new_grid, True

    # No valid adjacent floor space found for any key
    return grid, False



def generate_envs_dataset(
    rows, cols, num_maps,
    wall_p_range=(0.2, 0.5),
    door_p_range=(0.075, 0.15),
    key_p_range=(0.1, 0.3),
    max_len=1e7, random_gen_max=2e4,
    save_flag=False, save_path=None,
    start_point_flag=False
):
    all_map = True
    index = 0
    archive = {}

    key_door = (key_p_range[1] > 0) and (door_p_range[1] > 0)

    while max_len > 0:
        print(f'gen minigrid remain attempt: {max_len}')
        wall_p = random.uniform(*wall_p_range)
        door_p = random.uniform(*door_p_range)
        key_p = random.uniform(door_p, key_p_range[1])

        grid = None
        gen_times = 0
        while True:
            gen_times += 1
            max_len -= 1

            if gen_times > random_gen_max:
                grid = generate_single_env(rows, cols, 'path', wall_prob=wall_p, key_prob=key_p, door_prob=door_p)
                break
            else:
                grid = generate_single_env(rows, cols, 'random', wall_prob=wall_p, key_prob=key_p, door_prob=door_p)
                if is_reachable(grid, key_door):
                    break

        # Place an 'S' next to at least one key
        if start_point_flag:
            grid, placed = add_start_next_to_key(grid, start_value=3, key_value=5)
            if not placed:
                continue  # Skip this map if we couldn't place the start

        # Archive the map
        if all_map:
            archive[index] = grid
            if save_flag:
                visualize_grid(grid, save_flag=True, save_path=save_path, idx=f'map_{index}')
            index += 1
        else:
            desc = compute_descriptors(grid)
            bd1 = min(num_maps - 1, int(desc[0] * num_maps))
            bd2 = min(num_maps - 1, int(desc[1] * num_maps))
            key = (bd1, bd2)
            if key not in archive:
                archive[key] = grid
                if save_flag:
                    visualize_grid(grid, save_flag=True, save_path=save_path, idx=f'{bd1}_{bd2}')

        if len(archive) >= num_maps:
            break

    return archive

def is_reachable(grid: np.ndarray, key_door: bool = False) -> bool:
    h, w = grid.shape
    starts = np.argwhere(grid == 1)
    if len(starts) == 0:  
        return False

    goal_exists = np.any(grid == 8)

    if not goal_exists:
        return False

    # --- key_door 模式：要求完整路径（起点→钥匙→门→目标）
    if key_door:
        WALL, KEY, DOOR, GOAL = 2, 5, 4, 8

        # Step 1: get all non-wall-and-door positions
        traversable_init = lambda v: v != WALL and v != DOOR
        traversable_after_key = lambda v: v != WALL  # After getting key, doors are OK

        non_wall_door_tiles = np.argwhere(grid != WALL)
        if len(non_wall_door_tiles) == 0:
            return False

        # Step 2: check if any key is reachable (without going through doors)
        key_positions = np.argwhere(grid == KEY)
        if len(key_positions) == 0:
            return False  # No key, but door exists → invalid

        # From the first open tile, check if key is reachable
        def bfs(start_pos, valid_tile_func):
            visited = set()
            queue = deque([tuple(start_pos)])
            reached = set()

            while queue:
                y, x = queue.popleft()
                if (y, x) in visited:
                    continue
                visited.add((y, x))
                reached.add((y, x))

                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if (ny, nx) not in visited and valid_tile_func(grid[ny, nx]):
                            queue.append((ny, nx))
            return reached

        # Phase 1: reach the key without passing doors
        reachable_to_key = bfs(starts[0], traversable_init)
        if not any(tuple(pos) in reachable_to_key for pos in key_positions):
            return False  # Key is not reachable

        # Phase 2: after getting key, BFS with door tiles allowed
        reachable_after_key = bfs(key_positions[0], traversable_after_key)

        # Check if goal is reachable
        goal_positions = np.argwhere(grid == GOAL)
        if len(goal_positions) == 0:
            return False
        if not any(tuple(pos) in reachable_after_key for pos in goal_positions):
            return False  # Goal is unreachable even after opening doors

        # Final check: are all non-wall (including door) tiles reachable now?
        non_wall_tiles = np.argwhere(grid != WALL)
        if all(tuple(pos) in reachable_after_key for pos in non_wall_tiles):
            return True
        else:
            return False  # Enclosure detected (some area still unreachable)
    
    else:
        # 查找目标点（'G'）的位置
        starts = np.argwhere(grid != 2)

        # BFS 遍历从所有起始点及目标可达性
        visited = set()
        queue = deque()

        # 将所有起始点加入队列
        queue.append(starts[0])

        # 标记目标是否可达
        reached_goal = False

        while queue:
            y, x = queue.popleft()

            # 如果当前位置是目标，标记为到达目标
            if grid[y, x] == 8:
                reached_goal = True

            # 如果已经访问过这个节点，跳过
            if (y, x) in visited:
                continue

            # 标记为已访问
            visited.add((y, x))

            # 4个方向扩展
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 1 <= ny < h-1 and 1 <= nx < w-1 and (ny, nx) not in visited:
                    ntile = grid[ny, nx]
                    if ntile != 2:  # 如果不是墙壁，继续扩展
                        queue.append((ny, nx))

        if not reached_goal:
            return False  # 如果目标不可达，返回 False

        # 检查所有非墙壁区域是否都被访问，确保没有封闭区域
        if len(visited) == len(starts):
            return True  # 没有封闭区域并且目标可达
        else:
            return False  # 存在封闭区域



def visualize_grid(grid, count=10, save_flag=False, save_path='', idx =''):

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    singlet = True
    if save_flag:
        plt.ioff()
    if isinstance(grid, list) or isinstance(grid, np.ndarray):
        if isinstance(grid[0], list) or isinstance(grid[0], np.ndarray):
            if isinstance(grid[0][0], list) or isinstance(grid[0][0], np.ndarray):
                singlet = False
                if count > len(grid):
                    count = len(grid)
    if not np.issubdtype(grid.dtype, np.integer):
        grid = replace_vector_value(grid, 'chart')

    def show_grid(map, index=''):
        colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)    
        if save_flag:
            save_file_name = os.path.join(save_path, f'gen_{index}.png')
            plt.imshow(map, cmap=custom_cmap)  # 显示图像
            plt.colorbar()
            plt.savefig(save_file_name)  # 保存图像
            plt.close()  # 关闭图像，避免显示
        else:
            plt.imshow(map, cmap=custom_cmap)  # 显示图像
            plt.colorbar()
            plt.show()  # 显示图像
            
    if singlet:
        show_grid(grid, idx)
    else:
        
        for index in range(min(grid.shape[0],count)):
            plt.close()
            show_grid(grid[index,:,:], index)

def generate_path_branch(height, width, path_length=50, branch_num=5, branch_length=7, key_prob=0.0, door_prob=0.0):
    import numpy as np
    import random

    grid = 2 * np.ones((height, width), dtype=int)

    def valid(y, x):
        return 1 <= y < height - 1 and 1 <= x < width - 1

    path = []
    y, x = random.randint(1, height - 2), random.randint(1, width - 2)
    goal = y, x
    grid[y, x] = 1
    path.append((y, x))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 主路径
    for _ in range(path_length):
        random.shuffle(directions)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if valid(ny, nx) and grid[ny, nx] == 2:
                y, x = ny, nx
                grid[y, x] = 1
                path.append((y, x))
                break

    # 分支路径
    for _ in range(branch_num):
        if not path:
            break
        sy, sx = random.choice(path)
        for _ in range(branch_length):
            random.shuffle(directions)
            for dy, dx in directions:
                ny, nx = sy + dy, sx + dx
                if valid(ny, nx) and grid[ny, nx] == 2:
                    grid[ny, nx] = 1
                    sy, sx = ny, nx
                    path.append((sy, sx))
                    break

    # --- 添加 key 和 door ---
    path_no_goal = [pos for pos in path if pos != goal]  # 排除 goal 位置
    random.shuffle(path_no_goal)

    num_keys = int(len(path_no_goal) * key_prob)
    num_doors = int(len(path_no_goal) * door_prob)

    key_positions = path_no_goal[:num_keys]
    door_positions = path_no_goal[num_keys:num_keys + num_doors]

    for y, x in key_positions:
        grid[y, x] = 5  # key
    for y, x in door_positions:
        grid[y, x] = 4  # door

    # 设置目标点
    grid[goal] = 8
    return grid


#region
# def generate_main_path(height, width, path_len):
#     grid = 2 * np.ones((height, width), np.uint8)
#     start = (random.randint(1, height - 2), random.randint(1, width - 2))
#     path = [start]
#     grid[start] = 1

#     directions = [(-1,0), (1,0), (0,-1), (0,1)]
#     for _ in range(path_len):
#         y, x = path[-1]
#         random.shuffle(directions)
#         for dy, dx in directions:
#             ny, nx = y + dy, x + dx
#             if 1 <= ny < height-1 and 1 <= nx < width-1 and grid[ny, nx] == 2:
#                 grid[ny, nx] = 1
#                 path.append((ny, nx))
#                 break  # 添加一格就 break，控制路径连贯性

#     # 随机在主路径周围扩张空地（支路）
#     for _ in range(path_len * 2):
#         y, x = random.choice(path)
#         random.shuffle(directions)
#         for dy, dx in directions:
#             ny, nx = y + dy, x + dx
#             if 1 <= ny < height -1  and 1 <= nx < width - 1 and grid[ny, nx] == 2:
#                 grid[ny, nx] = 1
#                 break
#     return grid
#endregion

def random_generate(height, width, wall_ratio=0.1, key_ratio=0.05, door_ratio=0.05):
    grid = 2 * np.ones((height, width), np.uint8)  # 2: wall
    grid[1:-1, 1:-1] = 1  # 1: empty floor

    # Place goal (G = 8)
    goal_yx = (random.randint(1, height - 2), random.randint(1, width - 2))
    grid[goal_yx[0], goal_yx[1]] = 8

    # Get all empty positions (value == 1)
    empty_pos = np.argwhere(grid == 1)
    empty_count = len(empty_pos)

    # Number of each object to place
    num_walls = int(empty_count * wall_ratio)
    num_keys = int(num_walls * key_ratio)
    num_doors = int(num_walls * door_ratio)
    

    # Shuffle positions
    np.random.shuffle(empty_pos)

    # Place keys (value = 6)
    key_positions = empty_pos[:num_keys]
    for y, x in key_positions:
        grid[y, x] = 5

    # Place doors (value = 7)
    door_positions = empty_pos[num_keys:num_keys + num_doors]
    for y, x in door_positions:
        grid[y, x] = 4

    # Place additional walls (value = 2)
    wall_positions = empty_pos[num_keys + num_doors : num_keys + num_doors + num_walls]
    for y, x in wall_positions:
        grid[y, x] = 2

    return grid
    
def generate_single_env(height, width, mode, wall_prob=0.3, key_prob=0.15, door_prob=0.75):
    if mode == 'random':
        grid = random_generate(height, width, wall_prob, key_prob, door_prob)
    else:
        path_len = int((height-2) * (width-2) * (1-wall_prob))
        branch_num = random.randint(height//2,width)
        branch_length = random.randint(height//3,height-2)
        grid = generate_path_branch(height, width, path_len,branch_num=branch_num, branch_length=branch_length, key_prob=key_prob, door_prob=door_prob) 
    return grid

# =========================
# 2. 行为描述子提取
# =========================
def compute_descriptors(grid):
    empty_ratio = np.mean(grid == 1)
    path_length = estimate_path_length(grid)
    norm_path = path_length / (grid.shape[0] + grid.shape[1])
    return np.array([empty_ratio, norm_path])

def estimate_path_length(grid):
    h, w = grid.shape
    empties = np.argwhere(grid == 1)
    if len(empties) < 2:
        return 0

    max_dist = 0
    for i in range(len(empties)):
        visited = np.zeros_like(grid, dtype=bool)
        q = deque([(tuple(empties[i]), 0)])
        visited[tuple(empties[i])] = True
        while q:
            (x, y), d = q.popleft()
            max_dist = max(max_dist, d)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 1 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    q.append(((nx, ny), d + 1))
    return max_dist