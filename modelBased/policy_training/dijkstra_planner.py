import sys
from common.utils import PROJECT_ROOT
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
import networkx as nx
from omegaconf import DictConfig
import hydra
from datetime import datetime
from common import utils
from domain.minigrid import minigrid_support as minigrid_utils
import modelBased.world_model.AttentionWM_support as AttentionWM_support
import modelBased.world_model.Embedding_support as Embedding_support
import modelBased.world_model.MLP_support as MLP_support
from modelBased.policy_training.PPO_world_training import find_position, process_data, add_object_to_inventory
import wandb


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GraphPlanner:
    def __init__(self, model, num_actions, mask_size, valid_values_obj, valid_values_color, valid_values_state):
        self.model = model
        self.num_actions = num_actions
        self.mask_size = mask_size
        self.valid_values_obj = valid_values_obj
        self.valid_values_color = valid_values_color
        self.valid_values_state = valid_values_state
        self.G = nx.DiGraph()

    def _state_key(self, full_obs: np.ndarray) -> bytes:
        return full_obs.astype(np.float32).tobytes()

    def expand_node(self, full_obs: np.ndarray):
        key = self._state_key(full_obs)
        if key in self.G:
            return
        self.G.add_node(key)
        for a in range(self.num_actions):
            masked = process_data(full_obs, self.mask_size)
            with torch.no_grad():
                delta, _ = self.model(
                    torch.tensor(masked).unsqueeze(0).to(device),
                    torch.tensor([a]).to(device),
                    {'carrying_key': False}  
                )
            delta_np = delta.squeeze(0).cpu().numpy()
            next_masked = masked + delta_np
            next_masked = minigrid_utils.map_obs_to_nearest_value(
                next_masked,
                self.valid_values_obj,
                self.valid_values_color,
                self.valid_values_state
            )
            next_full = minigrid_utils.put_back_masked_state(
                next_masked,
                full_obs.copy(),
                self.mask_size,
                minigrid_utils.get_agent_position(full_obs),
            )
            self.G.add_edge(
                key,
                self._state_key(next_full),
                action=a,
                weight=1.0
            )

    def plan(self, start_full: np.ndarray, goal_full: np.ndarray, k: int = 1):
        self.expand_node(start_full)
        self.expand_node(goal_full)

        start_key = self._state_key(start_full)
        goal_key = self._state_key(goal_full)

        def heuristic(n1, n2):
            v1 = np.frombuffer(n1, dtype=np.float32)
            v2 = np.frombuffer(n2, dtype=np.float32)
            return np.linalg.norm(v1 - v2)

        path = nx.astar_path(self.G, start_key, goal_key, heuristic=heuristic, weight='weight')
        actions = [self.G.edges[(u, v)]['action'] for u, v in zip(path, path[1:])]
        return actions[:k]


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "trainer/conf"), config_name="config_test")
def run_planner_rollout(cfg: DictConfig):
    hparams = cfg
    hparams_wm = hparams.attention_model
    hparams_planner = hparams.planner

    MODEL_MAP = {
        'attention': AttentionWM_support.AttentionModule,
        'embedding': Embedding_support.EmbeddingModule,
        'mlp': MLP_support.SimpleNNModule
    }
    ModelClass = MODEL_MAP[hparams_wm.model_type.lower()]
    model = ModelClass(
        hparams_wm.data_type,
        hparams_wm.grid_shape,
        hparams_wm.attention_mask_size,
        hparams_wm.embed_dim,
        hparams_wm.num_heads
    ).to(device)
    utils.load_model_weight(model, hparams_wm.model_save_path)
    model.eval()

    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=hparams_planner.env_path,
            custom_mission="Find the key and open the door.",
            max_steps=hparams_planner.max_ep_len,
            render_mode=None
        )
    )

    action_dim = env.action_space.n
    planner = GraphPlanner(
        model,
        action_dim,
        hparams_wm.attention_mask_size,
        hparams_wm.valid_values_obj,
        hparams_wm.valid_values_color,
        hparams_wm.valid_values_state
    )

    max_steps = hparams_planner.max_training_timesteps
    max_len = hparams_planner.max_ep_len
    time_step = 0

    while time_step < max_steps:
        full_obs = env.reset()[0]['image']
        state_0 = minigrid_utils.ColRowCanl_to_CanlRowCol(full_obs)
        goal_yx = find_position(state_0, (8, 1, 0))
        goal_obs = state_0.copy()

        agent_yx = minigrid_utils.get_agent_position(goal_obs)
        goal_obs[:, agent_yx[0], agent_yx[1]] = np.array([0, 0, 0], dtype=goal_obs.dtype)
        goal_obs[:, goal_yx[0], goal_yx[1]] = np.array([10, 0, 0], dtype=goal_obs.dtype)

        planned_actions = planner.plan(state_0, goal_obs, k=max_len)

        info = {'carrying_key': False}
        
        if not planned_actions:
            print("No plan found.")
            continue

        for t, action_id in enumerate(planned_actions):
            action = torch.tensor([action_id], device=device)
            masked = process_data(state_0, hparams_wm.attention_mask_size)
            with torch.no_grad():
                delta, _ = model(masked.unsqueeze(0).to(device), action, info)
            delta_np = delta.squeeze(0).cpu().numpy()
            next_masked = masked + delta_np
            next_masked = minigrid_utils.map_obs_to_nearest_value(
                next_masked,
                hparams_wm.valid_values_obj,
                hparams_wm.valid_values_color,
                hparams_wm.valid_values_state
            )
            info = add_object_to_inventory((next_masked - masked), info)
            agent_pos = minigrid_utils.get_agent_position(state_0)
            state_0 = minigrid_utils.put_back_masked_state(
                next_masked,
                state_0,
                hparams_wm.attention_mask_size,
                agent_pos,
            )

            time_step += 1
            if time_step >= max_steps:
                break


if __name__ == "__main__":
    run_planner_rollout()
