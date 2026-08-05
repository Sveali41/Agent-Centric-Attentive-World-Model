import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from modelBased.common import utils
import random
import os


class FisherReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.mask_size = 3  # Cross mask size

    def compute_proxy_score_batch(
        self,
        model: torch.nn.Module,
        samples: List[Dict],
        top_k: int = 50
    ) -> List[Tuple[float, Dict]]:
        model.eval()
        device = next(model.parameters()).device
        scores = []

        with torch.no_grad():
            obs = torch.tensor(samples['obs']).to(device).float()
            act = torch.tensor(samples['act']).to(device)
            obs_next = torch.tensor(samples['obs_next']).to(device).float()
            if 'info' in samples:
                info = torch.tensor(samples['info']).to(device).float()
            else:
                info = None
            agent_postion_yx_batch = utils.get_agent_position(obs)
            agent_postion_yx_batch_next = utils.get_agent_position(obs_next)
            obs_masked = utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
            obs_next_masked = utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch_next)
            pred, _ = model(obs_masked, act, info)
            loss = [F.mse_loss(pred[i], obs_next_masked[i]).item() for i in range(len(pred))]

            # 可选加权项，例如状态变化量
            delta = [(obs_next[i] - obs[i]).abs().mean().item() for i in range(len(obs_next))]
            score = [l + 0.1 * d for l, d in zip(loss, delta)]  # 组合得分
        
            scored_samples = list(zip(score, [dict(obs=samples['obs'][i],
                                                   act=samples['act'][i],
                                                   obs_next=samples['obs_next'][i]) for i in range(len(score))]))
        scored_samples.sort(key=lambda x: -x[0])
        top_k_samples = [s for _, s in scored_samples[:top_k]]
        return top_k_samples

    def select_important_samples(
        self,
        samples: List[Dict],
        model: torch.nn.Module,
        fisher: Dict[str, torch.Tensor], 
        top_k: int = 50
    ) -> List[Dict]:
        scored = self.compute_proxy_score_batch(model, samples, top_k)
        return scored

    def update_with_top_k_recent(self, samples: Dict, model: torch.nn.Module, fisher: Dict[str, torch.Tensor], recent_k: int = 200, top_k: int = 50):
        samples['obs'] = samples['obs'][:recent_k]
        samples['obs_next'] = samples['obs_next'][:recent_k]
        samples['act'] = samples['act'][:recent_k]
        if 'info' in samples:
            samples['info'] = samples['info'][:recent_k]
        selected = self.select_important_samples(samples, model, fisher, top_k)
        self.buffer.extend(selected)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        
    def update_with_random_by_ratio(
        self,
        samples: Dict,
        ratio: float,
        static_ratio: float = 0.2
    ):
        """
        Input:
        - ratio: 从当前 samples 中按比例采样
        - static_ratio: 在选中样本中，静态样本占比

        静态样本: obs_next 与 obs 完全一致
        动态样本: 有任意位置不同
        """
        total_len = len(samples['obs'])
        if total_len == 0:
            return

        insert_k = int(total_len * ratio)
        if insert_k <= 0:
            return

        # === 判断变化位置 ===
        obs = torch.tensor(samples['obs'])         # (B, C, H, W)
        obs_next = torch.tensor(samples['obs_next'])
        changed_mask = (obs != obs_next).any(dim=1).any(dim=1).any(dim=1)  # shape: (B,)
        dynamic_indices = torch.where(changed_mask)[0].tolist()
        static_indices = torch.where(~changed_mask)[0].tolist()

        static_k = int(insert_k * static_ratio)
        dynamic_k = insert_k - static_k

        random.shuffle(dynamic_indices)
        random.shuffle(static_indices)

        dynamic_selected = dynamic_indices[:dynamic_k]
        static_selected = static_indices[:static_k]
        selected_indices = dynamic_selected + static_selected
        random.shuffle(selected_indices)

        selected = []
        for i in selected_indices:
            sample = {
                'obs': samples['obs'][i],
                'act': samples['act'][i],
                'obs_next': samples['obs_next'][i]
            }
            if 'info' in samples:
                sample['info'] = samples['info'][i]
            selected.append(sample)

        self.buffer.extend(selected)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]


    def update_with_random(
        self,
        samples: Dict,
        recent_k: int = 20000,
        random_k: int = 10000
    ):
        for k in ['obs', 'act', 'obs_next', 'info']:
            if k in samples:
                samples[k] = samples[k][:recent_k]

        total_len = len(samples['obs'])
        indices = list(range(total_len))
        random.shuffle(indices)
        selected_indices = indices[:random_k]

        selected = []
        for i in selected_indices:
            sample = {
                'obs': samples['obs'][i],
                'act': samples['act'][i],
                'obs_next': samples['obs_next'][i]
            }
            if 'info' in samples:
                sample['info'] = samples['info'][i]
            selected.append(sample)

        self.buffer.extend(selected)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get_agent_near_elements_mask(self, obs: torch.Tensor):
        """
        返回一个布尔 mask，表示哪些样本中 agent 紧邻 key/door/lava。
        agent 由 obj_map 中值为 10 的位置定义。
        obs: Tensor of shape (B, C, H, W) or (B, H, W, C)
        return: BoolTensor of shape (B,)
        """
        if obs.dim() == 4 and obs.shape[1] != obs.shape[-1]:  # (B, C, H, W)
            obj_map = obs[:, 0]  # object 通道
        else:  # (B, H, W, C)
            obj_map = obs[..., 0]  # object 通道

        B, H, W = obj_map.shape
        near_mask = torch.zeros(B, dtype=torch.bool, device=obs.device)

        for b in range(B):
            # 直接在 obj_map 中查找值为 10 的位置（agent）
            agent_pos = (obj_map[b] == 10).nonzero(as_tuple=False)
            if agent_pos.numel() == 0:
                continue

            y, x = agent_pos[0]  # 假设一个 agent
            neighbors = []
            if y > 0:
                neighbors.append(obj_map[b, y - 1, x])
            if y < H - 1:
                neighbors.append(obj_map[b, y + 1, x])
            if x > 0:
                neighbors.append(obj_map[b, y, x - 1])
            if x < W - 1:
                neighbors.append(obj_map[b, y, x + 1])

            for val in neighbors:
                if val.item() in [4, 5, 9]:  # door or key or lava
                    near_mask[b] = True
                    break

        return near_mask  # (B,)

    def update_combined(
        self,
        samples: Dict[str, np.ndarray],
        ratio: float,                 # 从当前 samples 中抽多少比例
        elements_ratio: float    # 其中 key/door/lava 样本占比
    ):
        """
        综合插入策略（基于当前 sample 数量）：
        1) 从 samples 中抽取 ratio 百分比数据
        2) 其中 key/door 样本占 keydoor_ratio 比例
        """
        total_len = len(samples['obs'])
        if total_len == 0:
            return

        total_quota = int(total_len * ratio)
        if total_quota <= 0:
            return

        # === Part 1: key/door 样本 ===
        obs = samples['obs']
        obs_tensor = torch.tensor(obs) if not isinstance(obs, torch.Tensor) else obs
        try:
            near_elements_mask = self.get_agent_near_elements_mask(obs_tensor)
            near_indices_all = torch.where(near_elements_mask)[0].cpu().numpy()
        except Exception as e:
            print("Error computing near_elements_mask:", e)
            near_indices_all = np.array([], dtype=int)

        elements_quota = int(total_quota * elements_ratio)
        elements_selected = []
        if len(near_indices_all) > 0 and elements_quota > 0:
            pick_n = min(elements_quota, len(near_indices_all))
            elements_selected = np.random.choice(near_indices_all, pick_n, replace=False).tolist()
        # === Part 2: 剩余 quota 从其他样本中随机选择 ===
        remaining_quota = total_quota - len(elements_selected)
        total_indices = list(range(total_len))
        non_elements_pool = [i for i in total_indices if i not in elements_selected]
        random.shuffle(non_elements_pool)
        random_selected = non_elements_pool[:remaining_quota]

        # === 合并采样并打乱 ===
        all_selected_indices = elements_selected + random_selected
        random.shuffle(all_selected_indices)

        selected = []
        for i in all_selected_indices:
            item = {
                'obs': samples['obs'][i],
                'act': samples['act'][i],
                'obs_next': samples['obs_next'][i]
            }
            if 'info' in samples:
                item['info'] = samples['info'][i]
            selected.append(item)

        self.buffer.extend(selected)

        # === 裁剪 buffer ===
        if len(self.buffer) > self.max_size:
            num_to_remove = len(self.buffer) - self.max_size
            all_indices = list(range(len(self.buffer)))
            indices_to_remove = np.random.choice(all_indices, size=num_to_remove, replace=False)
            indices_to_keep = sorted(list(set(all_indices) - set(indices_to_remove)))
            self.buffer = [self.buffer[i] for i in indices_to_keep]




    def export_dict(self) -> Dict[str, np.ndarray]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        keys = self.buffer[0].keys()
        return {k: np.stack([s[k] for s in self.buffer]) for k in keys}

    def load_from_dict(self, data_dict: Dict[str, np.ndarray]):
        self.buffer = []
        length = len(data_dict['obs'])
        for i in range(length):
            sample = {
                'obs': data_dict['obs'][i],
                'act': data_dict['act'][i],
                'obs_next': data_dict['obs_next'][i]
            }
            if 'info' in data_dict:
                sample['info'] = data_dict['info'][i]
            self.buffer.append(sample)

    def save_to_file(self, path: str):
        data = self.export_dict()
        torch.save(data, path)
        print(f"Fisher buffer saved to: {path}")

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No buffer file found at {path}")
        data = torch.load(path)
        self.load_from_dict(data)
        print(f"Fisher buffer loaded from: {path}")

    def __len__(self):
        return len(self.buffer)
