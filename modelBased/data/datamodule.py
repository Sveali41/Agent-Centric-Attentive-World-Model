import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
from modelBased.common.utils import get_env, normalize_obs
from modelBased.common.utils import merge_data_dicts

try:
    from func_timeout import func_set_timeout
except ImportError:
    # Fallback when func_timeout is not installed: keep behavior without timeout enforcement.
    def func_set_timeout(_seconds):
        def decorator(fn):
            return fn
        return decorator


def extract_agent_cross_mask(state):
        """
        Extract a cross-shaped mask centered on the agent's position.
        
        Parameters:
            state (np.ndarray): The 3D array representing the gridworld state.
                                
        Returns:
            np.ndarray: A 3D array of extracted content for the cross-shaped area
                        around the agent, with the layout of 3*3 square, padding with 0.
                        or None if agent is not found.
        """
        # Find agent's position in the grid
        # For the agent position, the object value is 10

                
        # Determine player ID: Crafter uses 13, MiniGrid uses 10
        # Heuristic: if max object ID > 12, it's likely Crafter encoding
        max_id = int(state[:, :, 0].max())
        player_id = 13 if max_id > 12 and np.any(state[:, :, 0] == 13) else 10
        agent_position = np.argwhere(state[:, :, 0] == player_id)

        # Check if the agent position is found
        if len(agent_position) == 0:
            # Could't find the agent position where =10,  take the position where closest to 10 as agent position
            index = np.argmax(state[:, :, 0])
            print(f"Warning! Agent position not found, assume max value: {state[:, :, 0].max()} as agent")
            y, x = index // state.shape[1], index % state.shape[1]
            # return None
        else:
            # Extract y, x coordinates of the agent's position
            y, x = agent_position[0]
            

        cross_structure = np.full((3, 3, state.shape[2]), 0)  # Create a 3x3 structure with None values

        # Extract the content for each valid neighbor position
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < state.shape[0] and 0 <= nx < state.shape[1]:
                cross_structure[dy + 1, dx + 1] = state[ny, nx]  # Place content in the cross structure

        return cross_structure

class WMRLDataset(Dataset):
    @func_set_timeout(100)
    def __init__(self, loaded, hparams, replay_data=None):
        self.hparams = hparams
        self.obs_norm_values = hparams.obs_norm_values
        self.act_norm_values = hparams.action_norm_values
        self.data = self.make_data(loaded, replay_data)

    @staticmethod
    def _to_crafter_nchw(arr):
        """
        Normalize Crafter observations to (N, C, H, W).
        Accepts either (N, H, W, 2) or already-canonical (N, 2, H, W).
        """
        if arr is None:
            return None
        if arr.ndim != 4:
            return arr
        # Raw rollout data from run_env is often (N, H, W, 2).
        if arr.shape[-1] == 2 and arr.shape[1] != 2:
            return np.moveaxis(arr, -1, 1)
        return arr

    @staticmethod
    def _to_minigrid_nchw(arr):
        """
        Normalize MiniGrid observations to (N, C, H, W).
        Accepts either (N, H, W, 3) or already-canonical (N, 3, H, W).
        """
        if arr is None:
            return None
        if arr.ndim != 4:
            return arr
        if arr.shape[-1] == 3 and arr.shape[1] != 3:
            return np.moveaxis(arr, -1, 1)
        return arr

    @staticmethod
    def _sanitize_minigrid_channels(arr, tag="obs"):
        """
        Keep MiniGrid discrete channels in valid ranges to prevent one_hot OOB on GPU.
        Channels: obj[0..10], color[0..5], state[0..3]
        """
        if arr is None or arr.ndim != 4 or arr.shape[1] != 3:
            return arr

        obj = arr[:, 0]
        color = arr[:, 1]
        state = arr[:, 2]

        obj_bad = int(np.count_nonzero((obj < 0) | (obj > 10)))
        color_bad = int(np.count_nonzero((color < 0) | (color > 5)))
        state_bad = int(np.count_nonzero((state < 0) | (state > 3)))
        total_bad = obj_bad + color_bad + state_bad

        if total_bad > 0:
            print(
                f"[DataModule][MiniGrid] Sanitizing {tag}: "
                f"obj_bad={obj_bad}, color_bad={color_bad}, state_bad={state_bad}"
            )
            arr = arr.copy()
            np.clip(arr[:, 0], 0, 10, out=arr[:, 0])
            np.clip(arr[:, 1], 0, 5, out=arr[:, 1])
            np.clip(arr[:, 2], 0, 3, out=arr[:, 2])
        return arr

    def state_batch_preprocess(self, state):
        obs = np.zeros((state.shape[0], 3, 3, state.shape[-1])) # The mask will extract a 3x3 square around the agent
        for i in range(state.shape[0]):  # Loop over the last dimension (channels)
            obs[i] = extract_agent_cross_mask(state[i])
        return obs

    @func_set_timeout(1000)
    def make_data(self, loaded, replay_data=None):
        """
        Build the training dataset by mixing newly collected data with replay-buffer data.

        Core logic:
        1. Replay control: cap replay volume so historical data does not drown out
           the current iteration's samples.
        2. Sampling: subsample replay data when it exceeds the allowed ratio.
        3. Target construction: use `obs_delta` (next frame minus current frame)
           as the world-model target when appropriate for better numerical stability.

        Args:
            loaded (dict): Current-iteration samples (obs, next, act, info).
            replay_data (dict, optional): Historical samples from the replay buffer.
        """
        import numpy as np
        seed = getattr(self.hparams, "seed", None)
        if seed is None:
            env_seed = os.environ.get("PYTHONHASHSEED")
            seed = int(env_seed) if env_seed is not None else 0
        rng = np.random.default_rng(int(seed))  # Use one RNG aligned with the global seed.

        # ===== Load raw arrays =====
        mask_size = self.hparams.attention_mask_size
        env_type  = self.hparams.env_type
        obs, obs_next, act = loaded['a'], loaded['b'], loaded['c']
        rew, done = loaded.get('d', None), loaded.get('e', None)
        info = loaded.get('f', None) if env_type == 'with_obj' else None
        # Crafter only: load inventory if available
        inv     = loaded.get('g', None) if env_type == 'crafter' else None
        inv_next = loaded.get('h', None) if env_type == 'crafter' else None

        if env_type == 'crafter':
            obs = self._to_crafter_nchw(obs)
            obs_next = self._to_crafter_nchw(obs_next)
        elif env_type == 'minigrid':
            obs = self._to_minigrid_nchw(obs)
            obs_next = self._to_minigrid_nchw(obs_next)
            obs = self._sanitize_minigrid_channels(obs, tag="current_obs")
            obs_next = self._sanitize_minigrid_channels(obs_next, tag="current_obs_next")

        current_n = len(obs)
        assert current_n == len(obs_next) == len(act), "[BUG] Current lengths inconsistent!"

        # Optional training-sample cap: use only a subset if configured.
        max_train_samples = int(getattr(self.hparams, "max_train_samples", 0))
        if 0 < max_train_samples < current_n:
            indices = rng.choice(current_n, size=max_train_samples, replace=False)
            obs, obs_next, act = obs[indices], obs_next[indices], act[indices]
            if rew is not None: rew = rew[indices]
            if done is not None: done = done[indices]
            if inv is not None: inv = inv[indices]
            if inv_next is not None: inv_next = inv_next[indices]
            if info is not None: info = info[indices]
            current_n = max_train_samples
            print(f"[Dataset] Capped current task data to {max_train_samples} samples.")

        # Short-circuit when both current and replay datasets are empty.
        if current_n == 0 and (replay_data is None or len(replay_data.get('obs', [])) == 0):
            return {'obs': np.array([]), 'obs_next': np.array([]), 'act': np.array([])}

        # ===== (0.5) Frame Stacking (If enabled for new data) =====
        frame_stack = int(getattr(self.hparams, "frame_stack", 1))
        if frame_stack > 1 and done is not None and len(done) > 0:
             # ... [Keep your stacking logic here, it only triggers if >1]
             # Identify episode starts
             is_first = np.zeros(len(done), dtype=bool)
             is_first[0] = True
             is_first[1:] = done[:-1]
             
             C, H, W = obs.shape[1], obs.shape[2], obs.shape[3]
             stacked_obs = np.zeros((current_n, C * frame_stack, H, W), dtype=obs.dtype)
             curr_start = 0
             for i in range(current_n):
                 if is_first[i]: curr_start = i
                 for k in range(frame_stack):
                     history_idx = max(curr_start, i - (frame_stack - 1 - k))
                     stacked_obs[i, k*C:(k+1)*C, :, :] = obs[history_idx]
             obs = stacked_obs
             print(f"Frame stacking enabled: K={frame_stack}. Input shape: {obs.shape}")
        else:
             # If stack=1, we do NOTHING. Exactly like original.
             pass

        # ===== (1) Control the replay ratio: replay <= configured fraction of new data =====
        # Read the optional replay ratio from hparams; default is 0.5.
        replay_frac = float(getattr(self.hparams, "replay_frac", 0.5))
        replay_frac = max(0.0, min(50.0, replay_frac))  # Allow higher ratios (e.g., 6.0 user request)
        
        # If current data is empty but we have replay data, we allow using replay data
        # but max_replay would be 0 if we strictly follow (current_n * replay_frac).
        # We handle this by allowing a minimum if current_n is 0 but replay_data exists.
        max_replay = int(current_n * replay_frac) if current_n > 0 else 1000000 

        if replay_data is not None and 'obs' in replay_data and replay_data['obs'] is not None:
            R = len(replay_data['obs'])
            # Sample at most `max_replay` items from the replay buffer.
            if R > max_replay and max_replay > 0:
                idx = rng.choice(R, size=max_replay, replace=False)
                r_obs      = replay_data['obs'][idx]
                r_obs_next = replay_data['obs_next'][idx]
                r_act      = replay_data['act'][idx]
                r_info     = (replay_data['info'][idx]
                            if (env_type == 'with_obj' and 'info' in replay_data and replay_data['info'] is not None)
                            else None)
                r_inv      = replay_data['inv'][idx] if ('inv' in replay_data and replay_data['inv'] is not None) else None
                r_inv_next = replay_data['inv_next'][idx] if ('inv_next' in replay_data and replay_data['inv_next'] is not None) else None
                
                # [Robustness] Handle cases where inventory is missing from replay data
                if r_inv is None and inv is not None:
                    # Pad with zero inventory matching new data's feature dimension
                    inv_dim = inv.shape[-1]
                    r_inv = np.zeros((len(idx), inv_dim), dtype=np.float32)
                if r_inv_next is None and inv_next is not None:
                    inv_dim = inv_next.shape[-1]
                    r_inv_next = np.zeros((len(idx), inv_dim), dtype=np.float32)
            else:
                r_obs, r_obs_next, r_act = replay_data['obs'], replay_data['obs_next'], replay_data['act']
                r_info = (replay_data['info']
                        if (env_type == 'with_obj' and 'info' in replay_data and replay_data['info'] is not None)
                        else None)
                r_inv      = replay_data.get('inv', None)
                r_inv_next = replay_data.get('inv_next', None)
                
                # [Robustness] Handle whole-batch missing inventory
                if r_inv is None and inv is not None:
                    inv_dim = inv.shape[-1]
                    r_inv = np.zeros((len(r_obs), inv_dim), dtype=np.float32)
                if r_inv_next is None and inv_next is not None:
                    inv_dim = inv_next.shape[-1]
                    r_inv_next = np.zeros((len(r_obs), inv_dim), dtype=np.float32)

            if env_type == 'crafter':
                r_obs = self._to_crafter_nchw(r_obs)
                r_obs_next = self._to_crafter_nchw(r_obs_next)
            elif env_type == 'minigrid':
                r_obs = self._to_minigrid_nchw(r_obs)
                r_obs_next = self._to_minigrid_nchw(r_obs_next)
                r_obs = self._sanitize_minigrid_channels(r_obs, tag="replay_obs")
                r_obs_next = self._sanitize_minigrid_channels(r_obs_next, tag="replay_obs_next")

            # Concatenate replay data. `r_obs` must already match the current input shape.
            if r_obs.shape[1:] == obs.shape[1:]:
                obs      = np.concatenate([obs,      r_obs     ], axis=0) if current_n > 0 else r_obs
                obs_next = np.concatenate([obs_next, r_obs_next], axis=0) if current_n > 0 else r_obs_next
                act      = np.concatenate([act,      r_act     ], axis=0) if current_n > 0 else r_act
                if env_type == 'with_obj' and r_info is not None:
                    info = np.concatenate([info, r_info], axis=0) if info is not None else r_info
                if r_inv is not None:
                    inv = np.concatenate([inv, r_inv], axis=0) if (current_n > 0 and inv is not None) else r_inv
                if r_inv_next is not None:
                    inv_next = np.concatenate([inv_next, r_inv_next], axis=0) if (current_n > 0 and inv_next is not None) else r_inv_next
            else:
                print(f"Warning: Replay buffer obs shape {r_obs.shape} does not match current obs shape {obs.shape}. Skipping replay.")

            # Shuffle all samples together.
            N = len(obs)
            if N > 0:
                perm = rng.permutation(N)
                obs, obs_next, act = obs[perm], obs_next[perm], act[perm]
                if env_type == 'with_obj' and info is not None and len(info) == N:
                    info = info[perm]
                if inv is not None and len(inv) == N:
                    inv = inv[perm]
                if inv_next is not None and len(inv_next) == N:
                    inv_next = inv_next[perm]

            print(f"Adding replay buffer with {len(r_obs)} samples.")
        if current_n == 0:
             print("[System] Using replay data only.")

        # If current is empty but replay exists, we might need a fallback for C_base
        if current_n > 0:
            C_base = obs_next.shape[1]
        elif replay_data is not None and len(replay_data['obs_next']) > 0:
            C_base = replay_data['obs_next'].shape[1]
        else:
            C_base = 3 # MiniGrid default

        # Use the latest frame in stacked obs to calculate delta against obs_next
        obs_latest = obs[:, -C_base:] if (obs.ndim > 1 and obs.shape[1] > C_base) else obs

        # ===== (2) Build training targets =====
        if self.hparams.data_type == 'norm':
            obs_f      = normalize_obs(obs,      self.obs_norm_values).astype(np.float32)
            obs_next_f = normalize_obs(obs_next, self.obs_norm_values).astype(np.float32)
            act_f      = act.astype(np.float32) / self.act_norm_values
            
            # Recalculate obs_f_latest if stacked
            obs_f_latest = normalize_obs(obs_latest, self.obs_norm_values).astype(np.float32)
            obs_delta  = (obs_next_f - obs_f_latest).astype(np.float32)

        elif self.hparams.data_type == 'discrete':
            act_f = act.astype(np.int64)
            if env_type == 'minigrid':
                act_bad = int(np.count_nonzero((act_f < 0) | (act_f >= int(self.act_norm_values))))
                if act_bad > 0:
                    print(
                        f"[DataModule][MiniGrid] Sanitizing actions: bad={act_bad}, "
                        f"valid=[0, {int(self.act_norm_values) - 1}]"
                    )
                    np.clip(act_f, 0, int(self.act_norm_values) - 1, out=act_f)
            obs_f = obs  # Keep discrete values unchanged for visualization/debugging.

            if env_type == 'crafter':
                # For Crafter: predict the ABSOLUTE next frame (not delta).
                # CrossEntropy loss requires class indices (0-16), not delta values.
                obs_delta = obs_next.astype(np.float32)
                print(f"[DataModule] Crafter mode: using absolute next frame as target (shape {obs_delta.shape})")
            else:
                # MiniGrid: predict delta for MSE regression (numerically stable)
                obs_delta = (obs_next.astype(np.int16) - obs_latest.astype(np.int16)).astype(np.float32)
                if getattr(self.hparams, "clip_discrete_delta", False):
                    np.clip(obs_delta, -1, 1, out=obs_delta)

        else:
            raise ValueError(f"Invalid data type: {self.hparams.data_type}")

        # ===== (3) Package the dataset =====
        data = {'obs': obs_f, 'obs_next': obs_delta, 'act': act_f}
        if env_type == 'with_obj' and info is not None:
            data['info'] = info
        # Crafter: add inventory
        if env_type == 'crafter' and inv is not None:
            data['inv']      = inv[:len(obs_f)].astype(np.float32)
            data['inv_next'] = inv_next[:len(obs_f)].astype(np.float32)
        return data

            



    def __len__(self):
        lengths = [len(self.data[k]) for k in self.data]
        if not all(l == lengths[0] for l in lengths):
            print(f"[BUG] Inconsistent lengths! { {k: len(self.data[k]) for k in self.data} }")
        return lengths[0]  # Use the first key as the canonical dataset length.

    def __getitem__(self, idx):
        try:
            return {key: self.data[key][idx] for key in self.data}
        except IndexError as e:
            print(f"[ERROR] idx={idx}, dataset length={len(self)}")
            raise e


class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, hparams=None, data: Optional[Dict[str, np.ndarray]] = None, replay_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize with hyperparameters and optionally directly with data.

        Parameters:
            hparams: Hyperparameters for data processing and dataloaders
            data: Optional data dictionary, e.g., {'a': np.array(...), 'b': np.array(...), 'c': np.array(...)}
        """
        super().__init__()
        # Keep config as a plain attribute instead of Lightning hparams.
        # Lightning merges module/datamodule hparams during validation and will
        # raise if both sides define the same keys with different values.
        self.cfg = hparams
        self.data_dir = self.cfg.data_dir
        self.direct_data = data  # Store the data passed directly
        self.replay_data = replay_data
        
    def setup(self, stage: Optional[str] = None):
        if self.direct_data is not None:
            loaded = self.direct_data  # Use directly passed data
        else:
            # Load data from a file if `self.data_dir` is set and data is not provided directly
            loaded = np.load(self.data_dir, allow_pickle=True) # Allow pickle for safety with complex data structures
        data = WMRLDataset(loaded, self.cfg, self.replay_data)
        if len(data) == 0:
            print("[Warning] Dataset is empty! Training and Test sets will be empty.")
            self.data_train = torch.utils.data.Subset(data, [])
            self.data_test = torch.utils.data.Subset(data, [])
            return

        split_size = int(len(data) * 9 / 10)
        
        # If the dataset is so small that test set would be 0 or tiny, 
        # we use the same data for both to avoid crashing.
        if (len(data) - split_size) < 1:
            print(f"[Warning] Dataset very small ({len(data)} samples). Using full data for both Train and Test.")
            self.data_train = torch.utils.data.Subset(data, range(len(data)))
            self.data_test = torch.utils.data.Subset(data, range(len(data)))
        else:
            self.data_train = torch.utils.data.Subset(data, range(0, split_size))
            self.data_test = torch.utils.data.Subset(data, range(split_size, len(data)))

    def train_dataloader(self):
        num_workers = int(getattr(self.cfg, "n_cpu", 0))
        return DataLoader(
            self.data_train, 
            batch_size=self.cfg.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0)
        )

    def val_dataloader(self):
        num_workers = int(getattr(self.cfg, "n_cpu", 0))
        return DataLoader(
            self.data_test, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0)
        )
