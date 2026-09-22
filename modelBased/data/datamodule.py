import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Optional
from modelBased.common.utils import get_env, normalize_obs
from modelBased.common.utils import merge_data_dicts
from domain.minigrid.transition_codec import MINIGRID_ACTION_COUNT

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
        self.replay_sampling_stats = None
        self.protected_replay_slot_counts = {}
        self.data = self.make_data(loaded, replay_data)

    @staticmethod
    def _crafter_replay_sampling_stats(types, actions, changed_slot_signatures=None):
        labels = ("inventory_change", "map_interaction", "agent_dynamics", "static")
        types = np.asarray(types, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        signatures = None
        if changed_slot_signatures is not None:
            signatures = np.asarray(changed_slot_signatures, dtype=np.uint16).reshape(-1)
            if len(signatures) != len(types):
                raise ValueError("Crafter replay changed-slot statistics are not aligned")
        result = {}
        for type_id, label in enumerate(labels):
            selected = actions[types == type_id]
            result[label] = {
                "count": int(len(selected)),
                "actions": {
                    int(a): int(c) for a, c in zip(*np.unique(selected, return_counts=True))
                },
                "changed_slots": (
                    {
                        int(slot + 4): int(
                            np.sum((signatures[types == type_id] & (1 << slot)) != 0)
                        )
                        for slot in range(12)
                        if np.any((signatures[types == type_id] & (1 << slot)) != 0)
                    }
                    if signatures is not None else {}
                ),
            }
        return result

    def _crafter_soft_replay_indices(self, replay_data, count, rng):
        """Sample historical Crafter replay by persisted transition/action bucket."""
        buckets = np.asarray(replay_data["_replay_bucket"], dtype=np.int64)
        types = np.asarray(replay_data["_replay_transition_type"], dtype=np.int64)
        actions = np.asarray(replay_data["act"], dtype=np.int64).reshape(-1)
        if len(buckets) != len(types) or len(types) != len(actions):
            raise ValueError("Crafter replay metadata is not aligned with obs/act")
        protected_counts = {}
        replay_cfg = getattr(self.hparams, "crafter_transition_replay", None)
        protected_cfg = (
            replay_cfg.get("protected_slot_replay", {}) if isinstance(replay_cfg, dict)
            else getattr(replay_cfg, "protected_slot_replay", {})
        ) if replay_cfg is not None else {}
        protected_enabled = bool(
            protected_cfg.get("enabled", False) if isinstance(protected_cfg, dict)
            else getattr(protected_cfg, "enabled", False)
        )
        protected_min = int(
            protected_cfg.get("min_samples_per_observed_slot", 32)
            if isinstance(protected_cfg, dict)
            else getattr(protected_cfg, "min_samples_per_observed_slot", 32)
        )
        if protected_min < 1:
            raise ValueError("protected_slot_replay.min_samples_per_observed_slot must be positive")
        if count >= len(buckets):
            selected = np.arange(len(buckets), dtype=np.int64)
        else:
            if isinstance(replay_cfg, dict):
                exponent = float(replay_cfg.get("balance_exponent", 0.5))
                include_action = bool(replay_cfg.get("include_action", True))
                include_changed_slot = bool(replay_cfg.get("include_changed_slot", False))
                inventory_fraction = float(replay_cfg.get("inventory_replay_fraction", 0.0))
            else:
                exponent = float(getattr(replay_cfg, "balance_exponent", 0.5))
                include_action = bool(getattr(replay_cfg, "include_action", True))
                include_changed_slot = bool(getattr(replay_cfg, "include_changed_slot", False))
                inventory_fraction = float(getattr(replay_cfg, "inventory_replay_fraction", 0.0))
            if not np.isfinite(inventory_fraction) or not 0.0 <= inventory_fraction <= 1.0:
                raise ValueError(
                    "crafter_transition_replay.inventory_replay_fraction must be in [0, 1]"
                )
            if include_changed_slot:
                signatures = np.asarray(
                    replay_data.get("_replay_changed_slot_signature"), dtype=np.int64
                ).reshape(-1)
                if len(signatures) != len(types):
                    raise ValueError("Crafter changed-slot replay metadata is not aligned with obs/act")
                base = types * 17 + actions if include_action else types
                sampling_buckets = base * (1 << 12) + signatures
            else:
                sampling_buckets = buckets if include_action else buckets // 17
            # Protected rehearsal uses only real historical item transitions.
            # Rarest observed slots claim their unique examples first; a
            # multi-slot transition can satisfy several quotas without ever
            # being duplicated in the training set.
            protected = []
            if protected_enabled:
                signatures_u16 = np.asarray(
                    replay_data["_replay_changed_slot_signature"], dtype=np.uint16
                ).reshape(-1)
                slot_freq = np.asarray([
                    np.count_nonzero(signatures_u16 & (1 << slot))
                    for slot in range(12)
                ])
                for slot in np.argsort(slot_freq, kind="stable"):
                    if slot_freq[slot] == 0 or len(protected) >= count:
                        continue
                    already_covered = int(np.count_nonzero(
                        signatures_u16[np.asarray(protected, dtype=np.int64)] & (1 << slot)
                    )) if protected else 0
                    needed = max(0, protected_min - already_covered)
                    if needed == 0:
                        continue
                    candidates = np.flatnonzero((signatures_u16 & (1 << slot)) != 0)
                    candidates = candidates[~np.isin(candidates, protected)]
                    take = min(needed, len(candidates), count - len(protected))
                    if take:
                        chosen = rng.choice(candidates, size=take, replace=False)
                        protected.extend(chosen.tolist())
                protected = np.asarray(protected, dtype=np.int64)
                for slot in range(12):
                    protected_counts[int(slot + 4)] = int(np.count_nonzero(
                        signatures_u16[protected] & (1 << slot)
                    )) if len(protected) else 0
            protected = np.asarray(protected, dtype=np.int64)
            # Reserve a deterministic fraction of historical replay for real
            # inventory transitions.  This is deliberately a quota rather
            # than a loss weight: map/pose still see the remaining natural
            # replay, while the inventory head receives a minimum amount of
            # CHANGE supervision.  If the buffer contains fewer observed
            # inventory transitions than the quota, all of them are retained.
            inventory_quota = int(np.ceil(count * inventory_fraction))
            if inventory_quota > 0 and len(protected) < count:
                protected_inventory = int(np.count_nonzero(types[protected] == 0)) if len(protected) else 0
                needed_inventory = max(0, inventory_quota - protected_inventory)
                inventory_candidates = np.flatnonzero(types == 0)
                inventory_candidates = inventory_candidates[
                    ~np.isin(inventory_candidates, protected)
                ]
                take = min(needed_inventory, len(inventory_candidates), count - len(protected))
                if take:
                    _, inverse, frequencies = np.unique(
                        sampling_buckets[inventory_candidates],
                        return_inverse=True,
                        return_counts=True,
                    )
                    weights = frequencies[inverse].astype(np.float64) ** (-exponent)
                    weights /= weights.sum()
                    chosen = rng.choice(
                        inventory_candidates, size=take, replace=False, p=weights
                    )
                    protected = np.concatenate((protected, chosen.astype(np.int64, copy=False)))
            if protected_enabled:
                for slot in range(12):
                    protected_counts[int(slot + 4)] = int(np.count_nonzero(
                        signatures_u16[protected] & (1 << slot)
                    )) if len(protected) else 0
            remaining = count - len(protected)
            available = np.setdiff1d(np.arange(len(buckets)), protected, assume_unique=False)
            if remaining:
                _, inverse, frequencies = np.unique(
                    sampling_buckets[available], return_inverse=True, return_counts=True
                )
                weights = frequencies[inverse].astype(np.float64) ** (-exponent)
                weights /= weights.sum()
                fill = rng.choice(available, size=remaining, replace=False, p=weights)
                selected = np.concatenate((protected, fill)).astype(np.int64, copy=False)
            else:
                selected = protected
        selected_signatures = None
        if "_replay_changed_slot_signature" in replay_data:
            selected_signatures = np.asarray(
                replay_data["_replay_changed_slot_signature"], dtype=np.uint16
            )[selected]
        self.replay_sampling_stats = self._crafter_replay_sampling_stats(
            types[selected], actions[selected], selected_signatures
        )
        self.protected_replay_slot_counts = protected_counts
        return selected

    def _minigrid_stochastic_latent_enabled(self):
        """Whether MiniGrid stochastic training needs natural sampling."""
        domain_switch = getattr(self.hparams, "stochastic_enabled", None)
        if domain_switch is not None:
            return bool(domain_switch)
        model = getattr(self.hparams, "stochastic_model", None)
        if model is not None:
            return str(model).lower() in {"outcome_v1", "latent_v2"}
        config = getattr(self.hparams, "stochastic_latent", None)
        if config is None:
            return False
        if isinstance(config, dict):
            return bool(config.get("enabled", False))
        return bool(getattr(config, "enabled", False))

    def _minigrid_outcome_labels_required(self):
        """Whether the selected stochastic model requires action_failed labels."""
        # The unified domain switch selects latent_v2, whose labels are only
        # optional auxiliary supervision rather than a dataset requirement.
        if getattr(self.hparams, "stochastic_enabled", None) is not None:
            return False
        model = getattr(self.hparams, "stochastic_model", None)
        if model is not None:
            return str(model).lower() == "outcome_v1"
        return self._minigrid_stochastic_latent_enabled()

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
            raise ValueError(
                f"[DataModule][MiniGrid] Invalid {tag}: "
                f"obj_bad={obj_bad}, color_bad={color_bad}, state_bad={state_bad}. "
                "Raw categorical observations must be absolute IDs in the declared ranges; "
                "the loader will not clip deltas into labels."
            )
        agent_counts = (obj == 10).reshape(obj.shape[0], -1).sum(axis=1)
        invalid_agents = np.flatnonzero(agent_counts != 1)
        if len(invalid_agents):
            raise ValueError(
                f"[DataModule][MiniGrid] {tag} must contain exactly one agent "
                f"per sample; invalid samples={invalid_agents[:10].tolist()}, "
                f"counts={agent_counts[invalid_agents[:10]].tolist()}"
            )
        return arr

    @staticmethod
    def _minigrid_inventory_from_info(info, done=None):
        """Decode colour-aware inventory tokens from transition metadata.

        Token 0 means empty hands; tokens 1..6 represent MiniGrid key colours
        0..5. Both sides must belong to the same transition.  Legacy metadata
        cannot be repaired by shifting adjacent rows because uniform/replay
        datasets are not guaranteed to preserve temporal adjacency.
        """
        if info is None:
            return None, None
        flat_info = np.asarray(info, dtype=object).reshape(-1)

        def _as_dict(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
                if isinstance(value, dict):
                    return value
            return {}

        records = [_as_dict(value) for value in flat_info]
        missing_current = [
            index for index, record in enumerate(records)
            if "current_carrying_token" not in record
        ]
        missing_next = [
            index for index, record in enumerate(records)
            if "next_carrying_token" not in record
        ]
        if missing_current or missing_next:
            raise ValueError(
                "[DataModule][MiniGrid] Colour-aware inventory supervision "
                "requires current_carrying_token and next_carrying_token on "
                "every transition. Legacy carrying_key/carrying_token fields "
                "cannot recover the two raw inventory states; recollect this "
                "dataset. "
                f"missing_current={missing_current[:10]}, "
                f"missing_next={missing_next[:10]}"
            )

        current = np.asarray([
            int(record["current_carrying_token"])
            for record in records
        ], dtype=np.int64)
        next_inventory = np.asarray([
            int(record["next_carrying_token"]) for record in records
        ], dtype=np.int64)
        invalid_current = (current < 0) | (current >= 7)
        invalid_next = (next_inventory < 0) | (next_inventory >= 7)
        if invalid_current.any() or invalid_next.any():
            raise ValueError(
                "[DataModule][MiniGrid] Inventory tokens must be in [0, 6]; "
                f"invalid_current={int(invalid_current.sum())}, "
                f"invalid_next={int(invalid_next.sum())}"
            )
        return current, next_inventory

    @staticmethod
    def _normalize_minigrid_info_for_batch(info):
        """Return a fixed-schema copy of MiniGrid transition metadata.

        Gym environment ``info`` dictionaries may contain event-only keys.  In
        particular, ``uniform_reset`` is present only on forced-reset rows in
        already collected target datasets.  PyTorch's default collator treats
        dictionaries as structured batch data and therefore requires every row
        to expose exactly the same keys.

        The world model consumes the explicit ``inv`` tensor, not arbitrary
        environment diagnostics.  Keep only the stable inventory fields plus a
        boolean reset marker so batching remains deterministic without changing
        the raw NPZ data.
        """
        if info is None:
            return None

        def _as_dict(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
                if isinstance(value, dict):
                    return value
            return {}

        normalized = []
        for value in np.asarray(info, dtype=object).reshape(-1):
            record = _as_dict(value)
            current_token = int(record.get("current_carrying_token", 0))
            next_token = int(
                record.get(
                    "next_carrying_token",
                    record.get("carrying_token", 0),
                )
            )
            normalized_record = {
                "current_carrying_key": bool(current_token > 0),
                "current_carrying_token": current_token,
                "carrying_key": bool(next_token > 0),
                "carrying_token": next_token,
                "next_carrying_key": bool(next_token > 0),
                "next_carrying_token": next_token,
                "uniform_reset": bool(record.get("uniform_reset", False)),
                # Keep the environment's transition-outcome label.  It is
                # deliberately not used as a model input: the stochastic
                # latent predicts it from state and requested action.
            }
            if "action_failed" in record:
                normalized_record["action_failed"] = bool(record["action_failed"])
            normalized.append(normalized_record)
        return np.asarray(normalized, dtype=object)

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
    3. Target construction: categorical MiniGrid/Crafter samples retain the
       absolute next frame; their semantic KEEP/SET_TO labels are derived in
       the loss. Continuous domains retain their normalized delta target.

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
        info = loaded.get('f', None) if env_type in ('with_obj', 'minigrid') else None
        # Crafter stores vector inventory in g/h. MiniGrid stores a compact,
        # colour-aware categorical inventory in transition metadata.
        inv = loaded.get('g', None) if env_type == 'crafter' else None
        inv_next = loaded.get('h', None) if env_type == 'crafter' else None
        replay_cfg = getattr(self.hparams, "crafter_transition_replay", None)
        if isinstance(replay_cfg, dict):
            crafter_transition_replay_enabled = bool(replay_cfg.get("enabled", False))
        else:
            crafter_transition_replay_enabled = bool(
                getattr(replay_cfg, "enabled", False)
            )
        if env_type == "crafter" and crafter_transition_replay_enabled:
            if inv is None or inv_next is None:
                raise ValueError(
                    "Crafter transition replay requires current inv/inv_next arrays"
                )
            inv_arr, inv_next_arr = np.asarray(inv), np.asarray(inv_next)
            if (
                inv_arr.ndim != 2
                or inv_next_arr.ndim != 2
                or inv_arr.shape != inv_next_arr.shape
                or inv_arr.shape[0] != len(obs)
                or inv_arr.shape[1] < 16
            ):
                raise ValueError(
                    "Crafter transition replay requires aligned inv/inv_next "
                    "arrays with shape (N, >=16)"
                )
        if env_type == 'minigrid':
            inv, inv_next = self._minigrid_inventory_from_info(info, done)

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
            crafter_transition_replay = crafter_transition_replay_enabled
            has_transition_metadata = (
                "_replay_transition_type" in replay_data
                and "_replay_bucket" in replay_data
            )
            if env_type == "crafter" and crafter_transition_replay and not has_transition_metadata:
                # The transition-aware path must never silently degrade to
                # natural replay: metadata is reconstructed by
                # FisherReplayBuffer.load_from_dict before this DataModule is
                # built.  A direct caller with a legacy dict should receive a
                # clear diagnostic instead of an unprotected training phase.
                raise ValueError(
                    "Crafter transition replay is enabled, but replay_data is "
                    "missing _replay_transition_type/_replay_bucket metadata; "
                    "load it through FisherReplayBuffer.load_from_dict() first."
                )
            if env_type == "crafter" and crafter_transition_replay:
                if isinstance(replay_cfg, dict):
                    include_changed_slot = bool(replay_cfg.get("include_changed_slot", False))
                else:
                    include_changed_slot = bool(
                        getattr(replay_cfg, "include_changed_slot", False)
                    )
                if include_changed_slot and "_replay_changed_slot_signature" not in replay_data:
                    raise ValueError(
                        "Crafter changed-slot replay is enabled, but replay_data is missing "
                        "_replay_changed_slot_signature; load it through "
                        "FisherReplayBuffer.load_from_dict() first."
                    )
            if env_type == "crafter" and crafter_transition_replay:
                replay_inv = replay_data.get("inv", None)
                replay_inv_next = replay_data.get("inv_next", None)
                if replay_inv is None or replay_inv_next is None:
                    raise ValueError(
                        "Crafter transition replay requires replay inv/inv_next arrays"
                    )
                replay_inv_arr = np.asarray(replay_inv)
                replay_inv_next_arr = np.asarray(replay_inv_next)
                if (
                    replay_inv_arr.ndim != 2
                    or replay_inv_next_arr.ndim != 2
                    or replay_inv_arr.shape != replay_inv_next_arr.shape
                    or replay_inv_arr.shape[0] != R
                    or replay_inv_arr.shape[1] < 16
                ):
                    raise ValueError(
                        "Crafter transition replay requires aligned replay "
                        "inv/inv_next arrays with shape (R, >=16)"
                    )
            # Sample at most `max_replay` items from the replay buffer.
            if R > max_replay and max_replay > 0:
                if env_type == "crafter" and crafter_transition_replay and has_transition_metadata:
                    idx = self._crafter_soft_replay_indices(replay_data, max_replay, rng)
                    # The mutable replay dict is an existing per-training-call transport
                    # object; retain diagnostics without changing the dataset contract.
                    replay_data["_replay_sampling_stats"] = self.replay_sampling_stats
                else:
                    idx = rng.choice(R, size=max_replay, replace=False)
                r_obs      = replay_data['obs'][idx]
                r_obs_next = replay_data['obs_next'][idx]
                r_act      = replay_data['act'][idx]
                r_info     = (replay_data['info'][idx]
                            if (env_type in ('with_obj', 'minigrid') and 'info' in replay_data and replay_data['info'] is not None)
                            else None)
                r_inv      = replay_data['inv'][idx] if ('inv' in replay_data and replay_data['inv'] is not None) else None
                r_inv_next = replay_data['inv_next'][idx] if ('inv_next' in replay_data and replay_data['inv_next'] is not None) else None
                if env_type == 'minigrid' and r_inv is None:
                    replay_done = replay_data.get('done', None)
                    replay_done = replay_done[idx] if replay_done is not None else None
                    r_inv, r_inv_next = self._minigrid_inventory_from_info(
                        r_info, replay_done
                    )
                
                # [Robustness] Handle cases where inventory is missing from replay data
                if r_inv is None and inv is not None and env_type != 'minigrid':
                    # Pad with zero inventory matching new data's feature dimension
                    inv_dim = inv.shape[-1]
                    r_inv = np.zeros((len(idx), inv_dim), dtype=np.float32)
                if r_inv_next is None and inv_next is not None and env_type != 'minigrid':
                    inv_dim = inv_next.shape[-1]
                    r_inv_next = np.zeros((len(idx), inv_dim), dtype=np.float32)
            else:
                r_obs, r_obs_next, r_act = replay_data['obs'], replay_data['obs_next'], replay_data['act']
                if env_type == "crafter" and crafter_transition_replay and has_transition_metadata:
                    self.replay_sampling_stats = self._crafter_replay_sampling_stats(
                        replay_data["_replay_transition_type"],
                        replay_data["act"],
                        replay_data.get("_replay_changed_slot_signature"),
                    )
                    replay_data["_replay_sampling_stats"] = self.replay_sampling_stats
                r_info = (replay_data['info']
                        if (env_type in ('with_obj', 'minigrid') and 'info' in replay_data and replay_data['info'] is not None)
                        else None)
                r_inv      = replay_data.get('inv', None)
                r_inv_next = replay_data.get('inv_next', None)
                if env_type == 'minigrid' and r_inv is None:
                    r_inv, r_inv_next = self._minigrid_inventory_from_info(
                        r_info, replay_data.get('done', None)
                    )
                
                # [Robustness] Handle whole-batch missing inventory
                if r_inv is None and inv is not None and env_type != 'minigrid':
                    inv_dim = inv.shape[-1]
                    r_inv = np.zeros((len(r_obs), inv_dim), dtype=np.float32)
                if r_inv_next is None and inv_next is not None and env_type != 'minigrid':
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
                if env_type in ('with_obj', 'minigrid') and r_info is not None:
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
                if env_type in ('with_obj', 'minigrid') and info is not None and len(info) == N:
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

        # Keep the latest frame available for continuous delta domains.
        obs_latest = obs[:, -C_base:] if (obs.ndim > 1 and obs.shape[1] > C_base) else obs

        # ===== (2) Build training targets =====
        transition_changed = None
        inventory_changed = None
        if self.hparams.data_type == 'discrete':
            change_axes = tuple(range(1, obs.ndim))
            transition_changed = np.any(obs != obs_next, axis=change_axes)
            if inv is not None and inv_next is not None:
                inv_change_axes = tuple(range(1, inv.ndim))
                if inv_change_axes:
                    inventory_changed = np.any(inv != inv_next, axis=inv_change_axes)
                else:
                    inventory_changed = inv != inv_next
                transition_changed |= inventory_changed

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
                act_bad = int(np.count_nonzero((act_f < 0) | (act_f >= MINIGRID_ACTION_COUNT)))
                if act_bad > 0:
                    raise ValueError(
                        f"[DataModule][MiniGrid] Invalid compact actions: bad={act_bad}; "
                        f"valid=[0, {MINIGRID_ACTION_COUNT - 1}]"
                    )
            obs_f = obs  # Keep discrete values unchanged for visualization/debugging.

            if env_type in ('crafter', 'minigrid'):
                # Categorical domains always retain the absolute following
                # frame. Their loss derives semantic effect labels from the
                # current/following pair; subtracting category IDs is invalid.
                obs_delta = obs_next.astype(np.float32)
            else:
                obs_delta = (obs_next.astype(np.int16) - obs_latest.astype(np.int16)).astype(np.float32)
                if getattr(self.hparams, "clip_discrete_delta", False):
                    np.clip(obs_delta, -1, 1, out=obs_delta)

        else:
            raise ValueError(f"Invalid data type: {self.hparams.data_type}")

        # ===== (3) Package the dataset =====
        data = {'obs': obs_f, 'obs_next': obs_delta, 'act': act_f}
        if transition_changed is not None:
            action_ids = act_f.reshape(len(act_f), -1)[:, 0].astype(np.int64)
            if env_type == 'minigrid':
                current_frame = obs[:, -3:]
                object_changed = np.any(current_frame[:, 0] != obs_next[:, 0], axis=(1, 2))
                color_changed = np.any(current_frame[:, 1] != obs_next[:, 1], axis=(1, 2))
                state_changed = np.any(current_frame[:, 2] != obs_next[:, 2], axis=(1, 2))
                inv_changed = (
                    np.any(inv != inv_next, axis=tuple(range(1, inv.ndim)))
                    if inv is not None and inv_next is not None
                    else np.zeros(len(obs), dtype=bool)
                )
                signature = (
                    object_changed.astype(np.int64)
                    | (color_changed.astype(np.int64) << 1)
                    | (state_changed.astype(np.int64) << 2)
                    | (inv_changed.astype(np.int64) << 3)
                )
                data['_sampling_bucket'] = action_ids * 16 + signature
            else:
                data['_sampling_bucket'] = action_ids * 2 + transition_changed.astype(np.int64)
        if env_type == 'minigrid' and self._minigrid_outcome_labels_required():
            if info is None:
                raise ValueError(
                    "[DataModule][MiniGrid] stochastic_latent.enabled requires "
                    "transition info with an action_failed label; recollect the dataset."
                )
            records = np.asarray(info, dtype=object).reshape(-1)
            if len(records) != len(obs):
                raise ValueError(
                    "[DataModule][MiniGrid] stochastic_latent.enabled requires "
                    "one transition info record per sample, including replay data; "
                    "recollect or rebuild the replay dataset. "
                    f"records={len(records)} samples={len(obs)}"
                )
            missing = []
            for index, value in enumerate(records):
                record = value.item() if isinstance(value, np.ndarray) and value.size == 1 else value
                if not isinstance(record, dict) or "action_failed" not in record:
                    missing.append(index)
            if missing:
                raise ValueError(
                    "[DataModule][MiniGrid] stochastic_latent.enabled requires "
                    "action_failed on every transition; recollect the dataset. "
                    f"missing={missing[:10]}"
                )
        if env_type == 'minigrid' and info is not None:
            info = self._normalize_minigrid_info_for_batch(info)
        if env_type in ('with_obj', 'minigrid') and info is not None:
            data['info'] = info
        if inv is not None:
            inventory_dtype = np.int64 if env_type == 'minigrid' else np.float32
            data['inv'] = inv[:len(obs_f)].astype(inventory_dtype)
            data['inv_next'] = inv_next[:len(obs_f)].astype(inventory_dtype)
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

        # DR checkpoint selection must use a fixed, non-target archive.  It is
        # deliberately constructed without replay and is never part of the
        # train subset or the Fisher loader.
        validation_base = data
        validation_archive = getattr(self.cfg, "dr_validation_archive", None)
        if validation_archive:
            validation_path = os.path.abspath(os.path.expanduser(str(validation_archive)))
            normalized = validation_path.replace("\\", "/").lower()
            if (
                "target_tasks" in normalized
                or "coverage_v2" in normalized
                or "coverage_v3" in normalized
            ):
                raise ValueError(
                    "DR checkpoint validation must not use target_tasks or controlled coverage archives"
                )
            if not os.path.isfile(validation_path):
                raise FileNotFoundError(
                    "Configured DR validation archive is missing: " + validation_path +
                    ". Build it from independent uniform-random, non-target DR archives with: "
                    "python trainer/analysis/build_crafter_dr_validation.py "
                    "--candidate <random_dr_archive.npz> --output " + validation_path +
                    " --seed <validation_seed> --training-seed <training_seed>"
                )
            validation_loaded = np.load(validation_path, allow_pickle=True)
            if "validation_provenance" not in validation_loaded.files or not np.all(
                validation_loaded["validation_provenance"] == "dr_uniform_random_non_target_v1"
            ):
                raise ValueError(
                    "DR validation archive lacks non-target uniform-random provenance; "
                    "build it with trainer/analysis/build_crafter_dr_validation.py"
                )
            if "training_seed" not in validation_loaded.files or np.any(
                validation_loaded["training_seed"] == validation_loaded["validation_seed"]
            ):
                raise ValueError("DR validation archive must record a distinct validation seed")
            configured_seed = getattr(self.cfg, "seed", None)
            if configured_seed is not None and np.any(
                validation_loaded["training_seed"] != int(configured_seed)
            ):
                raise ValueError(
                    "DR validation archive training_seed does not match the current run seed"
                )
            validation_base = WMRLDataset(validation_loaded, self.cfg, None)
            if len(validation_base) == 0:
                raise ValueError("DR validation archive is empty: " + validation_path)
            self.data_test = torch.utils.data.Subset(validation_base, range(len(validation_base)))
            # The archive replaces—not supplements—the historical 10% split.
            # Every current/replay transition remains eligible for training.
            self.data_train = torch.utils.data.Subset(data, range(len(data)))
            print(f"[DataModule] Using fixed non-target DR validation archive: {validation_path}")

        # Target archives are validation-only and can be much larger than is
        # useful for repeated MAC evaluation.  When requested by the caller,
        # select one deterministic random subset from the full archive.  A
        # fixed seed keeps exactly the same transitions across MAC iterations,
        # so changes in target loss reflect the model rather than resampling.
        max_validation_samples = int(
            getattr(self.cfg, "max_validation_samples", 0)
        )
        sample_from_full = bool(
            getattr(self.cfg, "validation_sample_from_full_dataset", False)
        )
        if max_validation_samples > 0:
            if sample_from_full:
                candidates = np.arange(len(validation_base), dtype=np.int64)
            else:
                candidates = np.asarray(self.data_test.indices, dtype=np.int64)
            if len(candidates) > max_validation_samples:
                subset_seed = int(
                    getattr(self.cfg, "validation_subset_seed", 0)
                )
                subset_rng = np.random.default_rng(subset_seed)
                selected = subset_rng.choice(
                    candidates,
                    size=max_validation_samples,
                    replace=False,
                )
                # Sorting does not change membership and makes traversal and
                # debugging reproducible as well as sampling reproducible.
                selected.sort()
                self.data_test = torch.utils.data.Subset(validation_base, selected.tolist())
                source = "full archive" if sample_from_full else "validation holdout"
                print(
                    f"[DataModule] Fixed validation subset: "
                    f"{max_validation_samples}/{len(candidates)} samples from "
                    f"{source} (seed={subset_seed})."
                )

    def train_dataloader(self):
        num_workers = int(getattr(self.cfg, "n_cpu", 0))
        sampler = None
        sample_weights = None
        dataset = self.data_train.dataset
        indices = np.asarray(self.data_train.indices, dtype=np.int64)
        if (
            getattr(self.data_train.dataset, "_minigrid_stochastic_latent_enabled", lambda: False)()
            and bool(getattr(self.cfg, "transition_balanced_sampling", False))
        ):
            raise ValueError(
                "[DataModule][MiniGrid] stochastic_latent requires natural transition "
                "sampling; set transition_balanced_sampling=false."
            )
        if bool(getattr(self.cfg, "transition_balanced_sampling", False)):
            buckets = dataset.data.get('_sampling_bucket', None)
            if buckets is not None:
                train_buckets = np.asarray(buckets)[indices]
                _, inverse, counts = np.unique(
                    train_buckets, return_inverse=True, return_counts=True
                )
                sample_weights = 1.0 / counts[inverse].astype(np.float64)
                print(
                    f"[DataModule] Transition-balanced training over "
                    f"{len(counts)} non-empty (action, changed) buckets."
                )

        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(indices),
                replacement=True,
            )
        return DataLoader(
            self.data_train, 
            batch_size=self.cfg.batch_size, 
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0)
        )

    def fisher_dataloader(self, samples: int):
        """A deterministic Fisher loader that covers each observed Crafter slot."""
        train_indices = np.asarray(self.data_train.indices, dtype=np.int64)
        selected = np.empty(0, dtype=np.int64)
        self.fisher_sampling_stats = {}
        cfg = getattr(self.cfg, "fisher_slot_stratified", None)
        enabled = bool(cfg.get("enabled", False) if isinstance(cfg, dict) else getattr(cfg, "enabled", False))
        per_slot = int(cfg.get("min_samples_per_observed_slot", 16) if isinstance(cfg, dict) else getattr(cfg, "min_samples_per_observed_slot", 16)) if cfg is not None else 16
        if enabled and str(getattr(self.cfg, "env_type", "")) == "crafter":
            data = self.data_train.dataset.data
            inv, inv_next = data.get("inv"), data.get("inv_next")
            if inv is None or inv_next is None:
                raise ValueError("fisher_slot_stratified requires Crafter inv/inv_next")
            sig = np.sum(
                (np.asarray(inv)[train_indices, 4:16] != np.asarray(inv_next)[train_indices, 4:16]).astype(np.uint16)
                * (1 << np.arange(12, dtype=np.uint16))[None, :], axis=1, dtype=np.uint16
            )
            positions = []
            frequencies = np.asarray([np.count_nonzero(sig & (1 << slot)) for slot in range(12)])
            rng = np.random.default_rng(int(getattr(self.cfg, "seed", 0)))
            for slot in np.argsort(frequencies, kind="stable"):
                if frequencies[slot] == 0 or len(positions) >= samples:
                    continue
                already_covered = int(np.count_nonzero(
                    sig[np.asarray(positions, dtype=np.int64)] & (1 << slot)
                )) if positions else 0
                needed = max(0, per_slot - already_covered)
                if needed == 0:
                    continue
                candidates = np.flatnonzero((sig & (1 << slot)) != 0)
                candidates = candidates[~np.isin(candidates, positions)]
                take = min(needed, len(candidates), samples - len(positions))
                if take:
                    positions.extend(rng.choice(candidates, size=take, replace=False).tolist())
            selected = train_indices[np.asarray(positions, dtype=np.int64)]
        remaining = min(int(samples) - len(selected), len(train_indices) - len(selected))
        if remaining > 0:
            available = np.setdiff1d(train_indices, selected, assume_unique=False)
            rng = np.random.default_rng(int(getattr(self.cfg, "seed", 0)) + 7919)
            selected = np.concatenate((selected, rng.choice(available, size=remaining, replace=False)))
        if enabled and str(getattr(self.cfg, "env_type", "")) == "crafter":
            all_inv = np.asarray(self.data_train.dataset.data["inv"])[selected, 4:16]
            all_next = np.asarray(self.data_train.dataset.data["inv_next"])[selected, 4:16]
            final_sig = np.sum(
                (all_inv != all_next).astype(np.uint16) * (1 << np.arange(12, dtype=np.uint16))[None, :],
                axis=1, dtype=np.uint16,
            )
            self.fisher_sampling_stats = {
                int(slot + 4): int(np.count_nonzero(final_sig & (1 << slot)))
                for slot in range(12) if np.any(final_sig & (1 << slot))
            }
        return DataLoader(
            torch.utils.data.Subset(self.data_train.dataset, selected.tolist()),
            batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=int(getattr(self.cfg, "n_cpu", 0)), pin_memory=True,
            persistent_workers=bool(int(getattr(self.cfg, "n_cpu", 0)) > 0),
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
