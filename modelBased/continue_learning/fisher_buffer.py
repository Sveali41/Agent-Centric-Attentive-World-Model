import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from domain.minigrid import minigrid_support as minigrid_utils
from domain.minigrid.transition_codec import minigrid_effect_cell_nll
import random
import os
import tempfile


class FisherReplayBuffer:
    """Replay storage with an optional Crafter transition-aware admission path."""
    CRAFTER_TRANSITION_TYPES = (
        "inventory_change", "map_interaction", "agent_dynamics", "static",
    )

    def __init__(
        self,
        max_size,
        contact_positive_ratio=0.5,
        crafter_transition_replay=None,
        seed=0,
    ):
        self.buffer = []
        self.max_size = max_size
        self.mask_size = 3  # Cross mask size
        self.contact_positive_ratio = contact_positive_ratio  # Ratio of contact=1 samples in label-balanced sampling
        replay_cfg = crafter_transition_replay or {}
        if isinstance(replay_cfg, dict):
            get_cfg = replay_cfg.get
        else:
            get_cfg = lambda key, default=None: getattr(replay_cfg, key, default)
        self.crafter_transition_replay_enabled = bool(get_cfg("enabled", False))
        self.crafter_transition_replay_exponent = float(get_cfg("balance_exponent", 0.5))
        self.crafter_transition_replay_include_action = bool(get_cfg("include_action", True))
        # Opt-in refinement: distinguish observed item-change signatures inside
        # the inventory-change class without duplicating a transition.
        self.crafter_transition_replay_include_changed_slot = bool(
            get_cfg("include_changed_slot", False)
        )
        if self.crafter_transition_replay_exponent < 0:
            raise ValueError("crafter transition replay balance_exponent must be non-negative")
        # Keep replay selection reproducible without perturbing process-global RNG state.
        self._rng = np.random.default_rng(int(seed))

    @staticmethod
    def _crafter_object_map(batch):
        """Return the object plane from a NCHW or NHWC Crafter batch."""
        arr = np.asarray(batch)
        if arr.ndim != 4:
            raise ValueError(f"Crafter transition replay expects 4D observations, got {arr.shape}")
        if arr.shape[1] <= 4 and arr.shape[-1] > 4:  # NCHW
            return arr[:, 0], arr, "nchw"
        if arr.shape[-1] <= 4 and arr.shape[1] > 4:  # NHWC
            return arr[..., 0], arr, "nhwc"
        raise ValueError(
            "Cannot infer Crafter observation layout; expected NCHW/NHWC with <=4 channels, "
            f"got {arr.shape}"
        )

    @classmethod
    def classify_crafter_transitions(cls, samples: Dict) -> np.ndarray:
        """Classify real Crafter outcomes; 0..3 match ``CRAFTER_TRANSITION_TYPES``."""
        required = ("obs", "obs_next", "act", "inv", "inv_next")
        missing = [key for key in required if key not in samples or samples[key] is None]
        if missing:
            raise ValueError(
                "Crafter transition replay requires " + ", ".join(required) +
                f"; missing {missing}"
            )
        obj, obs, layout = cls._crafter_object_map(samples["obs"])
        obj_next, obs_next, next_layout = cls._crafter_object_map(samples["obs_next"])
        inv = np.asarray(samples["inv"])
        inv_next = np.asarray(samples["inv_next"])
        act = np.asarray(samples["act"])
        n = len(obj)
        if next_layout != layout or len(obj_next) != n or len(inv) != n or len(inv_next) != n or len(act) != n:
            raise ValueError("Crafter transition replay fields must have aligned batch lengths/layouts")
        if inv.ndim != 2 or inv_next.ndim != 2 or inv.shape != inv_next.shape or inv.shape[1] < 16:
            raise ValueError(
                "Crafter transition replay requires aligned (B, >=16) inv/inv_next arrays"
            )

        # Rule-tracked physiological slots 0:4 intentionally do not define a critical event.
        inventory_changed = np.any(inv[:, 4:16] != inv_next[:, 4:16], axis=1)
        types = np.full(n, 3, dtype=np.int8)  # static
        types[inventory_changed] = 0

        for i in range(n):
            current_player = obj[i] == 13
            next_player = obj_next[i] == 13
            # Object interactions are evaluated after removing either agent footprint.
            non_agent_changed = np.any(
                (obj[i] != obj_next[i]) & ~current_player & ~next_player
            )
            if not inventory_changed[i] and non_agent_changed:
                types[i] = 1
                continue
            if inventory_changed[i]:
                continue

            current_pos = np.argwhere(current_player)
            next_pos = np.argwhere(next_player)
            position_changed = (
                len(current_pos) != len(next_pos)
                or (len(current_pos) > 0 and not np.array_equal(current_pos, next_pos))
            )
            direction_changed = False
            if not position_changed and len(current_pos) == 1:
                y, x = current_pos[0]
                if layout == "nchw":
                    direction_changed = not np.array_equal(obs[i, :, y, x], obs_next[i, :, y, x])
                else:
                    direction_changed = not np.array_equal(obs[i, y, x, :], obs_next[i, y, x, :])
            if position_changed or direction_changed:
                types[i] = 2
        return types

    def _crafter_buckets(self, transition_types, actions):
        return self._crafter_buckets_with_signatures(transition_types, actions, None)

    @staticmethod
    def crafter_changed_slot_signatures(samples: Dict) -> np.ndarray:
        """Bit-mask of changed item slots (local indices 0..11); survival is excluded."""
        inv = np.asarray(samples["inv"])
        inv_next = np.asarray(samples["inv_next"])
        if inv.ndim != 2 or inv_next.ndim != 2 or inv.shape != inv_next.shape or inv.shape[1] < 16:
            raise ValueError("Crafter replay requires aligned (B, >=16) inv/inv_next arrays")
        changed = inv[:, 4:16] != inv_next[:, 4:16]
        bit_values = (1 << np.arange(12, dtype=np.uint16))[None, :]
        return np.sum(changed.astype(np.uint16) * bit_values, axis=1, dtype=np.uint16)

    def _crafter_buckets_with_signatures(self, transition_types, actions, signatures):
        actions = np.asarray(actions).reshape(-1)
        if len(transition_types) != len(actions):
            raise ValueError("transition type and action lengths must match")
        # The persisted joint bucket remains stable even if sampling ignores actions.
        base = np.asarray(transition_types, dtype=np.int64) * 17 + actions.astype(np.int64)
        if not self.crafter_transition_replay_include_changed_slot:
            return base
        if signatures is None:
            raise ValueError("changed-slot replay requires a signature per transition")
        signatures = np.asarray(signatures, dtype=np.int64).reshape(-1)
        if len(signatures) != len(base) or np.any(signatures < 0) or np.any(signatures >= (1 << 12)):
            raise ValueError("Crafter changed-slot signatures must be aligned 12-bit values")
        return base * (1 << 12) + signatures

    def _crafter_sampling_buckets(self, transition_types, actions, signatures, persisted_buckets):
        if not self.crafter_transition_replay_include_changed_slot:
            return persisted_buckets if self.crafter_transition_replay_include_action else persisted_buckets // 17
        types = np.asarray(transition_types, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        signatures = np.asarray(signatures, dtype=np.int64).reshape(-1)
        base = types * 17 + actions if self.crafter_transition_replay_include_action else types
        return base * (1 << 12) + signatures

    def _soft_balanced_indices(self, indices, buckets, quota):
        indices = np.asarray(indices, dtype=np.int64)
        if quota >= len(indices):
            return indices.copy()
        if quota <= 0 or len(indices) == 0:
            return np.empty(0, dtype=np.int64)
        # Callers supply persisted buckets only; derive legacy sampling here.
        sample_buckets = buckets if self.crafter_transition_replay_include_action else buckets // 17
        _, inverse, counts = np.unique(sample_buckets, return_inverse=True, return_counts=True)
        weights = counts[inverse].astype(np.float64) ** (-self.crafter_transition_replay_exponent)
        weights /= weights.sum()
        return self._rng.choice(indices, size=quota, replace=False, p=weights)

    def transition_replay_stats(self, samples=None) -> Dict[str, Dict]:
        """Counts suitable for DR diagnostics; accepts raw samples or the live buffer."""
        if samples is None:
            if not self.buffer:
                return {name: {"count": 0, "actions": {}} for name in self.CRAFTER_TRANSITION_TYPES}
            samples = self.export_dict()
        if "_replay_transition_type" in samples:
            types = np.asarray(samples["_replay_transition_type"], dtype=np.int64)
        else:
            types = self.classify_crafter_transitions(samples)
        actions = np.asarray(samples["act"]).reshape(-1).astype(np.int64)
        signatures = (
            np.asarray(samples["_replay_changed_slot_signature"], dtype=np.uint16)
            if "_replay_changed_slot_signature" in samples
            else self.crafter_changed_slot_signatures(samples)
        )
        stats = {}
        for type_id, name in enumerate(self.CRAFTER_TRANSITION_TYPES):
            selected = actions[types == type_id]
            stats[name] = {
                "count": int(len(selected)),
                "actions": {int(action): int(count) for action, count in zip(*np.unique(selected, return_counts=True))},
                "changed_slots": {
                    int(slot + 4): int(np.sum((signatures[types == type_id] & (1 << slot)) != 0))
                    for slot in range(12)
                    if np.any((signatures[types == type_id] & (1 << slot)) != 0)
                },
            }
        return stats

    @staticmethod
    def _sample_at(samples: Dict, index: int) -> Dict:
        item = {}
        for key, value in samples.items():
            if value is None:
                continue
            try:
                item[key] = value[index]
            except Exception:
                continue
        return item

    @staticmethod
    def _pad_single_map_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """
        Pad/crop one map sample to target (H, W), supporting HWC or CHW layout.
        Non-image tensors are returned unchanged.
        """
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            return arr

        max_h, max_w = target_shape

        # HWC: (H, W, C) with small C (2/3/4)
        if arr.shape[-1] <= 4 and arr.shape[0] > 4:
            h, w, _ = arr.shape
            out = arr[:max_h, :max_w, :]
            pad_h = max_h - out.shape[0]
            pad_w = max_w - out.shape[1]
            if pad_h > 0 or pad_w > 0:
                out = np.pad(out, ((0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)), mode='constant', constant_values=0)
            return out

        # CHW: (C, H, W) with small C (2/3/4)
        if arr.shape[0] <= 4 and arr.shape[-1] > 4:
            _, h, w = arr.shape
            out = arr[:, :max_h, :max_w]
            pad_h = max_h - out.shape[1]
            pad_w = max_w - out.shape[2]
            if pad_h > 0 or pad_w > 0:
                out = np.pad(out, ((0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0))), mode='constant', constant_values=0)
            return out

        return arr

    def harmonize_buffer_map_shape(self, target_shape: tuple[int, int]) -> int:
        """
        In-place pad/crop existing replay samples (obs/obs_next) to target shape.
        Returns number of sample fields changed.
        """
        changed = 0
        for sample in self.buffer:
            for k in ("obs", "obs_next"):
                if k not in sample:
                    continue
                old = sample[k]
                new = self._pad_single_map_to_shape(old, target_shape)
                if isinstance(old, np.ndarray) and isinstance(new, np.ndarray) and new.shape != old.shape:
                    sample[k] = new
                    changed += 1
        return changed

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
            batch = {
                "obs": torch.tensor(samples["obs"]).to(device),
                "act": torch.tensor(samples["act"]).to(device),
                "obs_next": torch.tensor(samples["obs_next"]).to(device),
            }
            if "info" in samples and samples["info"] is not None:
                batch["info"] = torch.tensor(samples["info"]).to(device)
            if "inv" in samples and samples["inv"] is not None:
                batch["inv"] = torch.tensor(samples["inv"]).to(device)
            if "inv_next" in samples and samples["inv_next"] is not None:
                batch["inv_next"] = torch.tensor(samples["inv_next"]).to(device)

            if hasattr(model, "preprocess_batch"):
                obs_masked, act, obs_next_masked, info, _, _, inv, _ = model.preprocess_batch(
                    batch, training=False
                )
                pred, _, aux_pred = model(obs_masked, act, info, inv=inv)
                pred_for_loss = pred
            else:
                obs = batch["obs"].float()
                act = batch["act"]
                obs_next = batch["obs_next"].float()
                info = batch.get("info", None)
                agent_postion_yx_batch = minigrid_utils.get_agent_position(obs)
                obs_masked = minigrid_utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
                obs_next_masked = minigrid_utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch)
                pred_for_loss, _ = model(obs_masked, act, info)

            if getattr(model, "env_type", "") == "crafter":
                from domain.crafter.crafter_support import (
                    CRAFTER_EFFECT_CHANNELS,
                    crafter_classification_loss,
                    crafter_effect_loss,
                )
                if pred_for_loss.shape[1] == CRAFTER_EFFECT_CHANNELS:
                    per_cell = crafter_effect_loss(
                        pred_for_loss,
                        obs_masked,
                        obs_next_masked,
                        reduction="none",
                    )
                else:
                    per_cell = crafter_classification_loss(
                        pred_for_loss,
                        obs_next_masked,
                        reduction="none",
                        weighted=False,
                    )
                loss = per_cell.mean(dim=(1, 2)).detach().cpu().tolist()
            elif getattr(model, "env_type", "") == "minigrid":
                per_cell, _ = minigrid_effect_cell_nll(
                    pred_for_loss,
                    obs_masked,
                    obs_next_masked,
                    mode=getattr(model, "minigrid_transition_mode", "absolute"),
                )
                loss = per_cell.mean(dim=(1, 2)).detach().cpu().tolist()
            else:
                loss = [F.mse_loss(pred_for_loss[i], obs_next_masked[i]).item() for i in range(len(pred_for_loss))]

            # Categorical MiniGrid IDs are labels, not continuous magnitudes;
            # never rank replay samples by numeric ID differences.
            score = loss
        
        scored_samples = list(zip(
            score,
            [self._sample_at(samples, i) for i in range(len(score))],
        ))
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
        - ratio: sample this fraction from the current samples
        - static_ratio: desired fraction of static samples among selected items

        Static samples: `obs_next` is exactly identical to `obs`.
        Dynamic samples: at least one position differs.
        """
        total_len = len(samples['obs'])
        if total_len == 0:
            return

        insert_k = int(total_len * ratio)
        if insert_k <= 0:
            return

        # === Detect changed positions ===
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
            selected.append(self._sample_at(samples, i))

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
        Return a boolean mask indicating which samples place the agent next to
        key/door/lava elements.
        The agent is identified from the object map.
        obs: Tensor of shape (B, C, H, W) or (B, H, W, C)
        return: BoolTensor of shape (B,)
        """
        if obs.dim() != 4:
            raise ValueError(f"Expected 4D image batch, got {tuple(obs.shape)}")
        channel_first = obs.shape[1] in (2, 3, 4) and obs.shape[-1] > 4
        channel_last = obs.shape[-1] in (2, 3, 4) and obs.shape[1] > 4
        if channel_first:
            obj_map = obs[:, 0]
        elif channel_last:
            obj_map = obs[..., 0]
        else:
            raise ValueError(
                f"Cannot infer image layout for saliency mask: {tuple(obs.shape)}"
            )

        B, H, W = obj_map.shape
        near_mask = torch.zeros(B, dtype=torch.bool, device=obs.device)

        for b in range(B):
            # ID=13 for Crafter player, ID=10 for MiniGrid player
            if (obj_map[b] == 13).any():
                player_id = 13
                # Crafter interactive: Water(1), Tree(6), Stone(3/4), Coal(8), Iron(9), Diamond(10), Table(11), Furnace(12), Cow(14), Plant(18)
                # Note: ID 3/4 are Stone/Path in Crafter, ID 13 is Player
                interactive_ids = [1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 18]
            else:
                player_id = 10
                # MiniGrid interactive: door(4), key(5), lava(9)
                interactive_ids = [4, 5, 9]

            agent_pos = (obj_map[b] == player_id).nonzero(as_tuple=False)
            if agent_pos.numel() == 0:
                continue

            y, x = agent_pos[0]  # Assume a single agent.
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
                if val.item() in interactive_ids:
                    near_mask[b] = True
                    break

        return near_mask  # (B,)

    def update_combined(self, samples, current_sample_ratio=0.5, fisher_buffer_elements_ratio=0.9, target_shape=None):
        """
        Combined insertion strategy based on the current sample count:
        1. draw a `ratio` fraction from `samples`
        2. reserve a configured portion for key/door or salient-element samples
        """
        # Calculate how many samples to add based on ratio of current buffer size
        # But ensure we add at least some if buffer is empty
        
        total_len = len(samples['obs'])
        if total_len == 0:
            return

        total_quota = int(total_len * current_sample_ratio)
        if total_quota <= 0:
            return

        if self.crafter_transition_replay_enabled:
            self._update_crafter_transition_aware(samples, total_quota)
            return

        # === Part 1: salient element samples ===
        obs = samples['obs']
        is_vector_obs = (
            (isinstance(obs, np.ndarray) and obs.ndim == 2) or
            (isinstance(obs, torch.Tensor) and obs.ndim == 2)
        )
        
        # --- [Optional Padding Logic] ---
        # Only pad when target_shape is explicitly provided.
        # This avoids silently mangling NHWC data (e.g., MiniGrid rollouts from run_env).
        if target_shape is not None and isinstance(obs, np.ndarray) and not is_vector_obs:
            MAX_H, MAX_W = target_shape

            def _detect_layout(arr: np.ndarray) -> str:
                # Returns "nchw" or "nhwc" for 4-D arrays.
                if arr.ndim != 4:
                    raise ValueError(f"Expected 4D array for layout detection, got shape={arr.shape}")
                # Common case: NHWC image tensors (B, H, W, C) where C is small (2/3/4)
                if arr.shape[-1] <= 4 and arr.shape[1] > 4:
                    return "nhwc"
                # Common case: NCHW image tensors (B, C, H, W) where C is small (2/3/4)
                if arr.shape[1] <= 4 and arr.shape[-1] > 4:
                    return "nchw"
                # Fallback: treat as NCHW (historical default in this module).
                return "nchw"

            def pad_maps(maps_array: np.ndarray) -> np.ndarray:
                layout = _detect_layout(maps_array)
                if layout == "nhwc":
                    _, H, W, _ = maps_array.shape
                    pad_h = MAX_H - H
                    pad_w = MAX_W - W
                    if pad_h < 0 or pad_w < 0:
                        return maps_array[:, :MAX_H, :MAX_W, :]
                    if pad_h == 0 and pad_w == 0:
                        return maps_array
                    return np.pad(
                        maps_array,
                        ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                # layout == "nchw"
                _, _, H, W = maps_array.shape
                pad_h = MAX_H - H
                pad_w = MAX_W - W
                if pad_h < 0 or pad_w < 0:
                    return maps_array[:, :, :MAX_H, :MAX_W]
                if pad_h == 0 and pad_w == 0:
                    return maps_array
                return np.pad(
                    maps_array,
                    ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            samples['obs'] = pad_maps(samples['obs'])
            if isinstance(samples.get('obs_next', None), np.ndarray):
                samples['obs_next'] = pad_maps(samples['obs_next'])
            obs = samples['obs']

        if is_vector_obs:
            obs_tensor = torch.tensor(obs) if not isinstance(obs, torch.Tensor) else obs
            if obs_tensor.shape[-1] == 24:
                # BipedalWalker explicitly: indices 18-24 corresponds to front lidar sensors
                lidar_readings = obs_tensor[..., 18:24]
                # A min distance below 0.8 typically means there's an obstacle ahead (stump, stairs, pit edge)
                near_mask = lidar_readings.min(dim=-1)[0] < 0.8
                
                # Contact label mask (index 8 is leg_1, index 13 is leg_2).
                contact_mask = (obs_tensor[..., 8] > 0.5) | (obs_tensor[..., 13] > 0.5)
                no_contact_mask = ~contact_mask

                near_indices_all = torch.where(near_mask)[0].cpu().numpy()
                contact_indices_all = torch.where(contact_mask)[0].cpu().numpy()
                no_contact_indices_all = torch.where(no_contact_mask)[0].cpu().numpy()
            else:
                near_indices_all = np.array([], dtype=int)
                contact_indices_all = np.array([], dtype=int)
                no_contact_indices_all = np.array([], dtype=int)
        else:
            obs_tensor = torch.tensor(obs) if not isinstance(obs, torch.Tensor) else obs
            try:
                near_elements_mask = self.get_agent_near_elements_mask(obs_tensor)
                near_indices_all = torch.where(near_elements_mask)[0].cpu().numpy()
            except Exception as e:
                print("Error computing near_elements_mask:", e)
                near_indices_all = np.array([], dtype=int)

        elements_quota = int(total_quota * fisher_buffer_elements_ratio)
        elements_selected = []
        if len(near_indices_all) > 0 and elements_quota > 0:
            pick_n = min(elements_quota, len(near_indices_all))
            elements_selected = np.random.choice(near_indices_all, pick_n, replace=False).tolist()
        
        # === Part 2: fill the remaining quota with random samples, optionally balancing contact labels ===
        remaining_quota = total_quota - len(elements_selected)
        total_indices = list(range(total_len))
        non_elements_pool = [i for i in total_indices if i not in elements_selected]
        
        random_selected = []
        if is_vector_obs and 'contact_indices_all' in locals() and len(contact_indices_all) > 0:
            # Split the pool into contact-positive and contact-negative samples.
            pool_contact = [i for i in non_elements_pool if i in contact_indices_all]
            pool_no_contact = [i for i in non_elements_pool if i in no_contact_indices_all]
            random.shuffle(pool_contact)
            random.shuffle(pool_no_contact)
            
            # Allocate samples according to the configured contact-positive ratio.
            contact_quota = int(remaining_quota * self.contact_positive_ratio)
            pick_c = min(contact_quota, len(pool_contact))
            # Reassign unused quota if one side does not have enough samples.
            pick_nc = min(remaining_quota - pick_c, len(pool_no_contact))
            # Rebalance once more in the opposite direction if capacity remains.
            pick_c = min(remaining_quota - pick_nc, len(pool_contact)) 
            
            random_selected.extend(pool_contact[:pick_c])
            random_selected.extend(pool_no_contact[:pick_nc])
            
            # If quota still remains, fill it from the leftover pool at random.
            leftover = remaining_quota - len(random_selected)
            if leftover > 0:
                left_pool = [i for i in non_elements_pool if i not in random_selected]
                random.shuffle(left_pool)
                random_selected.extend(left_pool[:leftover])
        else:
            random.shuffle(non_elements_pool)
            random_selected = non_elements_pool[:remaining_quota]

        # === Merge selections and shuffle ===
        all_selected_indices = elements_selected + random_selected
        random.shuffle(all_selected_indices)

        selected = []
        for i in all_selected_indices:
            selected.append(self._sample_at(samples, i))

        self.buffer.extend(selected)

        # === Trim the buffer back to capacity ===
        if len(self.buffer) > self.max_size:
            num_to_remove = len(self.buffer) - self.max_size
            all_indices = list(range(len(self.buffer)))
            indices_to_remove = np.random.choice(all_indices, size=num_to_remove, replace=False)
            indices_to_keep = sorted(list(set(all_indices) - set(indices_to_remove)))
            self.buffer = [self.buffer[i] for i in indices_to_keep]

    def _update_crafter_transition_aware(self, samples: Dict, total_quota: int):
        """Admission for Crafter DR: retain real critical outcomes before fillers."""
        transition_types = self.classify_crafter_transitions(samples)
        signatures = self.crafter_changed_slot_signatures(samples)
        buckets = self._crafter_buckets_with_signatures(
            transition_types, samples["act"], signatures
        )
        critical = np.flatnonzero(transition_types <= 1)
        noncritical = np.flatnonzero(transition_types >= 2)

        if len(critical) <= total_quota:
            selected = critical.copy()
            remaining = total_quota - len(selected)
            # Fill from movement/static without giving proxy salience any role.
            if remaining:
                selected = np.concatenate([
                    selected,
                    self._rng.choice(
                        noncritical, size=min(remaining, len(noncritical)), replace=False
                    ),
                ])
        else:
            # Critical overflow is the only case where a critical outcome can be dropped.
            selected = self._crafter_soft_balanced_indices(
                critical, transition_types, samples["act"], signatures, buckets, total_quota
            )

        if len(selected) > 1:
            selected = self._rng.permutation(selected)
        for index in selected:
            sample = self._sample_at(samples, int(index))
            sample["_replay_transition_type"] = np.int8(transition_types[index])
            sample["_replay_changed_slot_signature"] = np.uint16(signatures[index])
            sample["_replay_bucket"] = np.int64(buckets[index])
            self.buffer.append(sample)

        # Preserve legacy buffer capacity semantics, but use the local RNG.
        if len(self.buffer) > self.max_size:
            # Unlike the legacy buffer, do not randomly erase rare observed
            # outcomes after admission.  Capacity retention uses the same
            # persisted transition/action buckets as replay sampling.
            buckets = np.asarray(
                [sample["_replay_bucket"] for sample in self.buffer], dtype=np.int64
            )
            all_types = np.asarray([sample["_replay_transition_type"] for sample in self.buffer])
            all_actions = np.asarray([sample["act"] for sample in self.buffer])
            all_signatures = np.asarray([
                sample.get("_replay_changed_slot_signature", 0) for sample in self.buffer
            ])
            keep = self._crafter_soft_balanced_indices(
                np.arange(len(self.buffer), dtype=np.int64), all_types, all_actions,
                all_signatures, buckets, self.max_size
            )
            self.buffer = [self.buffer[int(i)] for i in keep]

    def _crafter_soft_balanced_indices(self, indices, types, actions, signatures, buckets, quota):
        indices = np.asarray(indices, dtype=np.int64)
        if quota >= len(indices):
            return indices.copy()
        if quota <= 0 or len(indices) == 0:
            return np.empty(0, dtype=np.int64)
        sample_buckets = self._crafter_sampling_buckets(
            np.asarray(types)[indices], np.asarray(actions)[indices],
            np.asarray(signatures)[indices], np.asarray(buckets)[indices]
        )
        _, inverse, counts = np.unique(sample_buckets, return_inverse=True, return_counts=True)
        weights = counts[inverse].astype(np.float64) ** (-self.crafter_transition_replay_exponent)
        weights /= weights.sum()
        return self._rng.choice(indices, size=quota, replace=False, p=weights)

    def add_from_npz(self, path, current_sample_ratio=0.05, fisher_buffer_elements_ratio=0.5, target_shape=None):
        """Helper to load data from npz and add to buffer using update_combined."""
        if not os.path.exists(path):
            print(f"[FisherBuffer] Warning: File not found {path}")
            return
        
        try:
            data = np.load(path, allow_pickle=True)
            # Support both letter keys (a, b, c...) and descriptive string keys
            key_map = {
                'a': 'obs', 'b': 'obs_next', 'c': 'act', 'd': 'rew', 'e': 'done', 'f': 'info', 'g': 'inv', 'h': 'inv_next'
            }
            samples = {}
            for k in data.files:
                actual_k = key_map.get(k, k)
                samples[actual_k] = data[k]
            
            # Additional safety for missing expected keys
            if 'obs' not in samples and 'a' not in data.files:
                 print(f"[FisherBuffer] Warning: No 'obs' or 'a' key found in {path}")
            
            self.update_combined(
                samples, 
                current_sample_ratio=current_sample_ratio,
                fisher_buffer_elements_ratio=fisher_buffer_elements_ratio,
                target_shape=target_shape
            )
            print(f"[FisherBuffer] Added samples from {os.path.basename(path)}. Buffer size: {len(self.buffer)}")
        except Exception as e:
            if self.crafter_transition_replay_enabled:
                raise
            print(f"[FisherBuffer] Error loading {path}: {e}")

    def add_from_batch(
        self,
        batch: Dict,
        current_sample_ratio=0.05,
        fisher_buffer_elements_ratio=0.5,
        target_shape=None
    ):
        """
        Compatibility wrapper used by trainer baselines.
        Accepts either canonical keys (obs/obs_next/act/...) or legacy short keys (a/b/c/...).
        """
        if batch is None:
            return

        key_map = {
            "a": "obs",
            "b": "obs_next",
            "c": "act",
            "d": "rew",
            "e": "done",
            "f": "info",
            "g": "inv",
            "h": "inv_next",
        }
        samples = {}
        for k, v in batch.items():
            samples[key_map.get(k, k)] = v

        required = ("obs", "obs_next", "act")
        if any(k not in samples for k in required):
            missing = [k for k in required if k not in samples]
            raise KeyError(f"add_from_batch missing required keys: {missing}")

        self.update_combined(
            samples,
            current_sample_ratio=current_sample_ratio,
            fisher_buffer_elements_ratio=fisher_buffer_elements_ratio,
            target_shape=target_shape,
        )




    def export_dict(self) -> Dict[str, np.ndarray]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        required = {"obs", "obs_next", "act"}
        available = set(self.buffer[0].keys())
        for sample in self.buffer[1:]:
            available.intersection_update(sample.keys())
        missing = required - available
        if missing:
            raise ValueError(f"Replay samples are missing required keys: {sorted(missing)}")

        # Optional fields are exported only when every sample has them.  This
        # keeps arrays aligned and avoids object arrays containing a mixture of
        # real values and None after several curriculum phases.
        exported = {}
        for key in sorted(available):
            try:
                exported[key] = np.stack([sample[key] for sample in self.buffer])
            except (ValueError, TypeError):
                # Scalar/object metadata can still be represented as an object
                # array, provided it has one entry per transition.
                exported[key] = np.asarray([sample[key] for sample in self.buffer], dtype=object)
        return exported

    def load_from_dict(self, data_dict: Dict[str, np.ndarray]):
        self.buffer = []
        required = ("obs", "obs_next", "act")
        missing = [key for key in required if key not in data_dict]
        if missing:
            raise KeyError(f"Replay data is missing required keys: {missing}")
        length = len(data_dict['obs'])
        for key in required:
            if len(data_dict[key]) != length:
                raise ValueError(f"Replay field '{key}' has inconsistent length")

        # Checkpoint/replay compatibility: old Crafter buffers have no replay
        # metadata.  In opt-in mode rebuild it from the actual transition;
        # never silently fall back to natural replay sampling.
        if self.crafter_transition_replay_enabled and (
            "_replay_transition_type" not in data_dict
            or "_replay_bucket" not in data_dict
        ):
            reconstructed = self.classify_crafter_transitions(data_dict)
            data_dict = dict(data_dict)
            data_dict["_replay_transition_type"] = reconstructed
            if self.crafter_transition_replay_include_changed_slot:
                signatures = self.crafter_changed_slot_signatures(data_dict)
                data_dict["_replay_changed_slot_signature"] = signatures
                data_dict["_replay_bucket"] = self._crafter_buckets_with_signatures(
                    reconstructed, data_dict["act"], signatures
                )
            else:
                data_dict["_replay_bucket"] = self._crafter_buckets(
                    reconstructed, data_dict["act"]
                )
        if self.crafter_transition_replay_enabled and self.crafter_transition_replay_include_changed_slot and (
            "_replay_changed_slot_signature" not in data_dict
        ):
            data_dict = dict(data_dict)
            signatures = self.crafter_changed_slot_signatures(data_dict)
            data_dict["_replay_changed_slot_signature"] = signatures
            types = np.asarray(data_dict["_replay_transition_type"])
            data_dict["_replay_bucket"] = self._crafter_buckets_with_signatures(
                types, data_dict["act"], signatures
            )

        optional_keys = [
            key for key in data_dict
            if key not in required and data_dict[key] is not None
        ]
        for i in range(length):
            sample = {key: data_dict[key][i] for key in required}
            for key in optional_keys:
                if len(data_dict[key]) != length:
                    raise ValueError(f"Replay field '{key}' has inconsistent length")
                sample[key] = data_dict[key][i]
            self.buffer.append(sample)

    def save_to_file(self, path: str):
        data = self.export_dict()
        target = os.path.abspath(os.path.expanduser(str(path)))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.", suffix=".tmp",
            dir=os.path.dirname(target),
        )
        os.close(fd)
        try:
            torch.save(data, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"Fisher buffer saved to: {path}")

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No buffer file found at {path}")
        data = torch.load(path, weights_only=False)
        self.load_from_dict(data)
        print(f"Fisher buffer loaded from: {path}")

    def __len__(self):
        return len(self.buffer)
