import os
import csv
import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Union
from modelBased.common import utils
from domain.minigrid import minigrid_support as minigrid_utils
from domain.crafter.crafter_support import (
    crafter_classification_loss,
    crafter_reconstruct_from_logits,
    visualize_crafter_wm,
)
from . import AttentionWM_support
from . import Embedding_support
from . import MLP_support
import pandas as pd
import numpy as np


class AttentionWorldModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.mask_size = hparams.attention_mask_size
        self.channel, self.row, self.col = hparams.grid_shape
        self.lr= hparams.lr
        self.weight_decay = hparams.wd
        self.visualizationFlag = hparams.visualization
        self.visualize_every = hparams.visualize_every
        self.step_counter = 0  
        self.data_type = hparams.data_type
        self.ewc_ratio = getattr(hparams, "ewc_ratio", 0.2)           # Target ratio: EWC ~= 20% of obs_loss; set null in yaml to disable manual control.
        self.lambda_ema = getattr(hparams, "lambda_ema", 0.1)         # EMA smoothing factor for lambda.
        self.lambda_ewc_min = getattr(hparams, "lambda_ewc_min", 1e-4)
        self.lambda_ewc_max = getattr(hparams, "lambda_ewc_max", 1e3)
        self.lambda_ewc = float(getattr(hparams, "lambda_ewc", 1.0))
        self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
        self.loss_map_result = None
        self.inventory_loss_vector_result = None
        self.inventory_loss_accumulator = []
        self._val_step_outputs = []



        # Slow outer-loop drift controller.
        self.warmup_steps = getattr(hparams, "warmup_steps", 100)
        self.drift_cooldown = getattr(hparams, "drift_cooldown", 200)  # Minimum steps between adjustments.
        self._last_drift_update_step = -10**9
        self.drift_threshold = getattr(hparams, "drift_threshold", 1e-3)
        self.fisher = 0
        self.old_params = None
        self.env_type = hparams.env_type
        self.frame_stack = getattr(hparams, "frame_stack", 1)
        self.use_bipedal_flag = bool(getattr(hparams, "use_bipedal_attention", False))
        # Keep a single explicit flag for the vector-state BipedalWalker path.
        # Some call sites check `self.is_bipedal`, so define it before first use.
        self.is_bipedal = (self.env_type == "bipedalwalker") or self.use_bipedal_flag
        print(f"[AttentionWM Init] env_type: {self.env_type} | data_type: {self.data_type} | val_metric: {getattr(self.hparams, 'validation_metric', 'mse')}")
        
        MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
        # Initialize the prediction model.
        module_class = MODEL_MAPPING.get(hparams.model_type.lower())
        if module_class is not None:
            self.model = module_class(
                hparams.data_type, 
                hparams.grid_shape, 
                hparams.attention_mask_size, 
                hparams.embed_dim, 
                hparams.num_heads,
                env_type="bipedalwalker" if self.is_bipedal else self.env_type,
                frame_stack=self.frame_stack
            )
        else:
            print(f"Model type: {hparams.model_type} not supported")
            exit()
        

        if hparams.freeze_weight:
            utils.load_model_weight(self.model, hparams.model_save_path)
        self.loss = nn.MSELoss() # nn.SmoothL1Loss()
        self.visual_func = minigrid_utils.Visualization(hparams)
        self.train_token_loss_accumulator = {}
        self.train_token_acc_accumulator = {}
        self.train_token_loss_csv_path = None
        self.bipedal_token_loss_weights = dict(getattr(hparams, "bipedal_token_loss_weights", {}))
        if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
            self.train_token_loss_csv_path = os.path.join(
                os.path.dirname(hparams.model_save_path),
                "bipedal_train_token_losses.csv",
            )
        self.save_hyperparameters(hparams)


    def save_old_params(self):
        """Save current model parameters for EWC, moved to model's device."""
        device = next(self.parameters()).device
        old_params = {k: v.clone().detach().cpu() 
              for k, v in self.state_dict().items()}
        return old_params

    def get_bipedal_token_weight(self, token_name: str) -> float:
        return float(self.bipedal_token_loss_weights.get(token_name, 1.0))
    
    # def load_old_params(self, old_params):
    #     """Load previous-task parameters into the current model."""
    #     for n, p in self.named_parameters():
    #         if n in old_params:
    #             p.data.copy_(old_params[n])  # Overwrite the current parameter with the saved value.

    def compute_avg_param_drift(self) -> float:
        drift_sum = 0.0
        num = 0
        for name, param in self.named_parameters():
            if self.old_params is not None and name in self.old_params:
                drift = (param - self.old_params[name].to(param.device)).pow(2).mean()
                drift_sum += drift.item()
                num += 1
        avg_drift = drift_sum / max(num, 1)
        return avg_drift

    def update_lambda_ewc_by_ratio(self, obs_loss: torch.Tensor, ewc_raw: torch.Tensor, r: float | None):
        """
        [DEPRECATED / FIXED]
        Originally attempted to dynamically scale lambda based on loss ratio.
        Removed to ensure stability. Now uses fixed self.lambda_ewc.
        """
        pass 
        # Keep lambda updates independent from inverse-drift heuristics.
        # This caused exploding updates or vanishing constraints.
        # We rely on fixed lambda_ewc set in config.

    def update_lambda_ewc(self, avg_drift: float):
        """
        Slow outer-loop controller for `lambda` based on average parameter drift.
        It complements the fast ratio controller by:
        - learning `drift_threshold` during warmup when it is unset
        - using a cooldown window to reduce interference with fast updates
        - applying hysteresis so adjustments happen only outside a safe band
        - updating in log space, then smoothing with EMA and clamping to bounds
        """
        import math
        import torch

        # Protective defaults.
        if not hasattr(self, "lambda_ewc"):
            self.lambda_ewc = float(getattr(self.hparams, "lambda_ewc", 1.0))
        if not hasattr(self, "_drift_values"):
            self._drift_values = []
        if not hasattr(self, "_last_drift_update_step"):
            self._last_drift_update_step = -10**9

        # Read hyperparameters, allowing overrides from `hparams` or instance attributes.
        warmup_steps = getattr(self.hparams, "warmup_steps", getattr(self, "warmup_steps", 100))
        cooldown = getattr(self.hparams, "drift_cooldown", getattr(self, "drift_cooldown", 200))
        hi_ratio = getattr(self.hparams, "drift_hi", 1.10)  # Increase only above 110% of the threshold.
        lo_ratio = getattr(self.hparams, "drift_lo", 0.70)  # Decrease only below 70% of the threshold.
        up_step = getattr(self.hparams, "drift_up_step", 0.10)  # Log-space step size for increasing lambda.
        down_step = getattr(self.hparams, "drift_down_step", 0.05)  # Log-space step size for decreasing lambda.
        lam_min = getattr(self.hparams, "lambda_ewc_min", getattr(self, "lambda_ewc_min", 1e-4))
        lam_max = getattr(self.hparams, "lambda_ewc_max", getattr(self, "lambda_ewc_max", 1e3))
        lam_ema = getattr(self.hparams, "lambda_ema", getattr(self, "lambda_ema", 0.1))

        # Log drift statistics.
        self.log("train/avg_param_drift", avg_drift)

        # During warmup, infer the threshold from the running drift mean if needed.
        if getattr(self, "drift_threshold", None) is None:
            self._drift_values.append(float(avg_drift))
            if len(self._drift_values) >= warmup_steps:
                self.drift_threshold = float(sum(self._drift_values) / len(self._drift_values))
                print(f"[Auto-tuned] drift_threshold set to {self.drift_threshold:.6f}")
            # Do not adjust lambda during warmup.
            return

        # Cooldown prevents frequent interference with the fast controller.
        if self.global_step - self._last_drift_update_step < cooldown:
            return

        # Only adjust outside the hysteresis band.
        hi = self.drift_threshold * hi_ratio
        lo = self.drift_threshold * lo_ratio

        lam = float(max(self.lambda_ewc, lam_min))
        lam_log = math.log(lam)
        changed = False

        if avg_drift > hi:
            lam_log += up_step
            changed = True
        elif avg_drift < lo:
            lam_log -= down_step
            changed = True

        if changed:
            lam_new = math.exp(lam_log)
            # Clamp to configured bounds.
            lam_new = float(torch.clamp(torch.tensor(lam_new), lam_min, lam_max))
            # Apply EMA smoothing.
            self.lambda_ewc = (1.0 - lam_ema) * self.lambda_ewc + lam_ema * lam_new
            # Update the last-adjustment step.
            self._last_drift_update_step = self.global_step

    def load_old_params(self, old_params):
        device = next(self.parameters()).device
        self.old_params = {k: v.to(device) for k, v in old_params.items()}

    # def compute_fisher(self, dataloader, samples, scale_factor):
    #     fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
    #     self.eval()
    #     device = next(self.parameters()).device
    #     count = 0
    #     for i, batch in enumerate(dataloader):
    #         if count >= samples:
    #             break
    #         self.zero_grad()

    #         obs, act, obs_next, info, obs_masked = self.preprocess_batch(batch)
    #         obs = obs.to(device).float()
    #         act = act.to(device)
    #         obs_next = obs_next.to(device).float()
    #         obs_pred, _ = self(obs, act, info)
    #         loss = scale_factor * self.loss_function_weight(obs_pred, obs_next, obs_masked)['loss_obs']
    #         loss.backward(retain_graph=False)
    #         for n, p in self.named_parameters():
    #             if p.grad is not None:
    #                 fisher[n] += p.grad.detach().pow(2)
    #         count += 1
    #     for n in fisher:
    #         fisher[n] /= count
    #     # for k in fisher:
    #     #     fisher[k] = torch.sqrt(fisher[k] + 1e-8)
    #     #     fisher[k] *= 5
    #     # Fisher normalization by mean
    #     for k in fisher:
    #         mean_val = fisher[k].mean()
    #         fisher[k] = scale_factor * fisher[k] / (mean_val + 1e-8)

    #     all_f = torch.cat([f.flatten() for f in fisher.values()])
    #     print(f"[Fisher] mean={all_f.mean():.3e}, max={all_f.max():.3e}, min={all_f.min():.3e}")
    #     return fisher

    def compute_fisher(self, dataloader, samples, scale_factor):
        """
        Correct diagonal Fisher Information Matrix (FIM) estimation.
        Formula: F = E[ (\nabla \log p(x))^2 ]
        
        Implementation:
        1. Iterate through the dataloader.
        2. For each batch, iterate through INDIVIDUAL samples.
        3. Compute gradient for single sample -> square it -> accumulate.
        4. Average over total samples.
        """
        import torch
        self.eval() 
        device = next(self.parameters()).device

        fisher = {n: torch.zeros_like(p, dtype=torch.float32, device=device)
                for n, p in self.named_parameters() if p.requires_grad}

        count = 0
        total_samples_target = int(samples)
        
        print(f"[Fisher] Starting computation for ~{total_samples_target} samples...")

        for i, batch in enumerate(dataloader):
            if count >= total_samples_target:
                break
            
            # 1. Preprocess batch
            obs, act, obs_next, info, obs_masked, _, inv, inv_next = self.preprocess_batch(batch)
            # 2. Iterate over samples in the batch
            batch_size = obs.shape[0]
            for b in range(batch_size):
                if count >= total_samples_target:
                    break
                    
                self.zero_grad(set_to_none=True)

                # Extract single sample (keep dim 0 for model compatibility)
                s_obs = obs[b:b+1].to(device, dtype=torch.float32)
                s_act = act[b:b+1].to(device)
                s_info = {k: v[b:b+1] for k, v in info.items()} if info is not None else None
                s_obs_next = obs_next[b:b+1].to(device, dtype=torch.float32)
                s_obs_masked = obs_masked[b:b+1].to(device) if obs_masked is not None else None
                s_inv = inv[b:b+1].to(device) if inv is not None else None
                s_inv_next = inv_next[b:b+1].to(device) if inv_next is not None else None

                # Forward & Backward (fp32)
                with torch.amp.autocast("cuda", enabled=False):
                    pred, _, s_inv_pred = self(s_obs, s_act, s_info, inv=s_inv)
                    loss_dict = self.loss_function_weight(pred, s_obs_next, s_obs_masked, obs_prev=s_obs)
                    loss_sample = loss_dict['loss_obs']

                    if self.is_bipedal and isinstance(s_inv_pred, dict) and "contact_logits" in s_inv_pred:
                        contact_bce_total = torch.zeros((), device=device, dtype=torch.float32)
                        token_spec_map = dict(getattr(self.model, "bipedal_token_specs", []))
                        for token_name in getattr(self.model, "contact_token_names", set()):
                            token_indices = token_spec_map[token_name]
                            next_contact_target = (s_obs[:, token_indices] + s_obs_next[:, token_indices]).clamp(0.0, 1.0)
                            contact_logits = s_inv_pred["contact_logits"][token_name]
                            token_bce = F.binary_cross_entropy_with_logits(
                                contact_logits,
                                next_contact_target,
                            )
                            contact_bce_total = contact_bce_total + self.get_bipedal_token_weight(token_name) * token_bce
                        loss_sample = loss_sample + contact_bce_total
                    
                    if self.env_type == 'crafter' and s_inv_pred is not None and s_inv_next is not None:
                        if s_inv is not None:
                            inv_diff = torch.abs(s_inv_next - s_inv).float()
                            # If an inventory element changes, hit it with x100 magnitude
                            inv_w = 1.0 + (inv_diff > 1e-5).float() * 100.0
                            loss_inv = (F.mse_loss(s_inv_pred, s_inv_next.float(), reduction='none') * inv_w).mean()
                        else:
                            loss_inv = F.mse_loss(s_inv_pred, s_inv_next.float())
                        loss_sample = loss_sample + 10.0 * loss_inv
                
                loss_sample.backward()

                # Accumulate squares
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        # Square the gradient of this SINGLE sample
                        g2 = p.grad.detach().float().pow(2)
                        fisher[n] += g2
                
                count += 1
        
        if count == 0:
             print("[Fisher] Warning: No samples processed!")
             return fisher

        # 3. Normalize by N (Average)
        for n in fisher:
            fisher[n] /= float(count)

        # 4. Standardize Fisher (Normalize to Mean=1.0) to make lambda_ewc scale-invariant
        with torch.no_grad():
            all_vals = torch.cat([f.flatten() for f in fisher.values()])
            mean_val = all_vals.mean()
            max_val = all_vals.max()
            
            print(f"[Fisher] Computed with {count} samples. Raw Mean={mean_val:.3e}, Max={max_val:.3e}")
            
            if mean_val > 1e-20:
                scale = 1.0 / mean_val
                # Apply normalization FIRST
                for n in fisher:
                    fisher[n] *= scale
                print(f"[Fisher] Normalized to Mean=1.0. Applied scale: {scale:.3e}")
                
                # THEN apply user scale_factor if needed (usually 1.0 now)
                if scale_factor != 1.0:
                    for n in fisher:
                        fisher[n] *= scale_factor
                    print(f"[Fisher] Applied extra config scale_factor: {scale_factor}")
            else:
                print("[Fisher] Warning: Fisher values are essentially zero. Check gradients!")

        # Move to CPU for storage
        fisher = {k: v.detach().cpu() for k, v in fisher.items()}
        return fisher


    

    # def ewc_loss(self, lambda_ewc):
    #     if self.fisher is None or self.old_params is None:
    #         return torch.tensor(0.0, device=next(self.parameters()).device)
        
    #     device = next(self.parameters()).device
    #     loss = torch.tensor(0.0, device=device)  
    #     for n, p in self.named_parameters():
    #         if n in self.fisher and n in self.old_params:
    #             fisher = self.fisher[n].to(device)
    #             p_old = self.old_params[n].to(device)
    #             loss += (fisher * (p - p_old).pow(2)).sum()

    #     return lambda_ewc * loss

    def set_consolidation(self, old_params: dict, fisher: dict, load_weights: bool = True):
        """
        Register the EWC anchor state (old parameters + Fisher matrix) and
        optionally load the old parameters into the current model.

        Args:
            old_params (dict): Parameters saved from the previous phase.
            fisher (dict): Fisher information matrix.
            load_weights (bool): Whether to load the old weights into the model.
        """
        # ----------------------------------------------------------
        # (1) Old-parameter anchor
        # ----------------------------------------------------------
        if old_params is not None:
            # Store the old parameters on CPU in float32.
            self.old_params = {k: v.detach().cpu().float() for k, v in old_params.items()}

            if load_weights:
                # [FIX] Clear all hooks that might cause ReferenceError in state_dict() or weight loading
                if hasattr(self, "_state_dict_hooks"):
                    self._state_dict_hooks.clear()
                if hasattr(self, "_parameters"):
                    for p_name, p in self._parameters.items():
                        if p is not None and hasattr(p, "_hooks"):
                            p._hooks.clear()

                # Directly load without the complex state_dict() dance to avoid hooks
                # strict=False allows missing or mismatched keys without crashing
                self.load_state_dict(old_params, strict=False)
                
                print(f"[EWC] Attempted to load weights from previous task (strict=False).")
            else:
                print("[EWC] old_params received but model weights not loaded (load_weights=False).")
        else:
            self.old_params = None
            print("[EWC] No old_params provided — starting from scratch.")

        # ----------------------------------------------------------
        # (2) Fisher information matrix
        # ----------------------------------------------------------
        if fisher is not None:
            self.fisher = {k: v.detach().cpu().float() for k, v in fisher.items()}
            print(f"[EWC] Fisher matrix loaded with {len(self.fisher)} entries.")
        else:
            self.fisher = None
            print("[EWC] No Fisher matrix provided — no EWC regularization will be applied.")


    def ewc_loss(self):
        """
        Return the raw EWC value before multiplying by `lambda_ewc`, computed
        in fp32 for stability.

        Stabilization details:
        1. Explicitly move tensors to device and cast to float32.
        2. Normalize by model scale so the value is less sensitive to size.
        """
        device = next(self.parameters()).device
        if self.fisher is None or self.old_params is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        total = torch.zeros((), device=device, dtype=torch.float32)
        count = 0

        # Disable autocast so the EWC term is always evaluated in fp32.
        with torch.amp.autocast("cuda", enabled=False):
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if n not in self.fisher or n not in self.old_params:
                    continue

                f = self.fisher[n].to(device=device, dtype=torch.float32)
                d = (p.float() - self.old_params[n].to(device).float())
                total = total + (f * d.pow(2)).sum()
                count += d.numel()

            if count > 0:
                # Use a moderate normalization factor: dividing by all parameters
                # makes the term too small, while not normalizing at all makes it
                # too large. Averaging over trainable layers keeps the scale usable.
                num_layers = len([n for n, p in self.named_parameters() if p.requires_grad])
                total = total / (num_layers * 2.0)

        return total  # Intentionally return the raw term without multiplying by lambda.
    
    def accumulate_loss(self, loss_map, agent_pos):
        """
        loss_map: (mask_size, mask_size) local loss values
        agent_pos: (y, x) agent position on the full map
        """
        ay, ax = agent_pos
        half = self.mask_size // 2

        for dy in range(self.mask_size):
            for dx in range(self.mask_size):
                global_y = ay + (dy - half)
                global_x = ax + (dx - half)

                # Bounds check to avoid indexing outside the full map.
                if 0 <= global_y < self.row and 0 <= global_x < self.col:
                    value = loss_map[dy, dx].item()
                    self.loss_accumulator[global_y][global_x].append(value)

    def compute_cell_loss(self, next_pred, next_true):
        # Compute the per-cell error map.
        if self.env_type == 'crafter':
            # Classification loss per cell (with target clamping)
            loss_map = crafter_classification_loss(
                next_pred, next_true, reduction='none'
            )  # (B, H, W)
        else:
            # Standard Regression error
            error = torch.abs(next_pred - next_true)
            loss_map = error.mean(dim=1)  # (B, H, W)

        return loss_map



    def encode(self, state, inv=None):
        """
        Extract latent feature representations from observations.
        Used by P2E Ensemble to compute epistemic uncertainty (disagreement).
        
        Args:
            state: (B, C, H, W) raw observation tensor
            inv: optional inventory tensor for Crafter
        Returns:
            feat: (B, N, embed_dim) spatial token features after conv+positional encoding
        """
        if getattr(self, "is_bipedal", False) and hasattr(self.model, "tokenize_bipedal_state"):
            with torch.no_grad() if not self.training else torch.enable_grad():
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                state = state.float()
                x = self.model.tokenize_bipedal_state(state)
            return x

        if not hasattr(self.model, 'conv1'):
            # Fallback: flatten the obs as a simple feature
            B = state.shape[0]
            return state.view(B, -1, 1).float()

        with torch.no_grad() if not self.training else torch.enable_grad():
            B = state.shape[0]
            K = getattr(self.model, 'frame_stack', 1)
            TotalC = state.shape[1]
            C_base = TotalC // K
            H, W = state.shape[2], state.shape[3]

            import torch.nn.functional as F_local
            if self.model.data_type == 'discrete':
                all_frames_emb = []
                for k in range(K):
                    frame = state[:, k*C_base:(k+1)*C_base]
                    if self.env_type == 'crafter':
                        obj = frame[:, 0]
                        dir_id = frame[:, 1]
                        obj_oh = F_local.one_hot(obj.reshape(B, -1).long(), num_classes=20).float()
                        dir_oh = F_local.one_hot(dir_id.reshape(B, -1).long(), num_classes=5).float()
                        frame_emb = torch.cat([obj_oh, dir_oh], dim=-1)
                    else:
                        obj = frame[:, 0]
                        color = frame[:, 1]
                        dir_id = frame[:, 2]
                        obj_oh = F_local.one_hot(obj.reshape(B, -1).long(), num_classes=11).float()
                        color_oh = F_local.one_hot(color.reshape(B, -1).long(), num_classes=6).float()
                        dir_oh = F_local.one_hot(dir_id.reshape(B, -1).long(), num_classes=4).float()
                        frame_emb = torch.cat([obj_oh, color_oh, dir_oh], dim=-1)
                    all_frames_emb.append(frame_emb)
                state_emb = torch.cat(all_frames_emb, dim=-1)
                state_emb = state_emb.transpose(1, 2).reshape(B, self.model.input_channel, H, W)
            else:
                state_emb = state.float()

            # Conv embedding
            x = self.model.relu(self.model.bn1(self.model.conv1(state_emb)))
            x = self.model.relu(self.model.bn2(self.model.conv2(x)))
            x = self.model.flatten(x).transpose(1, 2)  # (B, N, D)
            x = x + self.model.pos_embedding               # add position encoding
        return x  # (B, N, embed_dim)

    def forward(self, state, action, info, inv=None):
        out = self.model(state, action, info, inv=inv)
        if len(out) == 3:
            next_state_pred, attentionWeight, aux_pred = out
            return next_state_pred, attentionWeight, aux_pred
        else:
            # Fallback for MLP or older models that only return 2 items
            next_state_pred, attentionWeight = out
            return next_state_pred, attentionWeight, None


    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict.flatten(1), next_observations_true.flatten(1))
        loss = {'loss_obs':loss_obs}
        return loss
    

    def loss_function_weight(
        self,
        next_observations_predict,
        next_observations_true,
        obs_masked=None,
        obs_prev=None,
        force_weighted: bool = False,
    ):
        """
        Custom Loss with tiered aggressive weighting:
        - If use_weighted_loss is True:
          - Base=1.0, Move=+10, Interaction=+50 (MiniGrid)
          - Tiered CE (Crafter)
        - Else: 1.0 (Standard CE/MSE)
        """
        device = next_observations_predict.device 
        use_weighted_loss = force_weighted or getattr(self.hparams, "use_weighted_loss", False)
        if self.is_bipedal:
            weighted_losses = []
            for token_name, token_indices in getattr(self.model, "bipedal_token_specs", []):
                if token_name in getattr(self.model, "contact_token_names", set()):
                    continue
                token_loss = F.mse_loss(
                    next_observations_predict[:, token_indices],
                    next_observations_true[:, token_indices],
                )
                weighted_losses.append(self.get_bipedal_token_weight(token_name) * token_loss)
            loss = torch.stack(weighted_losses).sum() if weighted_losses else F.mse_loss(
                next_observations_predict, next_observations_true
            )
            return {"loss_obs": loss}
        
        # 1. Base Loss (MSE for MiniGrid, CrossEntropy for Crafter)
        if self.env_type == 'crafter':
            # Classification loss from crafter_support (with optional tiered weighting)
            raw_error_map = crafter_classification_loss(
                next_observations_predict, next_observations_true, reduction='none', weighted=use_weighted_loss
            )  # (B, H, W)
            weights = 1.0 # Tiered weights are already inside crafter_classification_loss
        else:
            # MiniGrid: Standard Regression
            raw_sq_error = (next_observations_predict - next_observations_true) ** 2
            raw_error_map = raw_sq_error.mean(dim=1)  # (B, H, W)
            weights = 1.0

        # 2. Aggressive Spatial/Change Weighting (Only if use_weighted_loss is True and NOT handled by CE)
        if use_weighted_loss and self.env_type != 'crafter':
            if obs_prev is not None:
                 if obs_prev.dtype != next_observations_true.dtype:
                     obs_prev = obs_prev.float() 
                 
                 C_target = next_observations_true.size(1)
                 C_prev = obs_prev.size(1)
                 obs_prev_latest = obs_prev[:, -C_target:] if C_prev > C_target else obs_prev
                 
                 # For MiniGrid, next_observations_true is already the DELTA, so diff is just its absolute value.
                 diff = torch.abs(next_observations_true).sum(dim=1, keepdim=True)
                 change_mask = (diff > 1e-5).float()
                 
                 if C_target > 2:
                      state_change_mask = (torch.abs(next_observations_true[:, 2:3, :, :]) > 1e-5).float()
                 else:
                      state_change_mask = torch.zeros_like(change_mask)
            else:
                 change_mask = (next_observations_true.abs() > 1e-6).any(dim=1, keepdim=True).float()
                 state_change_mask = torch.zeros_like(change_mask)

            weights = 1.0 + (change_mask * 3.0) + (state_change_mask * 10.0)
            
            if obs_masked is not None:
                 if obs_masked.ndim == 3:
                     static_mask = obs_masked.unsqueeze(1).float()
                 else:
                     static_mask = obs_masked.float()
                 interaction_mask = change_mask * static_mask
                 weights = weights + (interaction_mask * 20.0)

        # 5. Weighted reduction
        if torch.is_tensor(weights) and weights.ndim > raw_error_map.ndim:
             weights = weights.squeeze(1) # [B, 1, H, W] -> [B, H, W]

        if torch.is_tensor(weights):
            if bool(getattr(self.hparams, "normalize_weighted_loss", True)):
                weighted_error = raw_error_map * weights
                loss = weighted_error.sum() / weights.sum().clamp_min(1.0)
            else:
                loss = (raw_error_map * weights).mean()
        else:
            loss = (raw_error_map * weights).mean()
        
        return {"loss_obs": loss}



    def configure_optimizers(self):
        # Use a higher learning rate for the classification head so it adapts
        # faster to new environment tiles, while keeping the base model stable.
        head_params = []
        base_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # `fc` is the output head that predicts tile classes.
            if 'model.fc.' in name:
                head_params.append(param)
            else:
                base_params.append(param)

        optimizer = optim.Adam([
            {'params': base_params, 'lr': self.lr},
            {'params': head_params, 'lr': self.lr * 2.0}  # Double the learning rate for the head.
        ], betas=(0.9, 0.999), eps=1e-6, weight_decay=self.weight_decay)

        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'avg_val_loss_wm',
                "frequency": 1
            },
        }

    def preprocess_batch(self, batch, training=False):
        '''
        Preprocess the batch data: extract masked observations and object positions.
        batch['obs']: (B, C, H, W)
        '''
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']
        if self.env_type == 'with_obj':
            info = batch['info']
        else:
            info = None
        
        inv = batch.get('inv', None)
        inv_next = batch.get('inv_next', None)

        if self.is_bipedal:
            self.step_counter += 1
            return obs.float(), act.float(), obs_next.float(), None, None, None, None, None

        player_id = 13 if self.env_type == 'crafter' else 10
        agent_postion_yx_batch = minigrid_utils.get_agent_position(obs, player_id=player_id)
        obs_masked = minigrid_utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
        obs_next_masked = minigrid_utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch)

        # extract positions where objects are located (use the most recent frame if stacked)
        C_base = 2 if self.env_type == 'crafter' else 3
        curr_obj_idx = (self.frame_stack - 1) * C_base
        object_map = obs_masked[:, curr_obj_idx]  # Object channel from the most recent frame: (B, H, W).
        if self.env_type == 'crafter':
            # Interactive elements in Crafter: Cow(14), Zombie(15), Skeleton(16), Arrow(17), Plant(18)
            # Exclusion: Player(13) must be predicted precisely, Table(11)/Furnace(12) are static.
            elements_mask = (object_map >= 14) & (object_map <= 18)
        else:
            key_mask = (object_map == 5)
            door_mask = (object_map == 4)
            lava_mask = (object_map == 9)
            elements_mask = key_mask | door_mask | lava_mask  # (B,H,W)
        
        ## visualization is now moved to training_step/validation_step for logits access
        self.step_counter += 1
        return obs_masked, act, obs_next_masked, info, elements_mask, agent_postion_yx_batch, inv, inv_next


    def training_step(self, batch, batch_idx):
        # Forward pass and primary loss.
        obs, act, obs_next, info, elements_mask, agent_pos, inv, inv_next = self.preprocess_batch(batch, True)
        obs_pred, attentionWeight, aux_pred = self(obs, act, info, inv=inv)

        # Restrict Crafter visualization to the final epoch to reduce overhead.
        is_last_epoch = False
        try:
            is_last_epoch = (self.current_epoch == self.trainer.max_epochs - 1)
        except:
            pass
            
        if self.visualizationFlag and (not self.is_bipedal) and is_last_epoch and (self.step_counter % self.visualize_every == 0):
            if self.env_type == 'crafter':
                visualize_crafter_wm(obs, obs_next, obs_pred, int(act[0].item()), self.step_counter, 
                                     save_dir=os.path.join("modelBased/log", "wm_visual/train"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_pos[0],
                                     inv=inv[0].cpu().numpy() if inv is not None else None,
                                     inv_next=inv_next[0].cpu().numpy() if inv_next is not None else None)
            else:
                # Fallback to legacy visualize_data for MiniGrid (requires whole map)
                # Note: this part needs whole map, which we have in 'batch'
                self.visual_func.visualize_data(batch['obs'], batch['obs'] + batch['obs_next'], act, obs, obs_next, info, self.step_counter, agent_pos)

        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        
        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # [TRAINING] Aggressive Weighted Loss for Optimization
        loss_dict = self.loss_function_weight(obs_pred, obs_next, elements_mask, obs_prev=obs)
        weighted_obs_loss = loss_dict['loss_obs']
        
        # [LOGGING] Metrics for Scientific Reporting
        if self.env_type == 'crafter':
            obs_pred_reconst = crafter_reconstruct_from_logits(obs_pred)
            raw_mse = F.mse_loss(obs_pred_reconst, obs_next)
            # Raw (unweighted) CE loss — the true training signal for Crafter
            from domain.crafter.crafter_support import crafter_classification_loss
            raw_ce = crafter_classification_loss(obs_pred, obs_next, reduction='mean')
        else:
            if self.is_bipedal and hasattr(self.model, "contact_indices"):
                contact_indices = self.model.contact_indices
                continuous_mask = torch.ones(obs_pred.size(1), dtype=torch.bool, device=obs_pred.device)
                continuous_mask[contact_indices] = False
                raw_mse = F.mse_loss(obs_pred[:, continuous_mask], obs_next[:, continuous_mask])
            else:
                raw_mse = F.mse_loss(obs_pred, obs_next)
            raw_ce = None
            if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
                for token_name, token_indices in self.model.bipedal_token_specs:
                    if token_name in getattr(self.model, "contact_token_names", set()):
                        contact_logits = aux_pred["contact_logits"][token_name]
                        next_contact_target = (obs[:, token_indices] + obs_next[:, token_indices]).clamp(0.0, 1.0)
                        token_loss = F.binary_cross_entropy_with_logits(contact_logits, next_contact_target)
                        token_pred = (torch.sigmoid(contact_logits) >= 0.5).float()
                        token_acc = (token_pred == next_contact_target).float().mean()
                        log_name = f"train/token_{token_name}_bce"
                        self.train_token_acc_accumulator[token_name].append(float(token_acc.detach().cpu()))
                    else:
                        token_loss = F.mse_loss(obs_pred[:, token_indices], obs_next[:, token_indices])
                        log_name = f"train/token_{token_name}_mse"
                    self.train_token_loss_accumulator[token_name].append(float(token_loss.detach().cpu()))
        
        # Raw EWC term before multiplying by lambda.
        ewc_raw = self.ewc_loss()

        # Fast controller: balance EWC against the weighted optimization loss.
        self.update_lambda_ewc_by_ratio(weighted_obs_loss, ewc_raw, self.ewc_ratio)

        # Compose the total loss.
        ewc_term = self.lambda_ewc * ewc_raw
        loss_total = weighted_obs_loss + ewc_term
        if self.is_bipedal and aux_pred is not None and "contact_logits" in aux_pred:
            contact_bce_total = torch.zeros((), device=obs.device, dtype=obs.dtype)
            total_contact_correct = torch.zeros((), device=obs.device, dtype=obs.dtype)
            total_contact_count = torch.zeros((), device=obs.device, dtype=obs.dtype)
            for token_name in getattr(self.model, "contact_token_names", set()):
                token_indices = dict(self.model.bipedal_token_specs)[token_name]
                next_contact_target = (obs[:, token_indices] + obs_next[:, token_indices]).clamp(0.0, 1.0)
                contact_logits = aux_pred["contact_logits"][token_name]
                token_bce = F.binary_cross_entropy_with_logits(
                    contact_logits,
                    next_contact_target,
                )
                token_pred = (torch.sigmoid(contact_logits) >= 0.5).float()
                total_contact_correct = total_contact_correct + (token_pred == next_contact_target).float().sum()
                total_contact_count = total_contact_count + torch.tensor(
                    next_contact_target.numel(), device=obs.device, dtype=obs.dtype
                )
                contact_bce_total = contact_bce_total + self.get_bipedal_token_weight(token_name) * token_bce
            loss_total = loss_total + contact_bce_total
            contact_acc = total_contact_correct / total_contact_count.clamp_min(1.0)
            self.log("train/contact_bce", contact_bce_total.detach(), on_step=True, on_epoch=True)
            self.log("train/contact_acc", contact_acc.detach(), on_step=True, on_epoch=True)
        
        # Add inventory MSE loss dynamically if applicable
        if self.env_type == 'crafter' and aux_pred is not None and inv_next is not None:
            inv_pred = aux_pred
            if inv is not None:
                inv_diff = torch.abs(inv_next - inv).float()
                inv_weights = 1.0 + (inv_diff > 1e-5).float() * 20.0
                inv_loss = (F.mse_loss(inv_pred, inv_next.float(), reduction='none') * inv_weights).mean()
            else:
                inv_loss = F.mse_loss(inv_pred, inv_next.float())
            # Keep the inventory term amplified so it stays competitive with map prediction.
            loss_total = loss_total + 5.0 * inv_loss
            self.log("train/inv_loss", inv_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Unified logging.
        # Log raw_mse as 'loss_obs' for compatibility
        self.log("loss_obs", raw_mse, prog_bar=True, on_step=True, on_epoch=True)
        
        if raw_ce is not None:
            # Crafter: also log raw CE separately for clear monitoring
            self.log("train/ce_loss", raw_ce, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log weighted loss separately for debugging
        self.log("train/loss_weighted", weighted_obs_loss, on_step=True, on_epoch=True)
        
        self.log("train/ewc_raw", ewc_raw.detach(), on_step=True, on_epoch=True)
        self.log("train/lambda_ewc", torch.tensor(self.lambda_ewc, device=obs.device),
                on_step=True, on_epoch=True)
        self.log("train/ewc_term", ewc_term.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_total", loss_total.detach(), on_step=True, on_epoch=True)

        return loss_total

    def on_train_epoch_start(self):
        if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
            self.train_token_loss_accumulator = {
                token_name: [] for token_name, _ in self.model.bipedal_token_specs
            }
            self.train_token_acc_accumulator = {
                token_name: [] for token_name, _ in self.model.bipedal_token_specs
                if token_name in getattr(self.model, "contact_token_names", set())
            }

    def on_train_epoch_end(self):
        if not (self.is_bipedal and hasattr(self.model, "bipedal_token_specs")):
            return
        if self.train_token_loss_csv_path is None:
            return

        os.makedirs(os.path.dirname(self.train_token_loss_csv_path), exist_ok=True)
        token_names = [token_name for token_name, _ in self.model.bipedal_token_specs]
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
        }
        for token_name in token_names:
            values = self.train_token_loss_accumulator.get(token_name, [])
            row[token_name] = float(np.mean(values)) if values else 0.0
        for token_name in getattr(self.model, "contact_token_names", set()):
            acc_values = self.train_token_acc_accumulator.get(token_name, [])
            row[f"{token_name}_acc"] = float(np.mean(acc_values)) if acc_values else 0.0

        csv_fieldnames = [
            "epoch",
            "global_step",
            *token_names,
            *[f"{token_name}_acc" for token_name in getattr(self.model, "contact_token_names", set())],
        ]
        file_exists = os.path.exists(self.train_token_loss_csv_path)
        if file_exists:
            with open(self.train_token_loss_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)
            if existing_fieldnames != csv_fieldnames:
                normalized_rows = []
                for existing_row in existing_rows:
                    normalized_rows.append({
                        field: existing_row.get(field, "")
                        for field in csv_fieldnames
                    })
                with open(self.train_token_loss_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writeheader()
                    writer.writerows(normalized_rows)

        with open(self.train_token_loss_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def validation_step(self, batch, batch_idx):
        obs, act, obs_next, info, elements_mask, agent_position, inv, inv_next = self.preprocess_batch(batch)
        obs_pred, attention_weight, inv_pred = self(obs, act, info, inv=inv)
        # if self.hparams.freeze_weight:
        #     diff = torch.abs(obs_pred - obs_next)  # (128, 3, 3, 3)
        #     max_diff_per_group, max_indices = diff.reshape(diff.shape[0], -1).max(dim=1)  
        #     mask = max_diff_per_group > 0.1
        #     indices = torch.nonzero(mask, as_tuple=True)[0]
        #     for idx in indices:
        #         flat_idx = max_indices[idx].item()
        #         pred_val = obs_pred[idx].reshape(-1)[flat_idx].item()
        #         true_val = obs_next[idx].reshape(-1)[flat_idx].item()
        #         print(f"Index {idx.item()} max diff: {max_diff_per_group[idx].item():.4f}, "
        #             f"pred={pred_val:.4f}, true={true_val:.4f}")
        
        # Map local loss back to global coordinates and store it.
        if getattr(self.hparams, "keep_cell_loss", False) and not getattr(self, 'is_bipedal', False):
            loss_map = self.compute_cell_loss(obs_pred, obs_next)
            batch_size = loss_map.shape[0]
            
            # [DEBUG] Check if we are actually getting coordinates and values
            if batch_idx == 0 and self.env_type == 'minigrid':
                print(f"[WM-Debug] LossMap Mean: {loss_map.mean():.6f}, AgentPos[0]: {agent_position[0]}")
                
            for i in range(batch_size):
                agent_pos = agent_position[i].tolist()  # (y, x)
                self.accumulate_loss(loss_map[i], agent_pos)
            if self.env_type == 'crafter' and inv_pred is not None and inv_next is not None:
                # Slot-wise inventory error signal for generator context (shape [16])
                inv_slot_mse = F.mse_loss(inv_pred, inv_next.float(), reduction='none').mean(dim=0)
                self.inventory_loss_accumulator.append(inv_slot_mse.detach().cpu())
  
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
            
        if getattr(self.hparams, "keep_cell_loss", False) and getattr(self, 'is_bipedal', False):
            if not hasattr(self, 'bipedal_semantic_acc'):
                self.bipedal_semantic_acc = []
            with torch.no_grad():
                hull_err = F.mse_loss(obs_pred[:, 0:4], obs_next[:, 0:4], reduction='none').mean(dim=1)
                leg1_err = F.mse_loss(obs_pred[:, 4:8], obs_next[:, 4:8], reduction='none').mean(dim=1)
                leg2_err = F.mse_loss(obs_pred[:, 9:13], obs_next[:, 9:13], reduction='none').mean(dim=1)
                lidar_err = F.mse_loss(obs_pred[:, 14:24], obs_next[:, 14:24], reduction='none').mean(dim=1)
                contact_err = F.mse_loss(obs_pred[:, [8, 13]], obs_next[:, [8, 13]], reduction='none').mean(dim=1)
                
                batch_semantic = torch.stack([hull_err, leg1_err, leg2_err, lidar_err, contact_err], dim=1) # [B, 5]
                self.bipedal_semantic_acc.append(batch_semantic.cpu())
        
        # Crafter validation visualization for the secondary dataset.
        if self.visualizationFlag and (not self.is_bipedal) and batch_idx == 0:
            if self.env_type == 'crafter':
                visualize_crafter_wm(obs, obs_next, obs_pred, int(act[0].item()), self.step_counter, 
                                     save_dir=os.path.join("modelBased/log", "wm_visual/val"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_position[0],
                                     inv=batch['inv'][0].cpu().numpy() if 'inv' in batch else None,
                                     inv_next=batch['inv_next'][0].cpu().numpy() if 'inv_next' in batch else None)

        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # [VALIDATION METRIC]
        # Crafter → CE is the correct metric (discrete IDs, not numeric)
        # MiniGrid → MSE as before
        if self.env_type == 'crafter':
            from domain.crafter.crafter_support import crafter_classification_loss
            val_ce = crafter_classification_loss(obs_pred, obs_next, reduction='mean')
            loss_val = val_ce
            
            if inv_pred is not None and inv_next is not None:
                val_inv_loss = F.mse_loss(inv_pred, inv_next.float())
                loss_val = loss_val + val_inv_loss
                self.log("val/inv_loss", val_inv_loss, on_step=False, on_epoch=True)

            self.log("val/ce_loss", val_ce, on_step=False, on_epoch=True)
        else:
            val_metric = str(getattr(self.hparams, "validation_metric", "mse")).lower()
            if self.is_bipedal and hasattr(self.model, "contact_indices"):
                contact_indices = self.model.contact_indices
                continuous_mask = torch.ones(obs_pred.size(1), dtype=torch.bool, device=obs_pred.device)
                continuous_mask[contact_indices] = False
                raw_mse = F.mse_loss(obs_pred[:, continuous_mask], obs_next[:, continuous_mask])
            else:
                raw_mse = F.mse_loss(obs_pred, obs_next)
            loss_val = raw_mse
            # Optional: weighted validation metric for MiniGrid comparison.
            # Keep default behavior unchanged unless validation_metric explicitly requests it.
            if (self.env_type == "minigrid") and (val_metric == "mse_weighted"):
                weighted_res = self.loss_function_weight(
                    obs_pred,
                    obs_next,
                    elements_mask,
                    obs_prev=obs,
                    force_weighted=True,
                )
                weighted_val = weighted_res["loss_obs"]
                loss_val = weighted_val
                self.log("val/mse_weighted", weighted_val, on_step=False, on_epoch=True)
                
            if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
                total_contact_bce = torch.zeros((), device=obs.device, dtype=obs.dtype)
                total_contact_correct = torch.zeros((), device=obs.device, dtype=obs.dtype)
                total_contact_count = torch.zeros((), device=obs.device, dtype=obs.dtype)
                for token_name, token_indices in self.model.bipedal_token_specs:
                    if token_name in getattr(self.model, "contact_token_names", set()):
                        contact_logits = inv_pred["contact_logits"][token_name]
                        next_contact_target = (obs[:, token_indices] + obs_next[:, token_indices]).clamp(0.0, 1.0)
                        token_loss = F.binary_cross_entropy_with_logits(contact_logits, next_contact_target)
                        token_pred = (torch.sigmoid(contact_logits) >= 0.5).float()
                        token_acc = (token_pred == next_contact_target).float().mean()
                        self.log(f"val/token_{token_name}_bce", token_loss, on_step=False, on_epoch=True)
                        self.log(f"val/token_{token_name}_acc", token_acc, on_step=False, on_epoch=True)
                        total_contact_bce = total_contact_bce + token_loss
                        total_contact_correct = total_contact_correct + (token_pred == next_contact_target).float().sum()
                        total_contact_count = total_contact_count + torch.tensor(
                            next_contact_target.numel(), device=obs.device, dtype=obs.dtype
                        )
                    else:
                        token_loss = F.mse_loss(obs_pred[:, token_indices], obs_next[:, token_indices])
                        self.log(f"val/token_{token_name}_mse", token_loss, on_step=False, on_epoch=True)
                self.log("val/contact_bce", total_contact_bce, on_step=False, on_epoch=True)
                contact_acc = total_contact_correct / total_contact_count.clamp_min(1.0)
                self.log("val/contact_acc", contact_acc, on_step=False, on_epoch=True)

        # No longer logging redundant "val_loss" here as we log "avg_val_loss_wm" at epoch end.

        self._val_step_outputs.append(loss_val.detach())

        return {
            "loss_wm_val": loss_val,             
        }

    def on_validation_epoch_start(self):
        self._val_step_outputs = []
        if getattr(self.hparams, "keep_cell_loss", False):
            self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
            self.inventory_loss_accumulator = []
            if getattr(self, "is_bipedal", False):
                self.bipedal_semantic_acc = []

    def on_validation_epoch_end(self):
        if getattr(self.hparams, "keep_cell_loss", False) and not getattr(self, 'is_bipedal', False):
            avg_loss_map = torch.zeros((self.row, self.col), device=self.device)
            for y in range(self.row):
                for x in range(self.col):
                    vals = self.loss_accumulator[y][x]
                    avg_loss_map[y, x] = sum(vals) / len(vals) if vals else 0

            self.loss_map_result = avg_loss_map.cpu().numpy()
            if self.env_type == 'minigrid':
                if self.loss_map_result.size > 0:
                    print(f"[WM-Debug] Final Heatmap Stats - Max: {self.loss_map_result.max():.6f}, Non-zero cells: {np.count_nonzero(self.loss_map_result)}")
                else:
                    print(f"[WM-Debug] Warning: Final Heatmap is empty (size 0)!")
        elif getattr(self.hparams, "keep_cell_loss", False) and getattr(self, 'is_bipedal', False):
            if hasattr(self, "bipedal_semantic_acc") and len(self.bipedal_semantic_acc) > 0:
                stacked_sem = torch.cat(self.bipedal_semantic_acc, dim=0) # [Total_Samples, 5]
                avg_sem = stacked_sem.mean(dim=0).numpy().astype(np.float32) # [5]
                self.loss_map_result = avg_sem.reshape(1, 5)
            else:
                self.loss_map_result = np.zeros((1, 5), dtype=np.float32)
            if self.env_type == 'crafter':
                if len(self.inventory_loss_accumulator) > 0:
                    stacked = torch.stack(self.inventory_loss_accumulator, dim=0)  # [num_batches, 16]
                    self.inventory_loss_vector_result = stacked.mean(dim=0).numpy().astype(np.float32)
                else:
                    self.inventory_loss_vector_result = np.zeros(16, dtype=np.float32)
            self.inventory_loss_accumulator = []
            # Clear cell-level history to avoid memory growth across repeated validations.
            self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
        
        if len(self._val_step_outputs) > 0:
            avg_loss = torch.stack(self._val_step_outputs).mean()
        else:
            avg_loss = torch.tensor(0.0, device=self.device)

        self.log("avg_val_loss_wm", avg_loss)
        self._val_step_outputs = []

    def on_save_checkpoint(self, checkpoint):
        # Example checkpoint customization: removing specific keys if needed
        t = checkpoint['state_dict']
        pass  # No specific filtering needed for a simple NN

    def calc_loss(self, trajectory_data):
        """
        Compute loss for Learning Progress (LP) reward calculation.
        Args:
            trajectory_data: dict with 'obs', 'act', 'obs_next', 'info' keys, containing tensors.
        Returns:
            loss: scalar tensor
        """
        device = self.device
        batch = {
            'obs': trajectory_data['obs'].to(device),
            'act': trajectory_data['act'].to(device),
            'obs_next': trajectory_data['obs_next'].to(device),
            'info': trajectory_data['info']
        }
        
        if self.env_type != 'with_obj':
             batch['info'] = None

        obs_masked, act, obs_next_masked, info, elements_mask, _, inv, inv_next = self.preprocess_batch(batch, training=False)
        
        obs_pred, _, _ = self(obs_masked, act, info, inv=inv)
        
        if obs_next_masked.dtype != obs_pred.dtype:
            obs_next_masked = obs_next_masked.float()
            
        loss_dict = self.loss_function_weight(obs_pred, obs_next_masked, elements_mask)
        return loss_dict

    def on_train_end(self):
        """Save a final visualization at the end of training."""
        if self.visualizationFlag and self.env_type == 'crafter':
            # We don't easily have the last batch here, but we can signal or just rely on the last training_step/val_step
            # For now, we prints a message to confirm.
            print(f"[WM] Training ended. Final visualizations saved in modelBased/log/wm_visual/")




   
