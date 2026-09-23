import os
import csv
import math
from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Union
from modelBased.common import utils
from modelBased.world_model.crafter_dynamics import (
    balanced_categorical_effect_loss,
    categorical_effect_target,
    categorical_inventory_effect_target,
    categorical_crafter_inventory_event_target,
    crafter_inventory_effect_loss,
    crafter_inventory_event_residual_loss,
    crafter_inventory_gate_loss,
    decode_crafter_inventory,
    decode_crafter_item_effects,
    decode_crafter_item_values,
    INVENTORY_VALUES,
    SURVIVAL_SLOTS,
    validate_discrete_inventory,
    crafter_pose_target,
)
from domain.minigrid import minigrid_support as minigrid_utils
from domain.crafter.crafter_support import (
    crafter_effect_loss,
    visualize_crafter_wm,
)
from domain.minigrid.transition_codec import (
    canonical_minigrid_schema,
    decode_minigrid_transition,
    minigrid_contract,
    minigrid_effect_cell_nll,
    validate_schema as validate_minigrid_schema,
)
from . import AttentionWM_support


def crafter_selection_loss(object_nll, direction_nll, slot_gate_nll, changed_value_nll):
    """Fixed non-target checkpoint-selection objective."""
    return torch.stack((object_nll, direction_nll, slot_gate_nll, changed_value_nll)).mean()
from . import Embedding_support
from . import MLP_support
import pandas as pd
import numpy as np


def _remove_legacy_sharded_tensor_state_hooks(module: nn.Module) -> int:
    """Remove obsolete Lightning hooks from models without ShardedTensors.

    The PyTorch Lightning version used by this project registers the legacy
    ShardedTensor save/load hooks on every ``LightningModule``.  The save hook
    scans every value in ``module.__dict__``.  Once a Trainer-owned weak proxy
    expires, merely calling ``state_dict()`` can therefore raise
    ``ReferenceError: weakly-referenced object no longer exists`` even though
    this model contains only ordinary tensors.

    Only the two known ShardedTensor compatibility hooks are removed; all
    other PyTorch and project checkpoint hooks are preserved.
    """
    removed = 0
    hook_groups = (
        ("_state_dict_hooks", {"state_dict_hook"}),
        ("_load_state_dict_pre_hooks", {"pre_load_state_dict_hook"}),
    )
    sharded_modules = {
        "torch.distributed._shard.sharded_tensor",
        "torch.distributed._sharded_tensor",
    }
    for attr_name, expected_names in hook_groups:
        hooks = getattr(module, attr_name, None)
        if hooks is None:
            continue
        for key, hook in list(hooks.items()):
            # PyTorch wraps load pre-hooks in _WrappedHook; its original
            # function is available as ``hook`` (and as ``__wrapped__``).
            function = getattr(hook, "hook", getattr(hook, "__wrapped__", hook))
            if (
                getattr(function, "__module__", "") in sharded_modules
                and getattr(function, "__name__", "") in expected_names
            ):
                del hooks[key]
                removed += 1
    return removed


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
        self.ewc_enabled = bool(getattr(hparams, "ewc_enabled", False))
        self.lambda_ewc = float(getattr(hparams, "lambda_ewc", 1.0))
        self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
        self.loss_map_result = None
        self.coverage_map_result = None
        self.inventory_loss_vector_result = None
        self.inventory_loss_accumulator = []
        self._val_step_outputs = []
        self._val_natural_outputs = []
        self._val_field_outputs = []
        self._val_minigrid_diagnostics = []
        self._val_minigrid_effect_diagnostics = []
        self._val_stochastic_outcomes = []
        # Crafter's changed-only diagnostic is accumulated as a sum/count,
        # rather than as a mean of batch means, so validation reports one
        # value over every changed unit in the full epoch.
        self._val_crafter_changed_loss_sum = None
        self._val_crafter_changed_count = None
        self._val_crafter_focal_diagnostics = None
        self._val_crafter_inventory_diagnostics = None

        self.fisher = 0
        self.old_params = None
        self.env_type = hparams.env_type
        # Crafter's native rule tracker owns health/food/drink/energy during
        # imagined planning.  Keep the default true so old configs and
        # checkpoints retain their original training/decoding behavior.
        self.predict_survival = bool(getattr(hparams, "predict_survival", True))
        pose_cfg = getattr(hparams, "crafter_pose", None)
        if isinstance(pose_cfg, dict):
            self.crafter_pose_enabled = bool(pose_cfg.get("enabled", False))
            self.crafter_pose_mode = str(pose_cfg.get("mode", "learned_detached")).lower()
        else:
            self.crafter_pose_enabled = bool(getattr(pose_cfg, "enabled", False)) if pose_cfg is not None else False
            self.crafter_pose_mode = str(getattr(pose_cfg, "mode", "learned_detached")).lower() if pose_cfg is not None else "learned_detached"
        if self.crafter_pose_enabled and self.env_type != "crafter":
            raise ValueError("crafter_pose.enabled is only valid for the Crafter domain")
        if self.crafter_pose_mode != "learned_detached":
            raise ValueError("crafter_pose.mode must be 'learned_detached'")
        self._val_crafter_pose_diagnostics = []
        self.crafter_inventory_output_mode = str(
            getattr(hparams, "crafter_inventory_output_mode", "categorical_gate")
        ).strip().lower()
        if self.crafter_inventory_output_mode not in {
            "categorical_gate", "categorical_effect"
        }:
            raise ValueError(
                "crafter_inventory_output_mode must be 'categorical_gate' or "
                f"'categorical_effect', got {self.crafter_inventory_output_mode!r}"
            )
        # The legacy gate head predicts KEEP/CHANGE plus a conditional value.
        # The unified effect head instead emits KEEP/-4/-2/-1/+1 directly.
        self.crafter_inventory_value_mode = str(
            getattr(hparams, "crafter_inventory_value_mode", "categorical_absolute")
        ).strip().lower()
        if self.crafter_inventory_value_mode not in {
            "categorical_absolute", "categorical_delta"
        }:
            raise ValueError(
                "crafter_inventory_value_mode must be 'categorical_absolute' "
                "or 'categorical_delta', got "
                f"{self.crafter_inventory_value_mode!r}"
            )
        self.crafter_inventory_effect_reduction = str(
            getattr(hparams, "crafter_inventory_effect_reduction", "balanced_mean")
        ).strip().lower()
        if self.crafter_inventory_effect_reduction not in {
            "balanced_mean", "sqrt_balanced", "mean"
        }:
            raise ValueError(
                "crafter_inventory_effect_reduction must be 'balanced_mean', "
                "'sqrt_balanced', or 'mean', got "
                f"{self.crafter_inventory_effect_reduction!r}"
            )
        self.crafter_inventory_architecture = str(
            getattr(hparams, "crafter_inventory_architecture", "shared_pool_v1")
        ).strip().lower()
        if self.crafter_inventory_architecture not in {
            "shared_pool_v1", "isolated_global_v1", "slot_attention_v1"
        }:
            raise ValueError(
                "crafter_inventory_architecture must be shared_pool_v1, "
                "isolated_global_v1, or slot_attention_v1"
            )
        if (self.env_type == "crafter"
                and self.crafter_inventory_architecture != "shared_pool_v1"
                and self.predict_survival):
            raise ValueError(
                "isolated Crafter inventory architectures require predict_survival=false"
            )
        event_residual_cfg = getattr(hparams, "crafter_inventory_event_residual", None)
        if isinstance(event_residual_cfg, dict):
            self.crafter_inventory_event_residual_enabled = bool(event_residual_cfg.get("enabled", False))
            self.crafter_inventory_event_residual_hidden_dims = tuple(event_residual_cfg.get("hidden_dims", (64,)))
            self.crafter_inventory_event_residual_action_embed_dim = int(event_residual_cfg.get("action_embed_dim", 8))
            self.crafter_inventory_event_residual_change_bias = float(event_residual_cfg.get("change_bias", 1.875))
            self.crafter_inventory_event_residual_loss_weight = float(event_residual_cfg.get("loss_weight", 1.0))
        else:
            self.crafter_inventory_event_residual_enabled = bool(getattr(event_residual_cfg, "enabled", False)) if event_residual_cfg is not None else False
            self.crafter_inventory_event_residual_hidden_dims = tuple(getattr(event_residual_cfg, "hidden_dims", (64,))) if event_residual_cfg is not None else (64,)
            self.crafter_inventory_event_residual_action_embed_dim = int(getattr(event_residual_cfg, "action_embed_dim", 8)) if event_residual_cfg is not None else 8
            self.crafter_inventory_event_residual_change_bias = float(getattr(event_residual_cfg, "change_bias", 1.875)) if event_residual_cfg is not None else 1.875
            self.crafter_inventory_event_residual_loss_weight = float(getattr(event_residual_cfg, "loss_weight", 1.0)) if event_residual_cfg is not None else 1.0
        if self.crafter_inventory_event_residual_enabled and (
            self.env_type != "crafter" or self.crafter_inventory_output_mode != "categorical_effect"
        ):
            raise ValueError("Crafter event residual requires crafter categorical_effect output")
        # ``stochastic_enabled`` is the single domain-level switch exposed by
        # the MiniGrid configs.  Keep the model-specific fields below as
        # backwards-compatible internal/ablation controls, but do not require
        # users to repeat the switch in every pipeline block.
        domain_stochastic_setting = getattr(hparams, "stochastic_enabled", None)
        domain_stochastic_enabled = bool(domain_stochastic_setting)
        stochastic_latent_cfg = getattr(hparams, "stochastic_latent", None)
        if isinstance(stochastic_latent_cfg, dict):
            self.stochastic_latent_enabled = bool(stochastic_latent_cfg.get("enabled", False))
            self.stochastic_latent_loss_weight = float(stochastic_latent_cfg.get("loss_weight", 1.0))
        else:
            self.stochastic_latent_enabled = bool(
                getattr(stochastic_latent_cfg, "enabled", False)
            ) if stochastic_latent_cfg is not None else False
            self.stochastic_latent_loss_weight = float(
                getattr(stochastic_latent_cfg, "loss_weight", 1.0)
            ) if stochastic_latent_cfg is not None else 1.0
        # v1 is retained as a labelled outcome-classifier ablation. Legacy
        # hparams without the domain switch can still select it explicitly;
        # resolved MiniGrid configs always let the domain switch win.
        configured_stochastic_model = getattr(hparams, "stochastic_model", None)
        if domain_stochastic_setting is not None:
            # A resolved domain flag is authoritative in both directions.
            # This prevents a legacy per-block model selector from enabling a
            # stochastic WM while the environment is deterministic.
            configured_stochastic_model = (
                "latent_v2" if domain_stochastic_enabled else "none"
            )
        elif configured_stochastic_model is None:
            configured_stochastic_model = (
                "latent_v2" if domain_stochastic_enabled
                else ("outcome_v1" if self.stochastic_latent_enabled else "none")
            )
        self.stochastic_model = str(configured_stochastic_model).lower()
        if self.stochastic_model not in {"none", "outcome_v1", "latent_v2"}:
            raise ValueError("stochastic_model must be one of none, outcome_v1, latent_v2")
        self.stochastic_latent_v2_enabled = self.stochastic_model == "latent_v2"
        self.stochastic_latent_enabled = self.stochastic_model == "outcome_v1"
        def _latent_cfg_value(name, default):
            if isinstance(stochastic_latent_cfg, dict):
                return stochastic_latent_cfg.get(name, default)
            return getattr(stochastic_latent_cfg, name, default) if stochastic_latent_cfg is not None else default
        self.latent_num_factors = int(_latent_cfg_value("num_factors", 1))
        self.latent_num_classes = int(_latent_cfg_value("num_classes", 2))
        self.latent_kl_weight = float(_latent_cfg_value("kl_weight", 0.1))
        self.latent_supervision_weight = float(_latent_cfg_value("supervision_weight", 1.0))
        if (self.stochastic_latent_enabled or self.stochastic_latent_v2_enabled) and self.env_type != "minigrid":
            raise ValueError("MiniGrid stochastic models are currently supported only for MiniGrid")
        self.frame_stack = getattr(hparams, "frame_stack", 1)
        self.observation_schema = list(getattr(hparams, "observation_schema", []) or [])
        # Categorical-effect fields are shared by MiniGrid and Crafter.  Keep
        # the loss parameter available for every schema, while domain configs
        # decide whether focal modulation is enabled.
        self.focal_gamma = float(getattr(hparams, "focal_gamma", 0.0))
        if self.env_type == "minigrid":
            self.minigrid_transition_mode = str(
                getattr(hparams, "minigrid_transition_mode", "effect")
            )
            self.effect_reduction = str(
                getattr(hparams, "effect_reduction", "balanced_mean")
            ).strip().lower()
            if not self.observation_schema:
                self.observation_schema = canonical_minigrid_schema(
                    self.minigrid_transition_mode,
                    self.effect_reduction,
                )
            else:
                validate_minigrid_schema(
                    self.observation_schema,
                    self.minigrid_transition_mode,
                    self.effect_reduction,
                )
            self.minigrid_contract = minigrid_contract(self.minigrid_transition_mode)
            if (
                self.minigrid_transition_mode == "effect"
                and int(getattr(hparams, "action_norm_values", 6)) != self.minigrid_contract.action_count
            ):
                raise ValueError(
                    "MiniGrid effect mode requires action_norm_values=6 "
                    f"(compact actions), got {getattr(hparams, 'action_norm_values', None)!r}"
                )
        if not self.observation_schema:
            raise ValueError("attention_model.observation_schema must define at least one field")
        self.use_bipedal_flag = bool(getattr(hparams, "use_bipedal_attention", False))
        # Keep a single explicit flag for the vector-state BipedalWalker path.
        # Some call sites check `self.is_bipedal`, so define it before first use.
        self.is_bipedal = (self.env_type == "bipedalwalker") or self.use_bipedal_flag
        print(
            f"[AttentionWM Init] env_type: {self.env_type} | "
            f"data_type: {self.data_type} | val_metric: "
            f"{getattr(hparams, 'validation_metric', 'mse')}"
            + (
                f" | effect_reduction: {self.effect_reduction}"
                if self.env_type == "minigrid"
                and self.minigrid_transition_mode == "effect"
                else ""
            )
        )
        self.crafter_inventory_gate_reduction = str(
            getattr(hparams, "crafter_inventory_gate_reduction", "global_balanced")
        ).strip().lower()
        if self.crafter_inventory_gate_reduction not in {
            "global_balanced", "slot_macro_changed"
        }:
            raise ValueError(
                "crafter_inventory_gate_reduction must be 'global_balanced' or "
                "'slot_macro_changed'"
            )
        
        MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
        # Initialize the prediction model.
        module_class = MODEL_MAPPING.get(hparams.model_type.lower())
        if module_class is not None:
            if (
                self.env_type == "minigrid"
                and self.minigrid_transition_mode == "effect"
                and module_class is not AttentionWM_support.AttentionModule
            ):
                raise ValueError(
                    "MiniGrid effect mode currently requires model_type=Attention; "
                    "Embedding/MLP heads do not expose the canonical 24/8 outputs."
                )
            module_kwargs = {
                "env_type": "bipedalwalker" if self.is_bipedal else self.env_type,
                "frame_stack": self.frame_stack,
            }
            if module_class is AttentionWM_support.AttentionModule and self.env_type == "crafter":
                uses_effects = any(
                    str(spec.get("distribution", "")) == "categorical_effect"
                    for spec in self.observation_schema
                )
                module_kwargs["crafter_output_mode"] = (
                    "effect" if uses_effects else "absolute"
                )
                inventory_spec = next(
                    (
                        spec for spec in self.observation_schema
                        if str(spec.get("name", "")) == "inventory"
                    ),
                    None,
                )
                if inventory_spec is None:
                    raise ValueError("Crafter observation schema must define inventory")
                inventory_distribution = str(
                    inventory_spec.get("distribution", "")
                )
                inventory_target_mode = str(
                    inventory_spec.get("target_mode", "")
                )
                expected_distribution = (
                    "categorical_inventory_gate" if self.crafter_inventory_output_mode == "categorical_gate"
                    else "categorical_inventory_effect"
                )
                expected_target_mode = self.crafter_inventory_output_mode
                if (inventory_distribution != expected_distribution
                    or inventory_target_mode != expected_target_mode):
                    raise ValueError(
                        "Crafter inventory schema does not match configured output mode: "
                        f"expected {expected_distribution}/{expected_target_mode}, got "
                        f"{inventory_distribution}/{inventory_target_mode}"
                    )
                schema_classes = int(inventory_spec.get("classes", 10))
                if self.crafter_inventory_output_mode == "categorical_effect":
                    effect_classes = int(inventory_spec.get("effect_classes", 5))
                    if schema_classes != INVENTORY_VALUES or effect_classes != 5:
                        raise ValueError(
                            "Crafter categorical_inventory_effect schema must define "
                            f"classes={INVENTORY_VALUES} and effect_classes=5, got "
                            f"classes={schema_classes}, effect_classes={effect_classes}"
                        )
                    # ``crafter_inventory_classes`` describes the 0..9 value
                    # range used by the retained survival-effect rows.  The
                    # item-effect class count is fixed separately at five.
                    module_kwargs["crafter_inventory_classes"] = INVENTORY_VALUES
                else:
                    if schema_classes != INVENTORY_VALUES:
                        raise ValueError(
                            "Crafter categorical_inventory_gate schema must define "
                            f"{INVENTORY_VALUES} inventory values, got {schema_classes}"
                        )
                    module_kwargs["crafter_inventory_classes"] = schema_classes
                module_kwargs["crafter_inventory_output_mode"] = inventory_target_mode
                module_kwargs["crafter_inventory_value_mode"] = (
                    self.crafter_inventory_value_mode
                )
                module_kwargs["crafter_pose_enabled"] = self.crafter_pose_enabled
                module_kwargs["crafter_inventory_architecture"] = (
                    self.crafter_inventory_architecture
                )
                module_kwargs["crafter_inventory_event_residual_enabled"] = (
                    self.crafter_inventory_event_residual_enabled
                )
                module_kwargs["crafter_inventory_event_residual_hidden_dims"] = (
                    self.crafter_inventory_event_residual_hidden_dims
                )
                module_kwargs["crafter_inventory_event_residual_action_embed_dim"] = (
                    self.crafter_inventory_event_residual_action_embed_dim
                )
                module_kwargs["crafter_inventory_event_residual_change_bias"] = (
                    self.crafter_inventory_event_residual_change_bias
                )
            if module_class is AttentionWM_support.AttentionModule and self.env_type == "minigrid":
                module_kwargs["minigrid_transition_mode"] = self.minigrid_transition_mode
                module_kwargs["stochastic_outcome"] = self.stochastic_latent_enabled
                module_kwargs["stochastic_model"] = self.stochastic_model
                module_kwargs["latent_num_factors"] = self.latent_num_factors
                module_kwargs["latent_num_classes"] = self.latent_num_classes
            self.model = module_class(
                hparams.data_type,
                hparams.grid_shape,
                hparams.attention_mask_size,
                hparams.embed_dim,
                hparams.num_heads,
                **module_kwargs,
            )
            if (
                self.env_type == "crafter"
                and module_class is AttentionWM_support.AttentionModule
            ):
                # The model module records its architecture; these two fields
                # are rollout semantics owned by the Lightning wrapper.
                self.model.checkpoint_contract.update({
                    "predict_survival": self.predict_survival,
                    "obs_norm_values": [
                        float(value) for value in getattr(hparams, "obs_norm_values", ())
                    ],
                })
        else:
            print(f"Model type: {hparams.model_type} not supported")
            exit()
        

        if hparams.freeze_weight:
            utils.load_model_weight(self.model, hparams.model_save_path)
        # Constructing the visualization helper creates its output directory.
        # Keep it entirely out of headless training when visualization is off.
        self.visual_func = (
            minigrid_utils.Visualization(hparams)
            if self.visualizationFlag
            else None
        )
        self.train_token_loss_accumulator = {}
        self.train_token_acc_accumulator = {}
        self.train_token_loss_csv_path = None
        if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
            metrics_dir = Path(
                str(getattr(hparams, "metrics_dir", utils.WM_RESULTS_PATH / "world_model"))
            )
            self.train_token_loss_csv_path = os.path.join(
                str(metrics_dir),
                "bipedal_train_token_losses.csv",
            )
        self.save_hyperparameters(hparams)
        # This model never owns torch.distributed ShardedTensor attributes.
        # Removing Lightning's legacy compatibility hooks prevents checkpoint
        # saves from inspecting expired Trainer weak references.
        _remove_legacy_sharded_tensor_state_hooks(self)


    def save_old_params(self):
        """Save current model parameters for EWC, moved to model's device."""
        device = next(self.parameters()).device
        old_params = {k: v.clone().detach().cpu() 
              for k, v in self.state_dict().items()}
        return old_params

    # def load_old_params(self, old_params):
    #     """Load previous-task parameters into the current model."""
    #     for n, p in self.named_parameters():
    #         if n in old_params:
    #             p.data.copy_(old_params[n])  # Overwrite the current parameter with the saved value.

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
    #         loss, _ = self.observation_loss(obs_pred, obs_next, obs_prev=obs)
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

        fisher = {
            n: torch.zeros_like(p, dtype=torch.float32, device=device)
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("model.crafter_pose_head.")
        }

        count = 0
        total_samples_target = int(samples)
        
        print(f"[Fisher] Starting computation for ~{total_samples_target} samples...")

        for i, batch in enumerate(dataloader):
            if count >= total_samples_target:
                break
            
            # 1. Preprocess batch
            # ``preprocess_batch`` already returns the agent-centred current
            # and next observations as its first and third values.  Its fifth
            # value is only an (B, H, W) elements mask for diagnostics, not an
            # observation tensor.
            obs, act, obs_next, info, _, _, inv, inv_next = self.preprocess_batch(batch)
            # 2. Iterate over samples in the batch
            batch_size = obs.shape[0]
            for b in range(batch_size):
                if count >= total_samples_target:
                    break
                    
                self.zero_grad(set_to_none=True)

                # Extract single sample (keep dim 0 for model compatibility)
                s_obs = obs[b:b+1].to(device, dtype=torch.float32)
                s_act = act[b:b+1].to(device)
                s_info = {
                    k: (v[b:b+1].to(device) if torch.is_tensor(v) else v[b:b+1])
                    for k, v in info.items()
                } if info is not None else None
                s_obs_next = obs_next[b:b+1].to(device, dtype=torch.float32)
                s_inv = inv[b:b+1].to(device) if inv is not None else None
                s_inv_next = inv_next[b:b+1].to(device) if inv_next is not None else None

                # Forward & Backward (fp32)
                with torch.amp.autocast("cuda", enabled=False):
                    # The model is trained on the agent-centred mask, so
                    # Fisher must use exactly the same input/target geometry.
                    # Feeding the full map here silently changed the token
                    # layout and made the continual-learning importance
                    # estimate unrelated to the training objective.
                    pred, _, s_inv_pred = self(s_obs, s_act, s_info, inv=s_inv)
                    loss_sample, _ = self.observation_loss(
                        pred,
                        s_obs_next,
                        obs_prev=s_obs,
                        aux_pred=s_inv_pred,
                        inv=s_inv,
                        inv_next=s_inv_next,
                    )
                
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
                # Preserve unrelated hooks and remove only the obsolete
                # ShardedTensor compatibility hooks described above.
                _remove_legacy_sharded_tensor_state_hooks(self)

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
                if n.startswith("model.crafter_pose_head."):
                    continue
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
                num_layers = len([
                    n for n, p in self.named_parameters()
                    if p.requires_grad and not n.startswith("model.crafter_pose_head.")
                ])
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

    @staticmethod
    def average_loss_and_coverage_maps(loss_accumulator):
        rows = len(loss_accumulator)
        cols = len(loss_accumulator[0]) if rows else 0
        avg_loss_map = np.zeros((rows, cols), dtype=np.float32)
        coverage_map = np.zeros((rows, cols), dtype=np.float32)
        for y, row in enumerate(loss_accumulator):
            for x, values in enumerate(row):
                if values:
                    avg_loss_map[y, x] = float(np.mean(values))
                    coverage_map[y, x] = 1.0
        return avg_loss_map, coverage_map

    def compute_cell_loss(self, next_pred, next_true, current=None):
        # Compute the per-cell error map.
        if self.env_type == 'crafter':
            if current is None:
                raise ValueError("Crafter effect cell loss requires the current state")
            loss_map = crafter_effect_loss(
                next_pred, current, next_true, reduction='none'
            )
        elif self.env_type == 'minigrid':
            if current is None:
                raise ValueError("MiniGrid categorical cell loss requires the current state")
            current_frame = current[:, -next_true.size(1):]
            loss_map, _ = minigrid_effect_cell_nll(
                next_pred,
                current_frame,
                next_true,
                mode=self.minigrid_transition_mode,
            )
        else:
            # Standard Regression error
            error = torch.abs(next_pred - next_true)
            loss_map = error.mean(dim=1)  # (B, H, W)

        return loss_map

    @torch.no_grad()
    def encode_map_features(self, state):
        """Expose the predictive MiniGrid map representation for UED novelty."""
        if self.env_type != "minigrid":
            raise ValueError("encode_map_features is only available for MiniGrid")
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model.encode_map_features(state)
        finally:
            self.model.train(was_training)

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

    def forward_pose(self, state, action, info=None, inv=None):
        """Ordinary Crafter transition plus opt-in detached pose logits."""
        if not self.crafter_pose_enabled:
            raise ValueError("forward_pose requires crafter_pose.enabled=true")
        return self.model(state, action, info, inv=inv, return_pose=True)

    @staticmethod
    def _crafter_pose_loss_and_metrics(pose_logits, current, following):
        target = crafter_pose_target(current, following)
        prediction = pose_logits.argmax(dim=1)
        return (
            F.cross_entropy(pose_logits, target),
            {
                "joint_accuracy": prediction.eq(target).float().mean(),
                "position_accuracy": prediction.div(4, rounding_mode="floor").eq(
                    target.div(4, rounding_mode="floor")
                ).float().mean(),
                "direction_accuracy": prediction.remainder(4).eq(target.remainder(4)).float().mean(),
            },
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Permit an old Crafter checkpoint to initialize an opt-in pose head.

        All other missing/unexpected keys retain PyTorch's strict semantics.
        """
        if self.crafter_pose_enabled:
            result = super().load_state_dict(state_dict, strict=False)
            allowed = lambda names: all(name.startswith("model.crafter_pose_head.") for name in names)
            if strict and (not allowed(result.missing_keys) or result.unexpected_keys):
                raise RuntimeError(
                    "Incompatible checkpoint outside optional Crafter pose head; "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            return result
        # A pose-enabled artifact may be inspected by a legacy/disabled
        # evaluator.  Ignore only its optional readout keys; the historical
        # transition model remains strict.
        pose_keys = [key for key in state_dict if key.startswith("model.crafter_pose_head.")]
        if pose_keys:
            filtered = {key: value for key, value in state_dict.items() if key not in pose_keys}
            return super().load_state_dict(filtered, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    def forward_stochastic(self, state, action, info, inv=None):
        """Predict the MiniGrid success branch and its two-way outcome prior.

        The ordinary ``forward`` contract intentionally remains unchanged for
        deterministic experiments and existing rollout consumers.
        """
        if not self.stochastic_latent_enabled:
            raise ValueError("forward_stochastic requires stochastic_latent.enabled=true")
        state_logits, attention, inventory_logits, outcome_logits = self.model(
            state, action, info, inv=inv, return_outcome=True
        )
        diagnostics = {
            "state_logits": state_logits,
            "attention_weights": attention,
            "inventory_logits": inventory_logits,
            "outcome_logits": outcome_logits,
            "outcome_probs": torch.softmax(outcome_logits, dim=-1),
        }

    def forward_distribution(
        self, state, action, info=None, inv=None, next_state=None,
        next_inventory=None, latent_labels=None, sample_mode="sample", generator=None,
    ):
        """One-step categorical latent distribution (v2).

        `next_state` is training-only: when absent, the returned sample comes
        from the prior and is safe for future imagined rollout integration.
        """
        if not self.stochastic_latent_v2_enabled:
            raise ValueError("forward_distribution requires stochastic_model='latent_v2'")
        return self.model(
            state, action, info, inv=inv, return_distribution=True,
            next_state=next_state, next_inventory=next_inventory,
            latent_labels=latent_labels,
            sample_mode=sample_mode, generator=generator,
        )

    @staticmethod
    def _categorical_kl(posterior_logits, prior_logits):
        posterior_log_probs = F.log_softmax(posterior_logits, dim=-1)
        posterior_probs = posterior_log_probs.exp()
        return (posterior_probs * (posterior_log_probs - F.log_softmax(prior_logits, dim=-1))).sum(dim=-1).mean()

    @staticmethod
    def _optional_action_failed_target(info, batch_size, device):
        if not isinstance(info, dict) or "action_failed" not in info:
            return None
        target = torch.as_tensor(info["action_failed"], device=device).reshape(-1).long()
        if target.numel() != batch_size or ((target < 0) | (target > 1)).any():
            raise ValueError("Invalid MiniGrid action_failed outcome target")
        return target

    def _stochastic_outcome_target(self, info, batch_size, device):
        if not self.stochastic_latent_enabled:
            return None
        if not isinstance(info, dict) or "action_failed" not in info:
            raise ValueError(
                "MiniGrid stochastic latent training requires batched info['action_failed']; "
                "recollect the dataset with the stochastic environment."
            )
        target = torch.as_tensor(info["action_failed"], device=device).reshape(-1).long()
        if target.numel() != batch_size or ((target < 0) | (target > 1)).any():
            raise ValueError("Invalid MiniGrid action_failed outcome target")
        return target


    @staticmethod
    def _symlog(value: torch.Tensor) -> torch.Tensor:
        """Scale count-like values without a domain-specific hand-tuned weight."""
        return torch.sign(value) * torch.log1p(torch.abs(value))

    @staticmethod
    def _normalized_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Categorical NLL normalized so a uniform predictor has loss 1."""
        target = target.long()
        if target.numel() and (target.min() < 0 or target.max() >= num_classes):
            raise ValueError(
                f"Categorical target range [{int(target.min())}, {int(target.max())}] "
                f"is invalid for {num_classes} classes"
            )
        return F.cross_entropy(
            logits,
            target,
            label_smoothing=label_smoothing,
        ) / math.log(num_classes)

    def observation_loss(
        self,
        obs_pred: torch.Tensor,
        obs_next: torch.Tensor,
        obs_prev: torch.Tensor | None = None,
        aux_pred=None,
        inv: torch.Tensor | None = None,
        inv_next: torch.Tensor | None = None,
        focal_gamma: float | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute one schema-driven observation prediction objective.

        Every observation field contributes one normalized likelihood/error
        term and the final objective is their unweighted mean.  Domain names
        select the observation schema only; they do not introduce manually
        tuned loss multipliers.
        """
        fields: Dict[str, torch.Tensor] = {}
        for spec in self.observation_schema:
            name = str(spec["name"])
            distribution = str(spec["distribution"])
            prediction_source = str(spec.get("prediction_source", "state"))
            target_source = str(spec.get("target_source", "observation"))
            target_mode = str(spec.get("target_mode", "delta"))

            if prediction_source == "state":
                if "prediction_slice" in spec:
                    start, stop = map(int, spec["prediction_slice"])
                    prediction = obs_pred[:, start:stop]
                else:
                    indices = list(map(int, spec["indices"]))
                    prediction = obs_pred[:, indices]
            elif prediction_source == "auxiliary":
                if aux_pred is None:
                    raise ValueError(f"Missing auxiliary prediction for observation field '{name}'")
                prediction = aux_pred
            elif prediction_source == "contact_logits":
                if not isinstance(aux_pred, dict) or "contact_logits" not in aux_pred:
                    raise ValueError(f"Missing contact logits for observation field '{name}'")
                prediction = aux_pred["contact_logits"][name]
            else:
                raise ValueError(f"Unknown prediction source '{prediction_source}' for field '{name}'")

            if target_source == "inventory":
                if inv_next is None:
                    raise ValueError(f"Missing inventory target for observation field '{name}'")
                target = inv_next.float()
                if target_mode == "delta":
                    if inv is None:
                        raise ValueError(f"Inventory delta field '{name}' requires the current inventory")
                    target = target - inv.float()
                elif target_mode not in {"absolute", "categorical_effect", "categorical_gate"}:
                    raise ValueError(
                        f"Unknown inventory target_mode '{target_mode}' for field '{name}'"
                    )
                if target_mode in {"categorical_effect", "categorical_gate"}:
                    if inv is None:
                        raise ValueError("Missing current discrete inventory")
                    validate_discrete_inventory(inv, "current")
                    validate_discrete_inventory(inv_next, "next")
            else:
                if "target_index" in spec:
                    target_index = int(spec["target_index"])
                    target = obs_next[:, target_index]
                    if target_mode == "current_plus_delta":
                        if obs_prev is None:
                            raise ValueError(f"Field '{name}' requires the current observation")
                        current = obs_prev[:, -obs_next.size(1):, ...][:, target_index]
                        target = current + target
                else:
                    indices = list(map(int, spec["indices"]))
                    target = obs_next[:, indices]
                    if target_mode == "current_plus_delta":
                        if obs_prev is None:
                            raise ValueError(f"Field '{name}' requires the current observation")
                        target = obs_prev[:, indices] + target

            if distribution == "categorical":
                fields[name] = self._normalized_cross_entropy(
                    prediction,
                    target,
                    int(spec["classes"]),
                    label_smoothing=float(spec.get("label_smoothing", 0.0)),
                )
            elif distribution == "bernoulli":
                fields[name] = F.binary_cross_entropy_with_logits(
                    prediction, target.clamp(0.0, 1.0)
                ) / math.log(2.0)
            elif distribution == "categorical_effect":
                if target_source == "inventory":
                    if inv is None:
                        raise ValueError(
                            f"Categorical inventory effect field '{name}' requires current inventory"
                        )
                    current = inv
                else:
                    if obs_prev is None:
                        raise ValueError(
                            f"Categorical effect field '{name}' requires the current observation"
                        )
                    if "target_index" not in spec:
                        raise ValueError(
                            f"Categorical effect field '{name}' requires target_index"
                        )
                    current_frame = obs_prev[:, -obs_next.size(1):, ...]
                    current = current_frame[:, int(spec["target_index"])]
                effect_target = categorical_effect_target(current, target)
                fields[name] = balanced_categorical_effect_loss(
                    prediction,
                    effect_target,
                    reduction=str(spec.get("effect_reduction", "balanced_mean")),
                    label_smoothing=float(spec.get("label_smoothing", 0.0)),
                    focal_gamma=self.focal_gamma if focal_gamma is None else float(focal_gamma),
                )
            elif distribution == "categorical_inventory_gate":
                if target_source != "inventory" or inv is None or inv_next is None:
                    raise ValueError(
                        "categorical_inventory_gate requires current and next inventory"
                    )
                fields[name], _ = crafter_inventory_gate_loss(
                    prediction, inv, inv_next,
                    predict_survival=self.predict_survival,
                    value_mode=self.crafter_inventory_value_mode,
                    gate_reduction=self.crafter_inventory_gate_reduction,
                )
            elif distribution == "categorical_inventory_effect":
                if target_source != "inventory" or inv is None or inv_next is None:
                    raise ValueError(
                        "categorical_inventory_effect requires current and next inventory"
                    )
                fields[name], _ = crafter_inventory_effect_loss(
                    prediction, inv, inv_next,
                    predict_survival=self.predict_survival,
                    effect_reduction=str(getattr(
                        self, "crafter_inventory_effect_reduction", "balanced_mean"
                    )),
                )
                if self.crafter_inventory_event_residual_enabled:
                    residual_loss, _ = crafter_inventory_event_residual_loss(
                        prediction, inv, inv_next,
                    )
                    fields[name] = fields[name] + (
                        self.crafter_inventory_event_residual_loss_weight * residual_loss
                    )
            elif distribution == "mse":
                fields[name] = F.mse_loss(prediction, target)
            elif distribution == "symlog_mse":
                fields[name] = F.mse_loss(self._symlog(prediction), self._symlog(target))
            else:
                raise ValueError(f"Unknown distribution '{distribution}' for field '{name}'")

        if not fields:
            raise ValueError("The observation schema produced no loss fields")
        return torch.stack(list(fields.values())).mean(), fields

    def _minigrid_effect_diagnostics(self, obs_pred, obs_next, obs, inv, inv_next, aux_pred):
        """Return aggregate changed-effect and false-SET validation metrics."""
        changed_losses, false_set_rates = [], []
        changed_count = obs_pred.new_zeros(())
        if self.env_type != "minigrid" or self.minigrid_transition_mode != "effect":
            return obs_pred.new_zeros(()), obs_pred.new_zeros(()), changed_count
        for spec in self.observation_schema:
            if str(spec.get("distribution")) != "categorical_effect":
                continue
            if str(spec.get("target_source", "observation")) == "inventory":
                if inv is None or inv_next is None or not isinstance(aux_pred, torch.Tensor):
                    continue
                current, target, prediction = inv, inv_next, aux_pred
            else:
                index = int(spec["target_index"])
                current, target = obs[:, index], obs_next[:, index]
                start, stop = map(int, spec["prediction_slice"])
                prediction = obs_pred[:, start:stop]
            effects = categorical_effect_target(current, target)
            per_cell = balanced_categorical_effect_loss(
                prediction, effects, reduction="none",
                label_smoothing=float(spec.get("label_smoothing", 0.0)),
                focal_gamma=self.focal_gamma,
            )
            changed = effects.ne(0)
            if changed.any():
                changed_losses.append(per_cell[changed].mean())
                changed_count = changed_count + changed.sum().to(dtype=changed_count.dtype)
            predicted = prediction.argmax(dim=1)
            keep = effects.eq(0)
            if keep.any():
                false_set_rates.append(predicted[keep].ne(0).float().mean())
        changed_loss = torch.stack(changed_losses).mean() if changed_losses else obs_pred.new_zeros(())
        false_set = torch.stack(false_set_rates).mean() if false_set_rates else obs_pred.new_zeros(())
        return changed_loss, false_set, changed_count

    def _crafter_changed_diagnostics(
        self, obs_pred, obs_next, obs, inv, inv_next, inv_pred
    ):
        """Return natural-CE sum/count over Crafter units that actually change.

        This is validation-only.  Object and direction effects contribute one
        term per changed cell.  Each changed inventory slot contributes one
        term: survival uses its effect CE; items average their gate and value
        CEs so the two prediction heads do not double-count one slot.
        """
        loss_sum = obs_pred.new_zeros(())
        changed_count = obs_pred.new_zeros(())
        if self.env_type != "crafter":
            return loss_sum, changed_count

        for spec in self.observation_schema:
            if str(spec.get("distribution")) != "categorical_effect":
                continue
            if str(spec.get("target_source", "observation")) == "inventory":
                # Crafter inventory uses the explicit gate distribution below.
                continue
            index = int(spec["target_index"])
            start, stop = map(int, spec["prediction_slice"])
            effects = categorical_effect_target(obs[:, index], obs_next[:, index])
            changed = effects.ne(0)
            if changed.any():
                per_cell = balanced_categorical_effect_loss(
                    obs_pred[:, start:stop], effects, reduction="none",
                    label_smoothing=float(spec.get("label_smoothing", 0.0)),
                    focal_gamma=0.0,
                )
                loss_sum = loss_sum + per_cell[changed].sum()
                changed_count = changed_count + changed.sum().to(dtype=changed_count.dtype)

        if inv is None or inv_next is None or not isinstance(inv_pred, dict):
            return loss_sum, changed_count
        validate_discrete_inventory(inv, "current")
        validate_discrete_inventory(inv_next, "next")
        current, following = inv.long(), inv_next.long()

        if self.predict_survival:
            survival_effect = categorical_effect_target(
                current[:, :SURVIVAL_SLOTS], following[:, :SURVIVAL_SLOTS]
            )
            survival_changed = survival_effect.ne(0)
            if survival_changed.any():
                survival_nll = balanced_categorical_effect_loss(
                    inv_pred["survival_effect_logits"], survival_effect, reduction="none",
                    focal_gamma=0.0,
                )
                loss_sum = loss_sum + survival_nll[survival_changed].sum()
                changed_count = changed_count + survival_changed.sum().to(dtype=changed_count.dtype)

        item_current = current[:, SURVIVAL_SLOTS:]
        item_following = following[:, SURVIVAL_SLOTS:]
        item_changed = item_following.ne(item_current)
        if item_changed.any():
            if getattr(self, "crafter_inventory_output_mode", "categorical_gate") == "categorical_effect":
                item_target = categorical_inventory_effect_target(item_current, item_following)
                item_nll = F.cross_entropy(
                    inv_pred["item_effect_logits"], item_target, reduction="none"
                ) / math.log(5.0)
            else:
                gate_nll = F.cross_entropy(
                    inv_pred["item_gate_logits"], item_changed.long(), reduction="none"
                ) / math.log(2.0)
                if self.crafter_inventory_value_mode == "categorical_absolute":
                    value_target = item_following
                    value_normalizer = math.log(INVENTORY_VALUES)
                else:
                    from modelBased.world_model.crafter_dynamics import (
                        categorical_inventory_delta_target, ITEM_DELTA_CLASSES,
                    )
                    value_target = categorical_inventory_delta_target(
                        item_current, item_following
                    )
                    value_normalizer = math.log(ITEM_DELTA_CLASSES)
                value_nll = F.cross_entropy(
                    inv_pred["item_value_logits"], value_target, reduction="none"
                ) / value_normalizer
                item_nll = (gate_nll + value_nll) * 0.5
            loss_sum = loss_sum + item_nll[item_changed].sum()
            changed_count = changed_count + item_changed.sum().to(dtype=changed_count.dtype)
        return loss_sum, changed_count

    def _crafter_focal_diagnostics(self, obs_pred, obs_next, obs, inv, inv_next, inv_pred):
        """Count-weighted changed-only focal diagnostics for Crafter validation."""
        zero = obs_pred.new_zeros((), dtype=torch.float32)
        result = {
            "layout_object_changed_loss_sum": zero, "layout_object_changed_count": zero,
            "layout_object_false_set_sum": zero, "layout_object_keep_count": zero,
            "layout_direction_changed_loss_sum": zero, "layout_direction_changed_count": zero,
            "layout_direction_false_set_sum": zero, "layout_direction_keep_count": zero,
            "inventory_changed_loss_sum": zero, "inventory_changed_count": zero,
            "inventory_false_set_sum": zero, "inventory_keep_count": zero,
            "inventory_focal_available": zero,
        }
        if self.env_type != "crafter":
            return result
        for spec in self.observation_schema:
            if str(spec.get("distribution")) != "categorical_effect" or str(spec.get("target_source", "observation")) == "inventory":
                continue
            index = int(spec["target_index"])
            start, stop = map(int, spec["prediction_slice"])
            effects = categorical_effect_target(obs[:, index], obs_next[:, index])
            per_cell = balanced_categorical_effect_loss(
                obs_pred[:, start:stop], effects, reduction="none",
                label_smoothing=float(spec.get("label_smoothing", 0.0)), focal_gamma=1.0,
            )
            prefix = "layout_object" if index == 0 else "layout_direction"
            changed, keep = effects.ne(0), effects.eq(0)
            result[f"{prefix}_changed_loss_sum"] = per_cell[changed].sum()
            result[f"{prefix}_changed_count"] = changed.sum().float()
            result[f"{prefix}_false_set_sum"] = obs_pred[:, start:stop].argmax(dim=1)[keep].ne(0).sum().float()
            result[f"{prefix}_keep_count"] = keep.sum().float()
        if (inv is None or inv_next is None or not isinstance(inv_pred, dict)
                or self.crafter_inventory_output_mode != "categorical_effect"):
            return result
        current = inv.long()[:, SURVIVAL_SLOTS:]
        following = inv_next.long()[:, SURVIVAL_SLOTS:]
        effects = categorical_inventory_effect_target(current, following)
        logits = inv_pred["item_effect_logits"]
        per_slot = balanced_categorical_effect_loss(logits, effects, reduction="none", focal_gamma=1.0)
        changed, keep = effects.ne(0), effects.eq(0)
        result.update({
            "inventory_changed_loss_sum": per_slot[changed].sum(),
            "inventory_changed_count": changed.sum().float(),
            "inventory_false_set_sum": logits.argmax(dim=1)[keep].ne(0).sum().float(),
            "inventory_keep_count": keep.sum().float(),
            "inventory_focal_available": zero.new_tensor(1.0),
        })
        return result

    def _crafter_inventory_diagnostics(self, inv, inv_next, inv_pred):
        """Return count-based, natural-distribution Crafter inventory diagnostics.

        These are validation diagnostics only.  In particular, they do not use
        the balanced training reduction, so recall and false-positive rates
        describe the unchanged target transition distribution.
        """
        if (
            self.env_type != "crafter" or inv is None or inv_next is None
            or not isinstance(inv_pred, dict)
        ):
            return None
        validate_discrete_inventory(inv, "current")
        validate_discrete_inventory(inv_next, "next")
        current, following = inv.long(), inv_next.long()
        inventory_output_mode = getattr(
            self, "crafter_inventory_output_mode", "categorical_gate"
        )
        decoded = decode_crafter_inventory(
            current, inv_pred, predict_survival=self.predict_survival,
            output_mode=inventory_output_mode,
            value_mode=getattr(
                self, "crafter_inventory_value_mode", "categorical_absolute"
            ),
        )
        slot_changed = current.ne(following)
        item_changed = slot_changed[:, SURVIVAL_SLOTS:]
        if inventory_output_mode == "categorical_effect":
            item_effect = inv_pred["item_effect_logits"].argmax(dim=1)
            # Residual-enabled diagnostics must report the final projected
            # event outcome, not the raw independent slot argmax.
            item_value = decoded[:, SURVIVAL_SLOTS:] if "item_event_scores" in inv_pred else decode_crafter_item_effects(
                current[:, SURVIVAL_SLOTS:], inv_pred["item_effect_logits"]
            )
            gate_changed = item_value.ne(current[:, SURVIVAL_SLOTS:])
            item_target = categorical_inventory_effect_target(
                current[:, SURVIVAL_SLOTS:], following[:, SURVIVAL_SLOTS:]
            )
            effect_nll = F.cross_entropy(
                inv_pred["item_effect_logits"], item_target, reduction="none"
            ) / math.log(5.0)
            gate_nll = effect_nll
            value_nll = effect_nll
        else:
            gate_changed = inv_pred["item_gate_logits"].argmax(dim=1).bool()
            item_value = decode_crafter_item_values(
                current[:, SURVIVAL_SLOTS:], inv_pred["item_value_logits"],
                value_mode=getattr(
                    self, "crafter_inventory_value_mode", "categorical_absolute"
                ),
            )
            gate_nll = F.cross_entropy(
                inv_pred["item_gate_logits"], item_changed.long(), reduction="none"
            ) / math.log(2.0)
            if getattr(self, "crafter_inventory_value_mode", "categorical_absolute") == "categorical_delta":
                from modelBased.world_model.crafter_dynamics import categorical_inventory_delta_target, ITEM_DELTA_CLASSES
                value_target = categorical_inventory_delta_target(
                    current[:, SURVIVAL_SLOTS:], following[:, SURVIVAL_SLOTS:]
                )
                value_norm = math.log(ITEM_DELTA_CLASSES)
            else:
                value_target, value_norm = following[:, SURVIVAL_SLOTS:], math.log(INVENTORY_VALUES)
            value_nll = F.cross_entropy(
                inv_pred["item_value_logits"], value_target, reduction="none"
            ) / value_norm
        zero = current.new_zeros((), dtype=torch.float32)

        core_decoded = decoded if self.predict_survival else decoded[:, SURVIVAL_SLOTS:]
        core_following = following if self.predict_survival else following[:, SURVIVAL_SLOTS:]
        core_changed = slot_changed if self.predict_survival else item_changed
        diagnostics = {
            # With survival enabled this is exactly the legacy all-16-slot
            # metric. With it disabled, the same stable metric name denotes
            # the 12 WM-owned item slots; the explicit flag below disambiguates
            # CSV rows from the two modes.
            "inventory_correct": core_decoded.eq(core_following).sum().float(),
            "inventory_total": core_following.new_tensor(
                float(core_following.numel()), dtype=torch.float32
            ),
            "inventory_changed_correct": (
                core_decoded.eq(core_following) & core_changed
            ).sum().float(),
            "inventory_changed_count": core_changed.sum().float(),
            "survival_changed_correct": (
                (decoded[:, :SURVIVAL_SLOTS].eq(following[:, :SURVIVAL_SLOTS])
                 & slot_changed[:, :SURVIVAL_SLOTS]).sum().float()
                if self.predict_survival else zero
            ),
            "survival_changed_count": (
                slot_changed[:, :SURVIVAL_SLOTS].sum().float()
                if self.predict_survival else zero
            ),
            "survival_prediction_enabled": following.new_tensor(
                float(self.predict_survival), dtype=torch.float32
            ),
            "item_value_changed_correct": (
                item_value.eq(following[:, SURVIVAL_SLOTS:]) & item_changed
            ).sum().float(),
            "item_value_changed_count": item_changed.sum().float(),
            "inventory_slot_changed_counts": slot_changed.sum(dim=0).float(),
            "_zero": zero,
        }
        metric_prefix = "inventory_effect" if inventory_output_mode == "categorical_effect" else "inventory_gate"
        diagnostics.update({
            f"{metric_prefix}_true_positive": (gate_changed & item_changed).sum().float(),
            f"{metric_prefix}_change_count": item_changed.sum().float(),
            f"{metric_prefix}_false_positive": (gate_changed & ~item_changed).sum().float(),
            f"{metric_prefix}_keep_count": (~item_changed).sum().float(),
            f"item_slot_{metric_prefix}_true_positive": (gate_changed & item_changed).sum(dim=0).float(),
            f"item_slot_{metric_prefix}_change_count": item_changed.sum(dim=0).float(),
            f"item_slot_{metric_prefix}_false_positive": (gate_changed & ~item_changed).sum(dim=0).float(),
            f"item_slot_{metric_prefix}_keep_count": (~item_changed).sum(dim=0).float(),
            f"item_slot_{metric_prefix}_nll_sum": (gate_nll * item_changed).sum(dim=0),
            f"item_slot_{metric_prefix}_nll_count": item_changed.sum(dim=0).float(),
            f"item_{metric_prefix}_changed_nll_sum": value_nll[item_changed].sum(),
            f"item_{metric_prefix}_changed_nll_count": item_changed.sum().float(),
            f"{metric_prefix}_row_exact_correct": item_value.eq(following[:, SURVIVAL_SLOTS:]).all(dim=1).sum().float(),
            f"{metric_prefix}_row_exact_total": zero.new_tensor(float(following.shape[0])),
        })
        if "item_event_scores" in inv_pred:
            event_target = categorical_crafter_inventory_event_target(
                current[:, SURVIVAL_SLOTS:], following[:, SURVIVAL_SLOTS:],
                codebook=inv_pred["item_event_codebook"],
            )
            event_nll = F.cross_entropy(
                inv_pred["item_event_scores"], event_target, reduction="none"
            ) / math.log(float(inv_pred["item_event_scores"].shape[1]))
            base_event = inv_pred["item_event_base_scores"].argmax(dim=1)
            hard_rows = event_target.ne(0) | (event_target.eq(0) & base_event.ne(0))
            diagnostics.update({
                "inventory_event_residual_nll_sum": event_nll.sum(),
                "inventory_event_residual_nll_count": zero.new_tensor(float(event_nll.numel())),
                "inventory_event_residual_hard_rows": hard_rows.sum().float(),
            })
        return diagnostics


    def configure_optimizers(self):
        # Use a higher learning rate for the classification head so it adapts
        # faster to new environment tiles, while keeping the base model stable.
        head_params = []
        base_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # `fc` is the output head that predicts tile classes.
            if (
                'model.fc.' in name
                or 'model.inv_head.' in name
                or 'model.outcome_head.' in name
                or 'model.crafter_pose_head.' in name
            ):
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
                "monitor": 'val/observation_loss',
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
        if self.env_type in ('with_obj', 'minigrid'):
            info = batch.get('info', None)
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
        outcome_target = self._stochastic_outcome_target(
            info, obs.shape[0], obs.device
        ) if self.stochastic_latent_enabled else None
        latent_target = self._optional_action_failed_target(info, obs.shape[0], obs.device) \
            if self.stochastic_latent_v2_enabled else None
        latent_prediction = None
        pose_logits = None
        if self.stochastic_latent_v2_enabled:
            latent_prediction = self.forward_distribution(
                obs, act, info, inv=inv, next_state=obs_next,
                next_inventory=inv_next, latent_labels=latent_target,
            )
            obs_pred = latent_prediction["state_logits"]
            attentionWeight = latent_prediction["attention_weights"]
            aux_pred = latent_prediction["inventory_logits"]
            outcome_logits = None
        elif self.stochastic_latent_enabled:
            stochastic_prediction = self.forward_stochastic(obs, act, info, inv=inv)
            obs_pred = stochastic_prediction["state_logits"]
            attentionWeight = stochastic_prediction["attention_weights"]
            aux_pred = stochastic_prediction["inventory_logits"]
            outcome_logits = stochastic_prediction["outcome_logits"]
        elif self.crafter_pose_enabled:
            obs_pred, attentionWeight, aux_pred, pose_logits = self.forward_pose(
                obs, act, info, inv=inv
            )
            outcome_logits = None
        else:
            obs_pred, attentionWeight, aux_pred = self(obs, act, info, inv=inv)
            outcome_logits = None

        # Restrict Crafter visualization to the final epoch to reduce overhead.
        is_last_epoch = False
        try:
            is_last_epoch = (self.current_epoch == self.trainer.max_epochs - 1)
        except:
            pass
            
        if self.visualizationFlag and (not self.is_bipedal) and is_last_epoch and (self.step_counter % self.visualize_every == 0):
            if self.env_type == 'crafter':
                visualize_crafter_wm(obs, obs_next, obs_pred, int(act[0].item()), self.step_counter, 
                                     save_dir=str(utils.WM_VISUALIZATIONS_PATH / "world_model" / "train"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_pos[0],
                                     inv=inv[0].cpu().numpy() if inv is not None else None,
                                     inv_next=inv_next[0].cpu().numpy() if inv_next is not None else None)
            else:
                # Fallback to legacy visualize_data for MiniGrid (requires whole map)
                # Note: this part needs whole map, which we have in 'batch'
                self.visual_func.visualize_data(batch['obs'], batch['obs_next'], act, obs, obs_next, info, self.step_counter, agent_pos)

        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        
        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # v1 predicts only the execute branch.  v2 conditions the decoder on
        # its outcome latent, hence both successful and failed transitions are
        # legitimate reconstruction targets.
        success_mask = outcome_target.eq(0) if outcome_target is not None else None
        if success_mask is None or bool(success_mask.any()):
            loss_mask = success_mask if success_mask is not None else slice(None)
            observation_loss, field_losses = self.observation_loss(
                obs_pred[loss_mask], obs_next[loss_mask], obs_prev=obs[loss_mask],
                aux_pred=aux_pred[loss_mask] if isinstance(aux_pred, torch.Tensor) else aux_pred,
                inv=inv[loss_mask] if inv is not None else None,
                inv_next=inv_next[loss_mask] if inv_next is not None else None,
            )
        else:
            # Retain a gradient-bearing zero for all-failure batches.
            observation_loss = obs_pred.sum() * 0.0
            field_losses = {}

        outcome_loss = (
            F.cross_entropy(outcome_logits, outcome_target)
            if outcome_logits is not None else observation_loss.new_zeros(())
        )
        if latent_prediction is not None:
            posterior_logits = latent_prediction["posterior_logits"]
            if posterior_logits is None:
                raise RuntimeError("latent_v2 training requires a posterior")
            latent_kl = self._categorical_kl(
                posterior_logits, latent_prediction["prior_logits"]
            )
            if latent_target is not None:
                latent_supervision = (
                    F.cross_entropy(latent_prediction["prior_logits"][:, 0], latent_target)
                    + F.cross_entropy(posterior_logits[:, 0], latent_target)
                ) * 0.5
            else:
                latent_supervision = observation_loss.new_zeros(())
        else:
            latent_kl = observation_loss.new_zeros(())
            latent_supervision = observation_loss.new_zeros(())
        if pose_logits is not None:
            pose_loss, pose_metrics = self._crafter_pose_loss_and_metrics(
                pose_logits, obs, obs_next
            )
        else:
            pose_loss = observation_loss.new_zeros(())
            pose_metrics = None

        # Preserve field-level values only in the local Bipedal diagnostics CSV.
        if self.is_bipedal and hasattr(self.model, "bipedal_token_specs"):
            for token_name, token_indices in self.model.bipedal_token_specs:
                self.train_token_loss_accumulator[token_name].append(
                    float(field_losses[token_name].detach().cpu())
                )
                if token_name in getattr(self.model, "contact_token_names", set()):
                    target = (obs[:, token_indices] + obs_next[:, token_indices]).clamp(0.0, 1.0)
                    logits = aux_pred["contact_logits"][token_name]
                    accuracy = ((torch.sigmoid(logits) >= 0.5).float() == target).float().mean()
                    self.train_token_acc_accumulator[token_name].append(float(accuracy.detach().cpu()))

        if self.ewc_enabled:
            ewc_raw = self.ewc_loss()
            ewc_weighted = self.lambda_ewc * ewc_raw
            optimization_objective = (
                observation_loss
                + self.stochastic_latent_loss_weight * outcome_loss
                + self.latent_kl_weight * latent_kl
                + self.latent_supervision_weight * latent_supervision
                + pose_loss
                + ewc_weighted
            )
        else:
            optimization_objective = (
                observation_loss + self.stochastic_latent_loss_weight * outcome_loss
                + self.latent_kl_weight * latent_kl
                + self.latent_supervision_weight * latent_supervision
                + pose_loss
            )

        # One public training-loss curve for every domain.
        self.log(
            "train/observation_loss",
            observation_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        if pose_metrics is not None:
            self.log("train/pose_loss", pose_loss, on_step=True, on_epoch=False)
            self.log("train/pose_joint_accuracy", pose_metrics["joint_accuracy"], on_step=True, on_epoch=False)
            self.log("train/pose_position_accuracy", pose_metrics["position_accuracy"], on_step=True, on_epoch=False)
            self.log("train/pose_direction_accuracy", pose_metrics["direction_accuracy"], on_step=True, on_epoch=False)
        if self.ewc_enabled:
            # Diagnostics only: they are detached and cannot change the
            # optimization objective. The ratio is relative to the base WM
            # observation loss, with an epsilon for near-zero losses.
            ewc_ratio = ewc_weighted.detach() / observation_loss.detach().abs().clamp_min(1e-8)
            self.log("train/ewc_raw", ewc_raw.detach(), on_step=True, on_epoch=False)
            self.log("train/ewc_weighted", ewc_weighted.detach(), on_step=True, on_epoch=False)
            self.log("train/ewc_to_wm_ratio", ewc_ratio, on_step=True, on_epoch=False)
            # Epoch means are the experiment-level diagnostics.  The existing
            # step values above remain available for last-batch debugging.
            self.log("train/ewc_raw_epoch", ewc_raw.detach(), on_step=False, on_epoch=True)
            self.log("train/ewc_weighted_epoch", ewc_weighted.detach(), on_step=False, on_epoch=True)
            self.log("train/ewc_to_wm_ratio_epoch", ewc_ratio, on_step=False, on_epoch=True)
        if outcome_logits is not None:
            outcome_probs = torch.softmax(outcome_logits, dim=-1)
            self.log("train/outcome_loss", outcome_loss, on_step=True, on_epoch=False)
            self.log("train/failure_probability", outcome_probs[:, 1].mean(), on_step=True, on_epoch=False)
            self.log("train/failure_rate", outcome_target.float().mean(), on_step=True, on_epoch=False)
        if latent_prediction is not None:
            self.log("train/latent_kl", latent_kl, on_step=True, on_epoch=False)
            self.log("train/latent_supervision", latent_supervision, on_step=True, on_epoch=False)
        return optimization_objective

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
        outcome_target = self._stochastic_outcome_target(
            info, obs.shape[0], obs.device
        ) if self.stochastic_latent_enabled else None
        latent_target = self._optional_action_failed_target(info, obs.shape[0], obs.device) \
            if self.stochastic_latent_v2_enabled else None
        latent_prediction = None
        pose_logits = None
        if self.stochastic_latent_v2_enabled:
            latent_prediction = self.forward_distribution(
                obs, act, info, inv=inv, next_state=obs_next,
                next_inventory=inv_next, latent_labels=latent_target, sample_mode="mode",
            )
            obs_pred = latent_prediction["state_logits"]
            attention_weight = latent_prediction["attention_weights"]
            inv_pred = latent_prediction["inventory_logits"]
            outcome_logits = None
        elif self.stochastic_latent_enabled:
            stochastic_prediction = self.forward_stochastic(obs, act, info, inv=inv)
            obs_pred = stochastic_prediction["state_logits"]
            attention_weight = stochastic_prediction["attention_weights"]
            inv_pred = stochastic_prediction["inventory_logits"]
            outcome_logits = stochastic_prediction["outcome_logits"]
        elif self.crafter_pose_enabled:
            obs_pred, attention_weight, inv_pred, pose_logits = self.forward_pose(
                obs, act, info, inv=inv
            )
            outcome_logits = None
        else:
            obs_pred, attention_weight, inv_pred = self(obs, act, info, inv=inv)
            outcome_logits = None
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
            loss_map = self.compute_cell_loss(obs_pred, obs_next, current=obs)
            batch_size = loss_map.shape[0]
            
            # [DEBUG] Check if we are actually getting coordinates and values
            if batch_idx == 0 and self.env_type == 'minigrid':
                print(f"[WM-Debug] LossMap Mean: {loss_map.mean():.6f}, AgentPos[0]: {agent_position[0]}")
                
            for i in range(batch_size):
                agent_pos = agent_position[i].tolist()  # (y, x)
                self.accumulate_loss(loss_map[i], agent_pos)
            if self.env_type == 'crafter' and inv_pred is not None and inv_next is not None:
                # Store per-slot categorical error rates for diagnostics.
                inv_reconstructed = decode_crafter_inventory(
                    inv, inv_pred, predict_survival=self.predict_survival,
                    output_mode=self.crafter_inventory_output_mode,
                    value_mode=self.crafter_inventory_value_mode,
                )
                inv_slot_error = inv_reconstructed.ne(inv_next.long()).float().mean(dim=0)
                self.inventory_loss_accumulator.append(inv_slot_error.detach().cpu())
  
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
                                     save_dir=str(utils.WM_VISUALIZATIONS_PATH / "world_model" / "val"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_position[0],
                                     inv=batch['inv'][0].cpu().numpy() if 'inv' in batch else None,
                                     inv_next=batch['inv_next'][0].cpu().numpy() if 'inv_next' in batch else None)

        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # Validation uses exactly the same observation objective as training.
        success_mask = outcome_target.eq(0) if outcome_target is not None else None
        if success_mask is None or bool(success_mask.any()):
            loss_mask = success_mask if success_mask is not None else slice(None)
            transition_loss, field_values = self.observation_loss(
                obs_pred[loss_mask], obs_next[loss_mask], obs_prev=obs[loss_mask],
                aux_pred=inv_pred[loss_mask] if isinstance(inv_pred, torch.Tensor) else inv_pred,
                inv=inv[loss_mask] if inv is not None else None,
                inv_next=inv_next[loss_mask] if inv_next is not None else None,
            )
        else:
            transition_loss = obs_pred.sum() * 0.0
            field_values = {}
        outcome_loss = (
            F.cross_entropy(outcome_logits, outcome_target)
            if outcome_logits is not None else transition_loss.new_zeros(())
        )
        if latent_prediction is not None:
            latent_kl = self._categorical_kl(
                latent_prediction["posterior_logits"], latent_prediction["prior_logits"]
            )
            latent_supervision = (
                (F.cross_entropy(latent_prediction["prior_logits"][:, 0], latent_target)
                 + F.cross_entropy(latent_prediction["posterior_logits"][:, 0], latent_target)) * 0.5
                if latent_target is not None else transition_loss.new_zeros(())
            )
        else:
            latent_kl = transition_loss.new_zeros(())
            latent_supervision = transition_loss.new_zeros(())
        if pose_logits is not None:
            pose_loss, pose_metrics = self._crafter_pose_loss_and_metrics(
                pose_logits, obs, obs_next
            )
            self._val_crafter_pose_diagnostics.append({
                "pose_loss": pose_loss.detach(),
                **{name: value.detach() for name, value in pose_metrics.items()},
            })
        else:
            pose_loss = transition_loss.new_zeros(())
        base_loss_val = (
            transition_loss + self.stochastic_latent_loss_weight * outcome_loss
            + self.latent_kl_weight * latent_kl
            + self.latent_supervision_weight * latent_supervision
        )
        loss_val = base_loss_val + pose_loss

        if success_mask is None or bool(success_mask.any()):
            natural_loss, natural_field_values = self.observation_loss(
                obs_pred[loss_mask], obs_next[loss_mask], obs_prev=obs[loss_mask],
                aux_pred=inv_pred[loss_mask] if isinstance(inv_pred, torch.Tensor) else inv_pred,
                inv=inv[loss_mask] if inv is not None else None,
                inv_next=inv_next[loss_mask] if inv_next is not None else None,
                focal_gamma=0.0,
            )
        else:
            natural_loss, natural_field_values = transition_loss, {}
        self._val_natural_outputs.append(natural_loss.detach())
        self._val_field_outputs.append({
            name: value.detach() for name, value in natural_field_values.items()
        })
        if self.env_type == "minigrid":
            with torch.no_grad():
                # The spatial/inventory heads represent the execute branch;
                # do not score their deterministic decode against failure
                # transitions, whose exact successor is the current state.
                diagnostic_mask = success_mask if success_mask is not None else slice(None)
                diagnostic_obs_pred = obs_pred[diagnostic_mask]
                diagnostic_obs_next = obs_next[diagnostic_mask]
                diagnostic_obs = obs[diagnostic_mask]
                diagnostic_inv = inv[diagnostic_mask] if inv is not None else None
                diagnostic_inv_next = inv_next[diagnostic_mask] if inv_next is not None else None
                diagnostic_inv_pred = (
                    inv_pred[diagnostic_mask]
                    if isinstance(inv_pred, torch.Tensor) else inv_pred
                )
                changed_focal, false_set, changed_count = self._minigrid_effect_diagnostics(
                    diagnostic_obs_pred, diagnostic_obs_next, diagnostic_obs,
                    diagnostic_inv, diagnostic_inv_next, diagnostic_inv_pred
                )
                decoded, decoded_inv, diagnostics = decode_minigrid_transition(
                    diagnostic_obs_pred,
                    diagnostic_obs,
                    diagnostic_inv_pred,
                    diagnostic_inv,
                    mode=self.minigrid_transition_mode,
                    constrain_agent=False,
                )
                target = diagnostic_obs_next.long()
                changed = diagnostic_obs.ne(target)
                self._val_minigrid_diagnostics.append({
                    "decoded_accuracy": decoded.eq(target).float().mean(),
                    "changed_accuracy": decoded.eq(target)[changed].float().mean()
                    if changed.any() else decoded.new_tensor(0.0, dtype=torch.float32),
                    "agent_missing_rate": diagnostics["raw_agent_count"].eq(0).float().mean(),
                    "agent_duplicate_rate": diagnostics["raw_agent_count"].gt(1).float().mean(),
                    "inventory_accuracy": decoded_inv.eq(inv_next.long()).float().mean()
                    if decoded_inv is not None and inv_next is not None
                    else decoded.new_tensor(0.0, dtype=torch.float32),
                })
                self._val_minigrid_effect_diagnostics.append({
                    "changed_focal_loss": changed_focal.detach(),
                    "false_set_rate": false_set.detach(),
                    "changed_count": changed_count.detach(),
                })
        elif self.env_type == "crafter":
            with torch.no_grad():
                changed_sum, changed_count = self._crafter_changed_diagnostics(
                    obs_pred, obs_next, obs, inv, inv_next, inv_pred
                )
                if self._val_crafter_changed_loss_sum is None:
                    self._val_crafter_changed_loss_sum = changed_sum.detach()
                    self._val_crafter_changed_count = changed_count.detach()
                else:
                    self._val_crafter_changed_loss_sum = (
                        self._val_crafter_changed_loss_sum + changed_sum.detach()
                    )
                    self._val_crafter_changed_count = (
                        self._val_crafter_changed_count + changed_count.detach()
                    )
                focal_diagnostics = self._crafter_focal_diagnostics(
                    obs_pred, obs_next, obs, inv, inv_next, inv_pred
                )
                if self._val_crafter_focal_diagnostics is None:
                    self._val_crafter_focal_diagnostics = {
                        name: value.detach() for name, value in focal_diagnostics.items()
                    }
                else:
                    for name, value in focal_diagnostics.items():
                        self._val_crafter_focal_diagnostics[name] = (
                            self._val_crafter_focal_diagnostics[name] + value.detach()
                        )
                inventory_diagnostics = self._crafter_inventory_diagnostics(
                    inv, inv_next, inv_pred
                )
                if inventory_diagnostics is not None:
                    if self._val_crafter_inventory_diagnostics is None:
                        self._val_crafter_inventory_diagnostics = {
                            name: value.detach()
                            for name, value in inventory_diagnostics.items()
                        }
                    else:
                        for name, value in inventory_diagnostics.items():
                            self._val_crafter_inventory_diagnostics[name] = (
                                self._val_crafter_inventory_diagnostics[name] + value.detach()
                            )
        if outcome_logits is not None:
            outcome_probs = torch.softmax(outcome_logits, dim=-1)
            self._val_stochastic_outcomes.append({
                "outcome_loss": outcome_loss.detach(),
                "failure_probability": outcome_probs[:, 1].mean().detach(),
                "failure_rate": outcome_target.float().mean().detach(),
                "outcome_accuracy": outcome_logits.argmax(dim=-1).eq(outcome_target).float().mean().detach(),
            })
        if latent_prediction is not None:
            self._val_stochastic_outcomes.append({
                "latent_kl": latent_kl.detach(),
                "latent_supervision": latent_supervision.detach(),
            })

        # Preserve the historical checkpoint metric: pose is an isolated
        # auxiliary readout, so it must not alter WM checkpoint selection.
        self._val_step_outputs.append(base_loss_val.detach())

        return {
            "loss_wm_val": loss_val,             
        }

    def on_validation_epoch_start(self):
        self._val_step_outputs = []
        self._val_natural_outputs = []
        self._val_field_outputs = []
        self._val_minigrid_diagnostics = []
        self._val_minigrid_effect_diagnostics = []
        self._val_stochastic_outcomes = []
        self._val_crafter_changed_loss_sum = None
        self._val_crafter_changed_count = None
        self._val_crafter_focal_diagnostics = None
        self._val_crafter_inventory_diagnostics = None
        self._val_crafter_pose_diagnostics = []
        if getattr(self.hparams, "keep_cell_loss", False):
            self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
            self.loss_map_result = None
            self.coverage_map_result = None
            self.inventory_loss_accumulator = []
            if getattr(self, "is_bipedal", False):
                self.bipedal_semantic_acc = []

    def on_validation_epoch_end(self):
        if getattr(self.hparams, "keep_cell_loss", False) and not getattr(self, 'is_bipedal', False):
            self.loss_map_result, self.coverage_map_result = (
                self.average_loss_and_coverage_maps(self.loss_accumulator)
            )
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
        if getattr(self.hparams, "keep_cell_loss", False) and self.env_type == 'crafter':
            if len(self.inventory_loss_accumulator) > 0:
                stacked = torch.stack(self.inventory_loss_accumulator, dim=0)  # [num_batches, 16]
                self.inventory_loss_vector_result = stacked.mean(dim=0).numpy().astype(np.float32)
            else:
                self.inventory_loss_vector_result = np.zeros(16, dtype=np.float32)
        if getattr(self.hparams, "keep_cell_loss", False):
            self.inventory_loss_accumulator = []
            # Clear cell-level history to avoid memory growth across repeated validations.
            self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
        
        if len(self._val_step_outputs) > 0:
            avg_loss = torch.stack(self._val_step_outputs).mean()
        else:
            avg_loss = torch.tensor(0.0, device=self.device)

        self.log("val/observation_loss", avg_loss)
        if self._val_natural_outputs:
            self.log("val/natural_ce", torch.stack(self._val_natural_outputs).mean())
        if self._val_field_outputs:
            field_means = {}
            field_names = sorted({name for row in self._val_field_outputs for name in row})
            for name in field_names:
                values = [row[name] for row in self._val_field_outputs if name in row]
                if values:
                    field_means[name] = torch.stack(values).mean()
                    self.log(f"val/{name}_nll", field_means[name])
        else:
            field_means = {}
        if self._val_minigrid_diagnostics:
            for name in self._val_minigrid_diagnostics[0]:
                self.log(
                    f"val/{name}",
                    torch.stack([row[name] for row in self._val_minigrid_diagnostics]).mean(),
                )
        if self._val_minigrid_effect_diagnostics:
            for name in ("changed_focal_loss", "false_set_rate", "changed_count"):
                self.log(
                    f"val/{name}",
                    torch.stack([
                        row[name] for row in self._val_minigrid_effect_diagnostics
                    ]).mean(),
                )
        if self.env_type == "crafter" and self._val_crafter_changed_loss_sum is not None:
            count = self._val_crafter_changed_count
            changed_nll = (
                self._val_crafter_changed_loss_sum / count
                if bool(count > 0)
                else self._val_crafter_changed_loss_sum
            )
            self.log("val/changed_nll", changed_nll)
            self.log("val/changed_count", count)
        if self.env_type == "crafter" and self._val_crafter_focal_diagnostics is not None:
            diagnostics = self._val_crafter_focal_diagnostics
            zero = diagnostics["layout_object_changed_count"].new_zeros(())
            def _ratio(numerator, denominator):
                return numerator / denominator if bool(denominator > 0) else zero
            layout_fields = ("layout_object", "layout_direction")
            layout_changed = [
                _ratio(diagnostics[f"{name}_changed_loss_sum"], diagnostics[f"{name}_changed_count"])
                for name in layout_fields if bool(diagnostics[f"{name}_changed_count"] > 0)
            ]
            layout_false_set = [
                _ratio(diagnostics[f"{name}_false_set_sum"], diagnostics[f"{name}_keep_count"])
                for name in layout_fields if bool(diagnostics[f"{name}_keep_count"] > 0)
            ]
            nan = zero.new_full((), float("nan"))
            layout_focal = torch.stack(layout_changed).mean() if layout_changed else nan
            layout_fpr = torch.stack(layout_false_set).mean() if layout_false_set else nan
            self.log("val/layout_changed_focal_loss", layout_focal)
            self.log("val/layout_false_set_rate", layout_fpr)
            self.log("val/layout_changed_count", sum(diagnostics[f"{name}_changed_count"] for name in layout_fields))
            groups = [layout_focal] if layout_changed else []
            if bool(diagnostics["inventory_focal_available"] > 0):
                inventory_focal = (_ratio(diagnostics["inventory_changed_loss_sum"], diagnostics["inventory_changed_count"])
                                   if bool(diagnostics["inventory_changed_count"] > 0) else nan)
                inventory_fpr = (_ratio(diagnostics["inventory_false_set_sum"], diagnostics["inventory_keep_count"])
                                 if bool(diagnostics["inventory_keep_count"] > 0) else nan)
                self.log("val/inventory_changed_focal_loss", inventory_focal)
                self.log("val/inventory_false_set_rate", inventory_fpr)
                self.log("val/inventory_changed_count", diagnostics["inventory_changed_count"])
                if bool(diagnostics["inventory_changed_count"] > 0):
                    groups.append(inventory_focal)
            self.log("val/changed_focal_loss", torch.stack(groups).mean() if groups else nan)
        if self.env_type == "crafter" and self._val_crafter_inventory_diagnostics is not None:
            diagnostics = self._val_crafter_inventory_diagnostics
            zero = diagnostics["_zero"]
            inventory_metric = (
                "inventory_effect" if self.crafter_inventory_output_mode == "categorical_effect"
                else "inventory_gate"
            )
            def _ratio(numerator, denominator):
                return numerator / denominator if bool(denominator > 0) else zero
            self.log(
                "val/inventory_overall_accuracy",
                _ratio(diagnostics["inventory_correct"], diagnostics["inventory_total"]),
            )
            self.log(
                f"val/{inventory_metric}_change_recall",
                _ratio(diagnostics[f"{inventory_metric}_true_positive"], diagnostics[f"{inventory_metric}_change_count"]),
            )
            self.log(
                f"val/{inventory_metric}_change_precision",
                _ratio(
                    diagnostics[f"{inventory_metric}_true_positive"],
                    diagnostics[f"{inventory_metric}_true_positive"]
                    + diagnostics[f"{inventory_metric}_false_positive"],
                ),
            )
            self.log(
                f"val/{inventory_metric}_false_positive_rate",
                _ratio(diagnostics[f"{inventory_metric}_false_positive"], diagnostics[f"{inventory_metric}_keep_count"]),
            )
            self.log(
                "val/inventory_changed_accuracy",
                _ratio(diagnostics["inventory_changed_correct"], diagnostics["inventory_changed_count"]),
            )
            self.log(
                "val/survival_changed_accuracy",
                _ratio(diagnostics["survival_changed_correct"], diagnostics["survival_changed_count"]),
            )
            self.log(
                "val/survival_prediction_enabled",
                zero.new_tensor(float(self.predict_survival)),
            )
            self.log(
                f"val/{'item_effect' if self.crafter_inventory_output_mode == 'categorical_effect' else 'item_value'}_accuracy_on_changed",
                _ratio(diagnostics["item_value_changed_correct"], diagnostics["item_value_changed_count"]),
            )
            self.log(
                f"val/{inventory_metric}_row_exact",
                _ratio(diagnostics[f"{inventory_metric}_row_exact_correct"], diagnostics[f"{inventory_metric}_row_exact_total"]),
            )
            for slot, count in enumerate(diagnostics["inventory_slot_changed_counts"]):
                self.log(f"val/inventory_slot_{slot}_changed_count", count)
            recalls = []
            for slot in range(12):
                slot_prefix = f"item_slot_{inventory_metric}"
                changed_count = diagnostics[f"{slot_prefix}_change_count"][slot]
                keep_count = diagnostics[f"{slot_prefix}_keep_count"][slot]
                recall = _ratio(diagnostics[f"{slot_prefix}_true_positive"][slot], changed_count)
                fpr = _ratio(diagnostics[f"{slot_prefix}_false_positive"][slot], keep_count)
                self.log(f"val/inventory_slot_{slot + 4}_{inventory_metric}_recall", recall)
                self.log(f"val/inventory_slot_{slot + 4}_{inventory_metric}_false_positive_rate", fpr)
                if bool(changed_count > 0):
                    recalls.append(recall)
            # A fixed DR holdout uses this alongside natural observation loss
            # for checkpoint selection; it has no gradient path.
            slot_macro_recall = torch.stack(recalls).mean() if recalls else zero
            self.log(f"val/{inventory_metric}_slot_macro_recall", slot_macro_recall)
            slot_nlls = [
                _ratio(diagnostics[f"{slot_prefix}_nll_sum"][slot], diagnostics[f"{slot_prefix}_nll_count"][slot])
                for slot in range(12)
                if bool(diagnostics[f"{slot_prefix}_nll_count"][slot] > 0)
            ]
            slot_macro_gate_nll = torch.stack(slot_nlls).mean() if slot_nlls else zero
            changed_value_nll = _ratio(
                diagnostics[f"item_{inventory_metric}_changed_nll_sum"], diagnostics[f"item_{inventory_metric}_changed_nll_count"]
            )
            self.log(f"val/{inventory_metric}_slot_macro_nll", slot_macro_gate_nll)
            self.log(f"val/item_{inventory_metric}_changed_nll", changed_value_nll)
            if "inventory_event_residual_nll_sum" in diagnostics:
                self.log(
                    "val/inventory_event_residual_nll",
                    _ratio(diagnostics["inventory_event_residual_nll_sum"],
                           diagnostics["inventory_event_residual_nll_count"]),
                )
                self.log(
                    "val/inventory_event_residual_hard_rows",
                    diagnostics["inventory_event_residual_hard_rows"],
                )
            # These field means are the natural (unweighted) observation NLLs.
            object_nll = field_means.get("layout_object_effect", zero)
            direction_nll = field_means.get("layout_direction_effect", zero)
            self.log("val/selection_loss", crafter_selection_loss(
                object_nll, direction_nll, slot_macro_gate_nll, changed_value_nll
            ))
        if self._val_crafter_pose_diagnostics:
            for name in self._val_crafter_pose_diagnostics[0]:
                self.log(
                    f"val/{name}",
                    torch.stack([row[name] for row in self._val_crafter_pose_diagnostics]).mean(),
                )
            pose_average = torch.stack([
                row["pose_loss"] for row in self._val_crafter_pose_diagnostics
            ]).mean()
            self.log("val/total_loss_with_pose", avg_loss + pose_average)
        if self._val_stochastic_outcomes:
            for name in self._val_stochastic_outcomes[0]:
                self.log(
                    f"val/{name}",
                    torch.stack([row[name] for row in self._val_stochastic_outcomes]).mean(),
                )
        self._val_step_outputs = []
        self._val_natural_outputs = []
        self._val_field_outputs = []
        self._val_minigrid_diagnostics = []
        self._val_minigrid_effect_diagnostics = []
        self._val_stochastic_outcomes = []
        self._val_crafter_changed_loss_sum = None
        self._val_crafter_changed_count = None
        self._val_crafter_focal_diagnostics = None
        self._val_crafter_inventory_diagnostics = None
        self._val_crafter_pose_diagnostics = []

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.model, "checkpoint_contract"):
            checkpoint["world_model_contract"] = dict(self.model.checkpoint_contract)
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
        
        if self.env_type not in ('with_obj', 'minigrid'):
             batch['info'] = None

        obs_masked, act, obs_next_masked, info, elements_mask, _, inv, inv_next = self.preprocess_batch(batch, training=False)
        
        obs_pred, _, aux_pred = self(obs_masked, act, info, inv=inv)
        
        if obs_next_masked.dtype != obs_pred.dtype:
            obs_next_masked = obs_next_masked.float()
            
        loss_obs, _ = self.observation_loss(
            obs_pred,
            obs_next_masked,
            obs_prev=obs_masked,
            aux_pred=aux_pred,
            inv=inv,
            inv_next=inv_next,
        )
        return {"loss_obs": loss_obs}

    @torch.no_grad()
    def calc_minigrid_probe_metrics(self, trajectory_data, include_spatial=False):
        """Evaluate held-out MiniGrid loss and optional global spatial error maps."""
        if self.env_type != "minigrid" or not trajectory_data:
            result = {"changed_focal_loss": 0.0}
            if include_spatial:
                result.update({
                    "error_map": np.zeros((self.row, self.col), dtype=np.float32),
                    "coverage_map": np.zeros((self.row, self.col), dtype=np.float32),
                })
            return result

        was_training = self.training
        self.eval()
        try:
            batch = {
                "obs": trajectory_data["obs"].to(self.device),
                "act": trajectory_data["act"].to(self.device),
                "obs_next": trajectory_data["obs_next"].to(self.device),
                "info": trajectory_data.get("info"),
            }
            for name in ("inv", "inv_next"):
                value = trajectory_data.get(name)
                if value is not None:
                    batch[name] = value.to(self.device) if torch.is_tensor(value) else value

            obs, act, obs_next, info, _, agent_positions, inv, inv_next = self.preprocess_batch(batch)
            obs_pred, _, aux_pred = self(obs, act, info, inv=inv)
            changed_focal, _, changed_count = self._minigrid_effect_diagnostics(
                obs_pred, obs_next, obs, inv, inv_next, aux_pred
            )
            result = {
                "changed_focal_loss": (
                    float(changed_focal.item())
                    if float(changed_count.item()) > 0.0
                    else 0.0
                )
            }
            if include_spatial:
                local_loss_maps = self.compute_cell_loss(
                    obs_pred, obs_next, current=obs
                ).detach().cpu()
                loss_accumulator = [
                    [[] for _ in range(self.col)] for _ in range(self.row)
                ]
                half = self.mask_size // 2
                for sample_index, agent_position in enumerate(agent_positions):
                    agent_y, agent_x = map(int, agent_position)
                    for local_y in range(self.mask_size):
                        for local_x in range(self.mask_size):
                            global_y = agent_y + local_y - half
                            global_x = agent_x + local_x - half
                            if 0 <= global_y < self.row and 0 <= global_x < self.col:
                                loss_accumulator[global_y][global_x].append(
                                    float(local_loss_maps[sample_index, local_y, local_x])
                                )
                error_map, coverage_map = self.average_loss_and_coverage_maps(
                    loss_accumulator
                )
                result.update({
                    "error_map": error_map,
                    "coverage_map": coverage_map,
                })
            return result
        finally:
            self.train(was_training)

    @torch.no_grad()
    def calc_minigrid_changed_focal_loss(self, trajectory_data):
        """Evaluate changed-effect focal loss for one collected trajectory."""
        return float(
            self.calc_minigrid_probe_metrics(trajectory_data)["changed_focal_loss"]
        )

    @torch.no_grad()
    def calc_crafter_changed_focal_loss(self, trajectory_data):
        """Changed-only focal loss for one held-out Crafter rollout.

        This mirrors validation: mean non-empty layout fields, then mean the
        layout and inventory groups. A rollout with no scored change is NaN.
        """
        if self.env_type != "crafter" or not trajectory_data:
            return float("nan")
        was_training = self.training
        self.eval()
        try:
            batch = {"obs": trajectory_data["obs"].to(self.device),
                     "act": trajectory_data["act"].to(self.device),
                     "obs_next": trajectory_data["obs_next"].to(self.device),
                     "info": None}
            for name in ("inv", "inv_next"):
                value = trajectory_data.get(name)
                if value is not None:
                    batch[name] = value.to(self.device) if torch.is_tensor(value) else value
            obs, act, obs_next, info, _, _, inv, inv_next = self.preprocess_batch(batch)
            obs_pred, _, inv_pred = self(obs, act, info, inv=inv)
            diagnostics = self._crafter_focal_diagnostics(
                obs_pred, obs_next, obs, inv, inv_next, inv_pred
            )
            layout = []
            for name in ("layout_object", "layout_direction"):
                count = diagnostics[f"{name}_changed_count"]
                if bool(count > 0):
                    layout.append(diagnostics[f"{name}_changed_loss_sum"] / count)
            groups = [torch.stack(layout).mean()] if layout else []
            count = diagnostics["inventory_changed_count"]
            if bool(diagnostics["inventory_focal_available"] > 0) and bool(count > 0):
                groups.append(diagnostics["inventory_changed_loss_sum"] / count)
            return float(torch.stack(groups).mean().item()) if groups else float("nan")
        finally:
            self.train(was_training)

    def on_train_end(self):
        """Save a final visualization at the end of training."""
        if self.visualizationFlag and self.env_type == 'crafter':
            # We don't easily have the last batch here, but we can signal or just rely on the last training_step/val_step
            # For now, we prints a message to confirm.
            print(f"[WM] Training ended. Final visualizations saved in {utils.WM_VISUALIZATIONS_PATH / 'world_model'}/")




   
