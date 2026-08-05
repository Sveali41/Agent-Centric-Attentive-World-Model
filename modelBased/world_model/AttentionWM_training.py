import os
import warnings

import torch
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(ROOTPATH)

from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.data.datamodule import WMRLDataModule, WMRLDataset, WMRLDataset
from modelBased.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import numpy as np
from modelBased.common.utils import TRAINER_PATH
from omegaconf import open_dict


warnings.filterwarnings(
    "ignore",
    message=r".*val_dataloader.*sampler has shuffling enabled.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
)


# === Define Custom Datamodule for Validation ===
# This allows using 100% data for validation without modifying the original library code.
class ValidationDataModule(WMRLDataModule):
    def setup(self, stage=None):
        if self.direct_data is not None:
            loaded = self.direct_data
        else:
            loaded = np.load(self.data_dir, allow_pickle=True)
        
        # Create dataset
        data = WMRLDataset(loaded, self.cfg, self.replay_data)
        
        # Use 100% data for test
        # Create a Subset that covers the full range
        self.data_test = torch.utils.data.Subset(data, range(0, len(data)))
        # Training split is intentionally empty in validation-only mode.
        self.data_train = torch.utils.data.Subset(data, range(0, 0))
        
        print(f"[ValidationDataModule] Used 100% data ({len(self.data_test)} samples) for validation.")

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_test, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            drop_last=False, # Allow small batches
            num_workers=self.cfg.n_cpu,
            pin_memory=True,
            persistent_workers=False
        )

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):
    net = AttentionWorldModel(cfg.attention_model)
    
    # ===== Step 1: Train on Dataset 1 =====
    print(f"\n{'='*60}")
    print(f"[Phase 1] TRAINING on: {cfg.attention_model.data_dir}")
    print(f"{'='*60}\n")
    result = run(cfg, net=net)

    # ===== Step 2: Validate on Dataset 2 (if configured) =====
    val_data_dir = getattr(cfg.attention_model, "validation_data_dir", None)
    if val_data_dir and result["mode"] == "train":
        print(f"\n{'='*60}")
        print(f"[Phase 2] VALIDATING on: {val_data_dir}")
        print(f"{'='*60}\n")
        
        from omegaconf import OmegaConf
        val_cfg = OmegaConf.to_container(cfg, resolve=True)
        val_cfg["attention_model"]["data_dir"] = val_data_dir
        val_cfg["attention_model"]["freeze_weight"] = True
        val_cfg = OmegaConf.create(val_cfg)
        
        val_result = run(val_cfg, net=result["net"])
        val_loss = val_result.get("avg_val_loss", "N/A")
        print(f"\n[Phase 2] Validation loss on dataset 2: {val_loss}")
        if isinstance(val_loss, list) and len(val_loss) > 0 and isinstance(val_loss[0], dict):
            metrics = val_loss[0]
            token_metrics = {
                k: v for k, v in metrics.items()
                if k.startswith("val/token_")
            }
            if token_metrics:
                ordered_keys = sorted(token_metrics.keys())
                summary = ", ".join(f"{k}={float(token_metrics[k]):.6f}" for k in ordered_keys)
                print(f"[Phase 2] Validation token metrics: {summary}")

def compare_params(net, old_params):
    if old_params is None:
        print('old params is None, skip comparison')
        return
    print("------ Comparing old_params to current model params ------")
    for name, param in net.named_parameters():
        if name in old_params:
            diff = (param.detach().cpu() - old_params[name]).abs().max().item()
            print(f"{name:40s} diff = {diff:.8f}")


def run(
    cfg: DictConfig,
    net: AttentionWorldModel = None,
    old_params=None,
    fisher=None,
    layout=None,
    replay_data=None,
    direct_data=None
):
    if net is None:
        from modelBased.world_model.AttentionWM import AttentionWorldModel
        net = AttentionWorldModel(cfg.attention_model)
    
    # Ensure net is on the right device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    print(f'*************************Data set: {cfg.attention_model.data_dir}************************')

    use_wandb = cfg.attention_model.use_wandb
    fisher_beta = float(getattr(cfg.attention_model, "fisher_beta", 0.5))

    # datamodule
    should_use_validation_mode = cfg.attention_model.freeze_weight
    
    if should_use_validation_mode:
        # Validation-only mode uses 100% of the data without a train/val split.
        datamodule = ValidationDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=None)
    elif cfg.attention_model.continue_learning:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=replay_data)
    else:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=None)

    # logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local_Attention_Training", log_model=True, reinit=True)
        # Avoid attaching W&B watch hooks during validation-only execution.
        # Those hooks would outlive the temporary logger instance.
        if not cfg.attention_model.freeze_weight:
            # wandb_logger.experiment.watch(net, log='all', log_freq=1000)
            pass

    # callbacks
    metric_to_monitor = 'avg_val_loss_wm'
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor,
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode="min"
    )
    try:
        tmp_dir = os.path.dirname(cfg.attention_model.model_save_path)
    except Exception as e:
        print("EXCEPTION AT E1:", e)
        
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=tmp_dir,
        filename="att-{epoch:02d}-{avg_val_loss_wm:.5f}",
        verbose=False
    )

    # trainer
    debug_mode = bool(getattr(cfg.attention_model, "debug_mode", False))
    show_progress_bar = bool(getattr(cfg.attention_model, "enable_progress_bar", debug_mode))

    trainer = pl.Trainer(
        precision=32,
        logger=wandb_logger if use_wandb else None,
        max_epochs=cfg.attention_model.n_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=False,
        enable_progress_bar=show_progress_bar,
    )


    result = {
        "mode": None,            
        "net": net,              
        "old_params": None,
        "fisher": None,
        "avg_val_loss": None,
    }

    # consolidation: Load weights if old_params are provided to continue learning
    net.set_consolidation(old_params, fisher, load_weights=(old_params is not None))

    if cfg.attention_model.freeze_weight:
        # ===== validation =====
        avg_val_loss = trainer.validate(net, datamodule)

        result["mode"] = "val"
        result["avg_val_loss"] = avg_val_loss
        return result

    else:
        # ===== training =====
        trainer.fit(net, datamodule)

        # Save the current parameters as the consolidation anchor.
        old_params = net.save_old_params()

        # Estimate the Fisher information matrix.
        fisher_samples = int(getattr(cfg.attention_model, "fisher_samples", 3000))
        scale_factor = cfg.attention_model.scale_factor
        new_fisher = net.compute_fisher(
            datamodule.train_dataloader(),
            samples=fisher_samples,
            scale_factor=scale_factor
        )

        # Merge Fisher estimates with EMA smoothing.
        if fisher is not None:
            fisher = {
                k: (1.0 - fisher_beta) * fisher[k] + fisher_beta * new_fisher[k]
                for k in new_fisher
            }
        else:
            fisher = new_fisher

    # ... (in run)
        # Save the training checkpoint.
        model_pth = cfg.attention_model.model_save_path
        trainer.save_checkpoint(model_pth)
        if use_wandb:
            wandb.save(str(model_pth))
            wandb.save(model_pth)

        result["mode"] = "train"
        result["old_params"] = old_params
        result["fisher"] = fisher
        
        # Capture best validation loss
        best_score = trainer.checkpoint_callback.best_model_score
        result["best_loss"] = best_score.item() if best_score is not None else 0.0
        
        # Capture all final metrics for logging (e.g. ce_loss, inv_loss, ewc_term)
        for k, v in trainer.callback_metrics.items():
             # Strip 'train/' prefix if present for uniform UED logging
             clean_k = k.replace("train/", "")
             result[clean_k] = v.item() if hasattr(v, 'item') else v

        return result


def train_api(
    cfg: DictConfig,
    net: AttentionWorldModel = None,
    old_params=None,
    fisher=None,
    env_layout=None,
    replay_data=None,
    direct_data=None
):
    result = run(
        cfg,
        net=net,
        old_params=old_params,
        fisher=fisher,
        layout=env_layout,
        replay_data=replay_data,
        direct_data=direct_data
    )

    return result, result.get("fisher"), net  # Return 3-tuple for compatibility with older unpacking logic



if __name__ == "__main__":
    print("THIS SCRIPT IS EXECUTING!")
    train()
