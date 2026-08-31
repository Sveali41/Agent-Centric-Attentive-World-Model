import os
import shutil
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.data.datamodule import WMRLDataModule, WMRLDataset, WMRLDataset
from modelBased.common.artifacts import dataset_matches, align_world_model_artifact_path
from modelBased.common.utils import PROJECT_ROOT, WORLD_MODEL_PATH, WM_RESULTS_PATH, get_env
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger
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

        # Validation archives can be large and are evaluated repeatedly during
        # MAC.  Subsample the raw aligned arrays before WMRLDataset performs
        # channel validation, inventory decoding, and sampling-bucket setup.
        # The fixed seed gives every MAC iteration exactly the same holdout.
        max_samples = int(getattr(self.cfg, "max_validation_samples", 0))
        total_samples = len(loaded["a"])
        if 0 < max_samples < total_samples:
            subset_seed = int(getattr(self.cfg, "validation_subset_seed", 0))
            rng = np.random.default_rng(subset_seed)
            selected = rng.choice(
                total_samples,
                size=max_samples,
                replace=False,
            )
            selected.sort()

            keys = loaded.files if hasattr(loaded, "files") else loaded.keys()
            sampled = {}
            for key in keys:
                value = loaded[key]
                # Transition arrays all have N as their leading dimension.
                # Scalar identity metadata must be copied without indexing.
                if getattr(value, "ndim", 0) > 0 and len(value) == total_samples:
                    sampled[key] = value[selected]
                else:
                    sampled[key] = value
            if hasattr(loaded, "close"):
                loaded.close()
            loaded = sampled
            print(
                f"[ValidationDataModule] Fixed raw subset: "
                f"{max_samples}/{total_samples} samples (seed={subset_seed})."
            )
        
        # Create dataset
        data = WMRLDataset(loaded, self.cfg, self.replay_data)
        
        # Use 100% data for test
        # Create a Subset that covers the full range
        self.data_test = torch.utils.data.Subset(data, range(0, len(data)))
        # Training split is intentionally empty in validation-only mode.
        self.data_train = torch.utils.data.Subset(data, range(0, 0))
        
        print(
            f"[ValidationDataModule] Validating on "
            f"{len(self.data_test)} prepared samples."
        )

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
    validation_matches = bool(
        val_data_dir
        and os.path.exists(str(val_data_dir))
        and dataset_matches(val_data_dir, cfg)
    )
    if val_data_dir and result["mode"] == "train" and validation_matches:
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
    elif val_data_dir and result["mode"] == "train":
        print(
            "[Phase 2] Validation dataset missing or identity mismatch; "
            f"skipping validation: {val_data_dir}"
        )

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
    direct_data=None,
    fit_ckpt_path=None,
):
    align_world_model_artifact_path(cfg)
    if net is None:
        from modelBased.world_model.AttentionWM import AttentionWorldModel
        net = AttentionWorldModel(cfg.attention_model)
    
    # Ensure net is on the right device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    print(f'*************************Data set: {cfg.attention_model.data_dir}************************')

    if not dataset_matches(cfg.attention_model.data_dir, cfg):
        allow_legacy = bool(getattr(cfg.attention_model, "allow_legacy_dataset", False))
        from modelBased.common.artifacts import dataset_metadata
        if allow_legacy and dataset_metadata(cfg.attention_model.data_dir) is None:
            print(
                "[WM] WARNING: using a legacy dataset without identity metadata. "
                "Verify task_name/layout_path manually."
            )
        else:
            raise RuntimeError(
                "Dataset identity does not match the selected domain/task/layout. "
                "Recollect data with: python -m modelBased.data.data_collect "
                f"domain={cfg.domain}"
            )

    use_wandb = cfg.attention_model.use_wandb
    ewc_enabled = bool(getattr(cfg.attention_model, "ewc_enabled", False))

    # datamodule
    should_use_validation_mode = cfg.attention_model.freeze_weight
    
    if should_use_validation_mode:
        # Validation-only mode uses 100% of the data without a train/val split.
        datamodule = ValidationDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=None)
    elif bool(getattr(cfg.attention_model, "continue_learning", False)):
        datamodule = WMRLDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=replay_data)
    else:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, data=direct_data, replay_data=None)

    # logger
    logger = None
    if use_wandb:
        wandb_dir = Path(
            str(
                getattr(
                    getattr(cfg, "paths", None),
                    "wandb",
                    WM_RESULTS_PATH.parent / "wandb",
                )
            )
        ).expanduser().resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            project="Local_Attention_Training",
            log_model=True,
            reinit=True,
            save_dir=str(wandb_dir),
        )
        # Avoid attaching W&B watch hooks during validation-only execution.
        # Those hooks would outlive the temporary logger instance.
        if not cfg.attention_model.freeze_weight:
            # logger.experiment.watch(net, log='all', log_freq=1000)
            pass
    else:
        # MAC already owns the experiment-level CSV in
        # the curriculum results directory. Disable Lightning's per-call CSVLogger by
        # default in the MAC config so every WM train/validation invocation
        # does not create a new ``world_model/version_*/metrics.csv``
        # directory. It remains available as an explicit opt-in for
        # standalone WM debugging.
        save_local_metrics = bool(
            getattr(cfg.attention_model, "save_local_metrics", True)
        )
        if save_local_metrics:
            logger = CSVLogger(
                save_dir=str(
                    getattr(cfg.attention_model, "metrics_dir", WM_RESULTS_PATH / "world_model")
                ),
                name="world_model",
            )
        else:
            logger = False

    # callbacks
    metric_to_monitor = 'val/observation_loss'
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor,
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode="min"
    )
    try:
        tmp_dir = os.path.dirname(cfg.attention_model.model_save_path)
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception as e:
        print("EXCEPTION AT E1:", e)
        
    checkpoint_dir = getattr(cfg.attention_model, "checkpoint_dir", None)
    checkpoint_dir = str(checkpoint_dir) if checkpoint_dir else tmp_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=checkpoint_dir,
        # Keep exactly one temporary checkpoint for this run.  The best
        # weights are copied to model_save_path below and the temporary file
        # is removed afterwards.
        filename=f"best-{str(cfg.domain)}",
        auto_insert_metric_name=False,
        save_last=True,
        verbose=False
    )

    # trainer
    debug_mode = bool(getattr(cfg.attention_model, "debug_mode", False))
    show_progress_bar = bool(getattr(cfg.attention_model, "enable_progress_bar", True))

    # When resuming a full Lightning checkpoint between MAC iterations,
    # Lightning also restores the previous epoch counter.  Extend the epoch
    # budget so this invocation still gets the configured number of fresh
    # epochs instead of stopping at the old max_epochs boundary.
    resume_epoch_offset = 0
    if fit_ckpt_path and os.path.isfile(str(fit_ckpt_path)):
        try:
            resume_meta = torch.load(
                str(fit_ckpt_path),
                map_location="cpu",
                weights_only=False,
            )
            resume_epoch_offset = max(0, int(resume_meta.get("epoch", -1)) + 1)
            print(
                f"[WM] Restoring optimizer state after epoch "
                f"{resume_epoch_offset - 1}; adding {int(cfg.attention_model.n_epochs)} new epochs."
            )
        except Exception as exc:
            print(f"[WM] Could not read resume epoch metadata: {exc}. Using a fresh epoch budget.")
            resume_epoch_offset = 0
    max_epochs = int(cfg.attention_model.n_epochs) + resume_epoch_offset

    trainer = pl.Trainer(
        precision=32,
        logger=logger,
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=False,
        enable_progress_bar=show_progress_bar,
        log_every_n_steps=int(getattr(cfg.attention_model, "log_every_n_steps", 10)),
    )


    result = {
        "mode": None,            
        "net": net,              
        "old_params": None,
        "fisher": None,
        "avg_val_loss": None,
    }

    if ewc_enabled:
        net.set_consolidation(old_params, fisher, load_weights=(old_params is not None))
    else:
        # Replay-only training carries the live model between phases. No
        # parameter anchor or Fisher matrix is created or restored.
        old_params = None
        fisher = None
        net.old_params = None
        net.fisher = None

    if cfg.attention_model.freeze_weight:
        # ===== validation =====
        avg_val_loss = trainer.validate(net, datamodule)

        result["mode"] = "val"
        result["avg_val_loss"] = avg_val_loss
        return result

    else:
        # ===== training =====
        resume_fit_path = None
        if fit_ckpt_path:
            candidate = Path(str(fit_ckpt_path)).expanduser().resolve()
            if candidate.is_file():
                resume_fit_path = str(candidate)
                print(f"[WM] Resuming Lightning phase checkpoint: {resume_fit_path}")
        if resume_fit_path:
            # PyTorch 2.6 changed torch.load's implicit default to
            # weights_only=True. The pinned Lightning version restores its
            # trainer checkpoint with torch.load() without an explicit flag,
            # so trusted local phase checkpoints would fail on resume when
            # they contain the OmegaConf objects stored by Lightning.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        trainer.fit(net, datamodule, ckpt_path=resume_fit_path)

        # Lightning leaves the module at the last epoch, while the desired
        # artifact is the best validation checkpoint. Load that checkpoint
        # before calculating consolidation statistics and writing the final
        # model file.
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            # Lightning checkpoints contain optimizer/configuration objects
            # in addition to tensors. This is a trusted local checkpoint;
            # PyTorch 2.6 otherwise defaults to weights_only=True and rejects
            # the embedded OmegaConf DictConfig.
            best_checkpoint = torch.load(
                best_model_path,
                map_location="cpu",
                weights_only=False,
            )
            net.load_state_dict(best_checkpoint["state_dict"])
            print(f"[WM] Loaded best checkpoint: {best_model_path}")

        if ewc_enabled and bool(getattr(cfg.attention_model, "compute_fisher", True)):
            # Save the current parameters as the consolidation anchor before
            # estimating Fisher for a later EWC-regularized phase.
            old_params = net.save_old_params()
            # Estimate the Fisher information matrix for a later continual phase.
            # Lightning moves the module back to CPU when fit() tears down. Move
            # it explicitly so the expensive per-sample backward passes do not
            # silently run on CPU.
            fisher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net.to(fisher_device)
            print(f"[Fisher] Using device: {fisher_device}")
            fisher_samples = int(getattr(cfg.attention_model, "fisher_samples", 3000))
            fisher_beta = float(getattr(cfg.attention_model, "fisher_beta", 0.5))
            scale_factor = float(getattr(cfg.attention_model, "scale_factor", 1.0))
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
        else:
            old_params = None
            fisher = None
            if bool(getattr(cfg.attention_model, "continue_learning", False)):
                print("[EWC/Fisher] Disabled; continual retention uses replay only.")

    # ... (in run)
        # Save the training checkpoint.
        model_pth = cfg.attention_model.model_save_path
        # Copy the actual Lightning best checkpoint instead of calling
        # trainer.save_checkpoint() again. Older Lightning versions can save
        # the trainer's last in-memory state there, even after the module was
        # restored, which makes the final artifact differ from best_model_path.
        if best_model_path and os.path.exists(best_model_path):
            shutil.copy2(best_model_path, model_pth)
        else:
            trainer.save_checkpoint(model_pth)
        # Lightning may add a version suffix to the temporary filename in
        # older releases. Remove every temporary best checkpoint for this
        # domain, while keeping only the canonical final artifact.
        final_path = Path(str(model_pth)).resolve()
        temporary_pattern = f"best-{str(cfg.domain)}*.ckpt"
        for temporary_path in Path(checkpoint_dir).glob(temporary_pattern):
            if temporary_path.resolve() != final_path:
                temporary_path.unlink(missing_ok=True)
                print(f"[WM] Removed temporary checkpoint: {temporary_path}")
        if use_wandb:
            wandb.save(str(model_pth))
            wandb.save(model_pth)

        result["mode"] = "train"
        result["old_params"] = old_params
        result["fisher"] = fisher
        
        # Capture best validation loss
        best_score = trainer.checkpoint_callback.best_model_score
        result["best_loss"] = best_score.item() if best_score is not None else 0.0
        
        # Capture the unified observation metric (and any non-loss diagnostics).
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
    direct_data=None,
    fit_ckpt_path=None,
):
    result = run(
        cfg,
        net=net,
        old_params=old_params,
        fisher=fisher,
        layout=env_layout,
        replay_data=replay_data,
        direct_data=direct_data,
        fit_ckpt_path=fit_ckpt_path,
    )

    # Return the actual trained module.  Older curriculum code passed a
    # possibly-None `net` argument back to its caller, which made warm-start
    # training silently lose the updated weights.
    return result, result.get("fisher"), result["net"]



if __name__ == "__main__":
    print("THIS SCRIPT IS EXECUTING!")
    train()
