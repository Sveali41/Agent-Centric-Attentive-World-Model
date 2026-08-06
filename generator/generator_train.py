import torch
from torch import nn
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data.datamodule import GenDataModule
from data.datamodule_vae import VaeDataModule
import hydra
from modelBased.common.utils import PROJECT_ROOT, get_env
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from gen import GAN
from modelBased.common.utils import GENERATOR_PATH
import wandb
from generator.common.utils import map_index_to_value
from data.env_dataset_support import *

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    # data
    if hparams.training_generator.generator == "vae":
        dataloader = VaeDataModule(hparams = hparams.dataloader_vae)
        # dataloader.setup()
    else:
        dataloader = GenDataModule(hparams = hparams.dataloader_gan)
    # model
    if hparams.training_generator.generator == "deconv":
        from deconv_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.deconv.z_shape, len(hparams.training_generator.map_element)), 
                    discriminator=Discriminator(dropout = hparams.deconv.dropout), 
                    z_size=hparams.deconv.z_shape, lr=hparams.training_generator.lr, wd=hparams.training_generator.wd)
        
    elif hparams.training_generator.generator == "basic":
        from basic_gen import Generator, Discriminator
        model = GAN(generator=Generator(hparams.basic.z_shape, hparams.basic.dropout), discriminator=Discriminator(hparams.basic.input_channels, hparams.basic.dropout), z_size=hparams.basic.z_shape, lr=0.0002, wd=0.0)
    elif hparams.training_generator.generator == "vae":
        from vae import VAE
        model = VAE(hparams.vae)
    wandb_logger = None 
    if hparams.training_generator.use_wandb:
        wandb_logger = WandbLogger(project="Gen Training", log_model=True)
        wandb_logger.experiment.watch(model, log='all', log_freq=1000)

    # Define the trainer
    if hparams.training_generator.generator == "basic" or hparams.training_generator.generator == "deconv":
        metric_to_monitor = 'd_loss' #"loss"
    elif hparams.training_generator.generator == "vae":
        metric_to_monitor = 'train_loss'
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.001, patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('GENERATOR_MODEL_PATH'),
                            filename ="gen-{epoch:02d}-{g_loss:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger,
                    max_epochs=hparams.training_generator.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(model,dataloader)
    # Log the trained model
    model_pth = hparams.training_generator.pth_path
    trainer.save_checkpoint(model_pth)
    if hparams.training_generator.use_wandb:
        wandb.save(str(model_pth))



@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def validate(cfg: DictConfig):
    hparams = cfg
    # instantiate model based on config
    if hparams.training_generator.generator == "deconv":
        from deconv_gen import Generator, Discriminator
        model = GAN(
            generator=Generator(hparams.deconv.z_shape, len(hparams.training_generator.map_element)), 
            discriminator=Discriminator(dropout = hparams.deconv.dropout), 
            z_size=hparams.deconv.z_shape,
            lr=hparams.training_generator.lr,
            wd=hparams.training_generator.wd
        )
    elif hparams.training_generator.generator == "basic":
        from basic_gen import Generator, Discriminator
        model = GAN(
            generator=Generator(hparams.basic.z_shape, hparams.basic.dropout),
            discriminator=Discriminator(hparams.basic.input_channels, hparams.basic.dropout),
            z_size=hparams.basic.z_shape,
            lr=0.0002,
            wd=0.0
        )
    elif hparams.training_generator.generator == "vae":
        # ====== NEW VAE BRANCH ======
        from vae import VAE
        model = VAE(hparams.vae)
    else:
        raise ValueError(f"Unknown model: {hparams.training_generator.model}")

    # load checkpoint
    checkpoint = torch.load(hparams.training_generator.validation_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    batch_size = hparams.training_generator.batch_size
    num_tests = 1

    for i in range(num_tests):
        # for GANs you sample z and do model(z)
        if hparams.training_generator.generator in ["basic", "deconv"]:
            z = torch.randn(batch_size, hparams.deconv.z_shape)
            with torch.no_grad():
                generated = model(z)
                # if your GAN’s output is logits over map-elements:
                generated_maps = torch.argmax(generated, dim=1)
        # for VAE you sample from prior and decode
        elif hparams.training_generator.generator == "vae":
            # sample z ~ N(0,1)
            class_values = hparams.vae.class_value_list
            z = torch.randn(batch_size, hparams.vae.latent_dim)
            with torch.no_grad():
                # your VAE should have a decode method
                generated = model.decode(z)  
                # if decode returns continuous, maybe threshold or argmax
                generated_maps = torch.argmax(generated, dim=1)
                generated_maps = map_index_to_value(generated_maps, class_values)
                visualize_grid(
                    generated_maps,
                    count=32,
                    save_flag=True,
                    save_path=str(PROJECT_ROOT / 'generator' / 'result'),
                )

if __name__ == "__main__":
    train()
    validate()
    pass
