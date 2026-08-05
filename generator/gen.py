import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

###################For regression#########################
# class GAN(pl.LightningModule):
#     def __init__(
#         self,
#         generator: torch.nn.Module,
#         discriminator: torch.nn.Module,
#         z_size: int,
#         lr: float = 0.0002,
#         wd: float = 0.0
#     ):
#         super(GAN, self).__init__()
#         self.generator = generator
#         self.discriminator = discriminator
#         self.z_size = z_size
#         self.lr = lr
#         self.wd = wd

#         self.save_hyperparameters()

#     def forward(self, z):
#         return self.generator(z)

#     def generator_loss_function(self, discriminator_output_fake):
#         loss = F.binary_cross_entropy(discriminator_output_fake, torch.ones_like(discriminator_output_fake))
#         return loss

#     def discriminator_loss_function(self, discriminator_output_real, discriminator_output_fake):
#         loss_real = F.binary_cross_entropy(discriminator_output_real, torch.ones_like(discriminator_output_real))
#         loss_fake = F.binary_cross_entropy(discriminator_output_fake, torch.zeros_like(discriminator_output_fake))
#         loss = loss_real + loss_fake
#         return loss

#     def configure_optimizers(self):
#         gen_params = self.generator.parameters()
#         disc_params = self.discriminator.parameters()

#         generator_optimizer = optim.Adam(gen_params, lr=self.lr, betas=(0.5, 0.999))
#         discriminator_optimizer = optim.Adam(disc_params, lr=self.lr, betas=(0.5, 0.999))

#         reduce_lr_on_plateau_gen = {
#             "scheduler": ReduceLROnPlateau(generator_optimizer, mode='min', verbose=True, min_lr=1e-8),
#             "monitor": "g_loss",
#             "frequency": 1,
#             "interval": "epoch",
#             "strict": False
#         }
#         reduce_lr_on_plateau_disc = {
#             "scheduler": ReduceLROnPlateau(discriminator_optimizer, mode='min', verbose=True, min_lr=1e-8),
#             "monitor": "d_loss",
#             "frequency": 1,
#             "interval": "epoch",
#             "strict": False
#         }
#         return [generator_optimizer, discriminator_optimizer], [reduce_lr_on_plateau_gen, reduce_lr_on_plateau_disc]


#     def training_step(self, batch, batch_idx, optimizer_idx):
#         real_images = batch
#         batch_size = real_images.size(0)
#         z = torch.randn(batch_size, self.z_size, 1, 1, device=self.device)

#         if optimizer_idx == 0:
#             # Generator training
#             fake_images = self(z)
#             discriminator_output_fake = self.discriminator(fake_images)
#             g_loss = self.generator_loss_function(discriminator_output_fake)
#             self.log("g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#             return g_loss

#         if optimizer_idx == 1:
#             # Discriminator training
#             fake_images = self(z).detach()
#             discriminator_output_real = self.discriminator(real_images)
#             discriminator_output_fake = self.discriminator(fake_images)
#             d_loss = self.discriminator_loss_function(discriminator_output_real, discriminator_output_fake)
#             self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#             return d_loss

#     def validation_step(self, batch,batch_idx):
#         real_images = batch
#         batch_size = real_images.size(0)

#         z = torch.randn(batch_size, self.z_size, 1, 1, device=self.device)
#         fake_images = self(z)

#         discriminator_output_real = self.discriminator(real_images)
#         discriminator_output_fake = self.discriminator(fake_images)

#         g_loss = self.generator_loss_function(discriminator_output_fake)
#         d_loss = self.discriminator_loss_function(discriminator_output_real, discriminator_output_fake)

#         self.log("val_g_loss", g_loss, on_epoch=True, prog_bar=True)
#         self.log("val_d_loss", d_loss, on_epoch=True, prog_bar=True)



###################For classification#########################

class GAN(pl.LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        z_size: int,
        lr: float = 0.0002,
        wd: float = 0.0
    ):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.z_size = z_size
        self.lr = lr
        self.wd = wd

        self.save_hyperparameters()

    def forward(self, z):
        return self.generator(z)

    def generator_loss_function(self, discriminator_output_fake):
        loss = F.binary_cross_entropy(discriminator_output_fake, torch.ones_like(discriminator_output_fake))
        return loss

    def discriminator_loss_function(self, discriminator_output_real, discriminator_output_fake):
        loss_real = F.binary_cross_entropy(discriminator_output_real, torch.ones_like(discriminator_output_real))
        loss_fake = F.binary_cross_entropy(discriminator_output_fake, torch.zeros_like(discriminator_output_fake))
        loss = loss_real + loss_fake
        return loss

    def configure_optimizers(self):
        gen_params = self.generator.parameters()
        disc_params = self.discriminator.parameters()

        generator_optimizer = optim.Adam(gen_params, lr=self.lr, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(disc_params, lr=self.lr, betas=(0.5, 0.999))

        reduce_lr_on_plateau_gen = {
            "scheduler": ReduceLROnPlateau(generator_optimizer, mode='min', verbose=True, min_lr=1e-8),
            "monitor": "g_loss",
            "frequency": 1,
            "interval": "epoch",
            "strict": False
        }
        reduce_lr_on_plateau_disc = {
            "scheduler": ReduceLROnPlateau(discriminator_optimizer, mode='min', verbose=True, min_lr=1e-8),
            "monitor": "d_loss",
            "frequency": 1,
            "interval": "epoch",
            "strict": False
        }
        return [generator_optimizer, discriminator_optimizer], [reduce_lr_on_plateau_gen, reduce_lr_on_plateau_disc]


    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images = batch
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, self.z_size, 1, 1, device=self.device)

        if optimizer_idx == 0:
            # Generator training
            fake_softmax = self(z)
            discriminator_output_fake = self.discriminator(fake_softmax)
            g_loss = self.generator_loss_function(discriminator_output_fake)
            self.log("g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return g_loss

        if optimizer_idx == 1:
            # Discriminator training
            fake_images = self(z).detach()
            discriminator_output_real = self.discriminator(real_images)
            discriminator_output_fake = self.discriminator(fake_images)
            d_loss = self.discriminator_loss_function(discriminator_output_real, discriminator_output_fake)
            self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return d_loss

    def validation_step(self, batch,batch_idx):
        real_images = batch
        batch_size = real_images.size(0)

        z = torch.randn(batch_size, self.z_size, 1, 1, device=self.device)
        fake_images = self(z)

        discriminator_output_real = self.discriminator(real_images)
        discriminator_output_fake = self.discriminator(fake_images)

        g_loss = self.generator_loss_function(discriminator_output_fake)
        d_loss = self.discriminator_loss_function(discriminator_output_real, discriminator_output_fake)

        self.log("val_g_loss", g_loss, on_epoch=True, prog_bar=True)
        self.log("val_d_loss", d_loss, on_epoch=True, prog_bar=True)