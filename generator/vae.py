
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from common.utils import map_value_to_index, map_index_to_value
import torch.nn.functional as F

class VAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters(hparams)
        latent_dim = hparams.latent_dim
        self.latent_dim = latent_dim
        vae_task = hparams.vae_task
        self.num_classes = 3
        self.class_values = hparams.class_value_list
        # Encoder
        self.encoder_conv = nn.Sequential(
        nn.Conv2d(self.num_classes, latent_dim // 2, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(latent_dim // 2, latent_dim, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(latent_dim, 2 * latent_dim, 3, 1, 1), nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(18 * latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(18 * latent_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 2 * latent_dim * 3 * 3)

        self.decoder_deconv = nn.Sequential(
        nn.ConvTranspose2d(2 * latent_dim, latent_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(latent_dim),
        nn.ReLU(),
        nn.Dropout2d(0.3),
        nn.ConvTranspose2d(latent_dim, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 2 * self.latent_dim, 3, 3)
        x_recon = self.decoder_deconv(h)
        return x_recon

    def forward(self, x):
        B, H, W = x.shape
        x_onehot = F.one_hot(x.reshape(B, -1).long(), num_classes=self.num_classes)
        x = x_onehot.permute(0, 2, 1).reshape(B, self.num_classes, H, W).float()
        if self.training:  # 只在训练阶段加噪声，推理阶段不加
            x = x + 0.01 * torch.randn_like(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon_logits = self.decode(z)
        return x_recon_logits, mu, logvar
    
    
    def kl_weight(self):
        if self.current_epoch < 5:
            return 0.1
        elif self.current_epoch < 10:
            return 0.2
        elif self.current_epoch < 15:
            return 0.5
        else:
            return 1.0  # 或者慢慢提到10

    def loss_function_classification(self, recon_x, target_x, mu, logvar):
        # recon_x shape: [batch_size, 3, H, W]
        # target_x shape: [batch_size, H, W]
        weights = torch.tensor([1.0, 1.0, 20.0], device=recon_x.device)  # 让类别2（Goal）更重要！
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = kl_per_dim.mean(dim=0)
        free_bits = 0.1
        kl_loss = torch.sum(torch.maximum(kl_per_dim, torch.tensor(free_bits).to(kl_per_dim.device)))
        recon_loss = nn.functional.cross_entropy(recon_x, target_x, weight=weights, reduction='mean')
        beta = self.kl_weight()
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


    def loss_function_regression(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):

        batch = map_value_to_index(batch, self.class_values)
        recon, mu, logvar = self(batch)
        # map the value to index
        batch = batch.long()
        loss, r, k = self.loss_function_classification(recon, batch, mu, logvar)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_recon', r, on_epoch=True)
        self.log('train_kl', k, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # map the value to index
        batch = map_value_to_index(batch, self.class_values)
        recon, mu, logvar = self(batch)

        # map the value to index
        batch = batch.squeeze(1).long()
        loss, r, k = self.loss_function_classification(recon, batch, mu, logvar)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_recon', r, on_epoch=True)
        self.log('val_kl', k, on_epoch=True)
        if batch_idx % 100 == 0:  # 每100步打印一次
            print(f"Epoch {self.current_epoch} | Step {batch_idx} | Train Loss: {loss.item():.4f} | Recon Loss: {r.item():.4f} | KL Loss: {k.item():.4f}")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

