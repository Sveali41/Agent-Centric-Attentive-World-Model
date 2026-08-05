
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from common.utils import map_value_to_index, map_index_to_value


class VAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters(hparams)
        latent_dim = hparams.latent_dim
        vae_task = hparams.vae_task
        ch = 1
        if vae_task == 'classification':
            self.class_values = hparams.class_value_list

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, latent_dim // 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(latent_dim // 2,latent_dim, 3, 1, 0), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder
        if vae_task == 'regression':
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (latent_dim, 1, 1)),
                nn.ConvTranspose2d(latent_dim, latent_dim // 2, 4, 2, 1), nn.ReLU(),  # 1 -> 2
                nn.ConvTranspose2d(latent_dim // 2, 32, 4, 2, 1), nn.ReLU(),          # 2 -> 4
                nn.ConvTranspose2d(32, 32, 4, 2, 0), nn.ReLU(),                       # 4 -> 10
                nn.ConvTranspose2d(32, ch, 3, 1, 0), nn.Sigmoid(),                    # 8 -> ??
            )
        elif vae_task == 'classification':
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (latent_dim, 1, 1)),
                nn.ConvTranspose2d(latent_dim, latent_dim // 2, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(latent_dim // 2, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 4, 2, 0), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 3, 1, 0)  
            )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    
    def kl_weight(self):
        return min(1.0, self.current_epoch / 50) * 0.1   # 第50轮以后到0.1

    def loss_function_classification(self, recon_x, target_x, mu, logvar):
        # recon_x shape: [batch_size, 3, H, W]
        # target_x shape: [batch_size, H, W]
        weights = torch.tensor([1.0, 1.0, 5.0], device=recon_x.device)  # 让类别2（Goal）更重要！
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target_x.size(0)
        recon_loss = nn.functional.cross_entropy(recon_x, target_x, weight=weights, reduction='mean')
        beta = self.kl_weight()
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


    def loss_function_regression(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        if self.hparams.vae_task == 'classification':
            # map the value to index
            batch = map_value_to_index(batch, self.class_values)
        batch = batch.unsqueeze(1)
        recon, mu, logvar = self(batch)
        if self.hparams.vae_task == 'classification':
            # map the value to index
            batch = batch.squeeze(1).long()
            loss, r, k = self.loss_function_classification(recon, batch, mu, logvar)
        elif self.hparams.vae_task == 'regression':
            loss, r, k = self.loss_function_regression(recon, batch, mu, logvar)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_recon', r, on_epoch=True)
        self.log('train_kl', k, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.vae_task == 'classification':
            # map the value to index
            batch = map_value_to_index(batch, self.class_values)
        batch = batch.unsqueeze(1)
        recon, mu, logvar = self(batch)
        if self.hparams.vae_task == 'classification':
            # map the value to index
            batch = batch.squeeze(1).long()
            loss, r, k = self.loss_function_classification(recon, batch, mu, logvar)
        elif self.hparams.vae_task == 'regression':
            loss, r, k = self.loss_function_regression(recon, batch, mu, logvar)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_recon', r, on_epoch=True)
        self.log('val_kl', k, on_epoch=True)
        if batch_idx % 100 == 0:  # 每100步打印一次
            print(f"Epoch {self.current_epoch} | Step {batch_idx} | Train Loss: {loss.item():.4f} | Recon Loss: {r.item():.4f} | KL Loss: {k.item():.4f}")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

