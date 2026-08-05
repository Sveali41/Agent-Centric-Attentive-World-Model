import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class VaeDataset(Dataset):
    def __init__(self, pkl_path):
        # 1) load once
        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        # keep a list of keys
        self.keys = list(self.data_dict.keys())   
    def __len__(self):
        return len(self.data_dict)  

    def __getitem__(self, idx):
        key = self.keys[idx]
        feat = self.data_dict[key]  # assume value is just feature array
        x = torch.tensor(feat, dtype=torch.float32)
        return x


class VaeDataModule(pl.LightningDataModule):
    def __init__(self, hparams = None):
        super().__init__()
        self.pkl_path   = hparams.data_dir
        self.batch_size = hparams.batch_size
        self.num_workers= hparams.n_cpu

    def setup(self, stage=None):
        # called on every GPU
        self.dataset = VaeDataset(self.pkl_path)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)