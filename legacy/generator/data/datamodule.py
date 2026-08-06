import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List
import json
import multiprocessing as mp
from tqdm import tqdm
import os

class GenDataset(Dataset):
    def __init__(self, data: List[str], char_to_int: dict):
        # Parse the raw data into input-output pairs
        self.data_pairs = []
        i = 0
        # Get current processing information
        process = mp.current_process()
        for episode_data in data:
            i += 1
            if i % 300 == 0:
                print(f"Process Name: {process.name}, PID: {os.getpid()}, DataIndex: {i}")
            object_map = episode_data.split('\n')
            # maps = episode_data.strip().split('\n\n')
            # object_map = maps[0].split('\n')
            # color_map = maps[1].split('\n')

            # # Convert maps into flattened tensor format
            # object_tensor = self.map_to_tensor(object_map).flatten()
            # color_tensor = self.map_to_tensor(color_map).flatten()

            object_tensor = self.map_to_tensor(object_map, char_to_int)
            # color_tensor = self.map_to_tensor(color_map, char_to_int)
            
            # Combine object and color tensors into a single tensor
            # combined_tensor = torch.cat((object_tensor, color_tensor), dim=0)
            # combined_tensor = torch.stack([object_tensor , color_tensor], dim=0)
            # just keep the obj_map for the data
            self.data_pairs.append(object_tensor)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

    def map_to_tensor(self, map_data: List[str], char_to_int) -> torch.Tensor:
        # Define a mapping from characters to numerical values
        tensor = torch.tensor([[char_to_int[char] for char in row] for row in map_data], dtype=torch.long)
        return tensor

class GenDataModule(pl.LightningDataModule):
    def __init__(self, hparams = None):
        super().__init__()
        self.data = hparams.data_dir
        self.batch_size = hparams.batch_size
        self.n_cpu = hparams.n_cpu
        self.char_to_int = hparams.map_element

    def setup(self, stage: Optional[str] = None):
        with open(self.data, 'r') as file:
            loaded = json.load(file)
        ## ----- 2024 09 09 multiprocessing to load the data ----- 
        # loaded = loaded[0:1000]
        dataset = GenDataset(loaded, self.char_to_int)
        # totalCnt = len(loaded)
        # cupCnt = mp.cpu_count()
        # split_size = totalCnt // cupCnt
        # loadeds = []
        # for i in range(cupCnt):
        #     if i == cupCnt - 1:
        #         loadeds.append(loaded[i * split_size:])
        #     else:
        #         loadeds.append(loaded[i * split_size: (i + 1) * split_size])

        # with mp.Pool(mp.cpu_count()) as pool:
        #     dataset = pool.starmap(GenDataset, [(curLoaded, self.char_to_int) for curLoaded in loadeds])  
        ## ----------------------------------------------------------------

        split_size = int(len(dataset) * 0.9)
        self.data_train, self.data_val = torch.utils.data.random_split(
            dataset, [split_size, len(dataset) - split_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
            pin_memory=True,
        )

# Example usage:
if __name__ == "__main__":
    # Sample data
    sample_data = [
        "WWWWWWWW\nWEWEEDEW\nWWWEWWWW\nWEWEWWWW\nWWEEKWEW\nWWEWEWEW\nWEEWEEEW\nWWWWWWWW\n\nWWWWWWWW\nWEWEEYEW\nWWWEWWWW\nWEWEWWWW\nWWEEYWEW\nWWEWEWEW\nWEEWEEEW\nWWWWWWWW",
        "WWWWWWWW\nWWKEEWEW\nWWWWEWDW\nWWWWEWEW\nWEEWWWEW\nWEEEEEWW\nWWEWWEWW\nWWWWWWWW\n\nWWWWWWWW\nWWYEEWEW\nWWWWEWYW\nWWWWEWEW\nWEEWWWEW\nWEEEEEWW\nWWEWWEWW\nWWWWWWWW",
        # More episodes...
    ]

    # Initialize the data module
    gen_data_module = GenDataModule(sample_data, batch_size=2, n_cpu=2)

    # Setup the data module
    gen_data_module.setup()

    # Fetch a batch of data
    train_loader = gen_data_module.train_dataloader()

    for batch in train_loader:
        print("Combined Tensor:", batch)
        break