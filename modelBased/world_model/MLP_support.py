import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import sys
import pytorch_lightning as pl

class SimpleNNModule(pl.LightningModule):
    def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads):
        super(SimpleNNModule, self).__init__()
        obs_size = mask_size**2 * grid_shape[0]
        n_hidden = embed_dim * 2
        total_input_size = obs_size + 1 # plus action size 1 dimension

        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_size, n_hidden),
            nn.BatchNorm1d(n_hidden),  
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),  # Added hidden layer
            nn.ReLU()
        )

        self.state_head = nn.Sequential(
            nn.Linear(n_hidden, obs_size),  # output size
            # nn.Linear(self.n_hidden),
            # nn.ReLU()
        )


    def forward(self, state, action):
        orginal_dim = state.ndim
        if orginal_dim == 3:  # Single sample
            state = state.unsqueeze(0)  # Expand to (1, C, H, W).
            action = torch.tensor([action]).to(state.device)
        B, C, H, W = state.size()
        state = state.flatten(1)
        action = action.unsqueeze(1) 
        combined_input = torch.cat((state, action), dim=1)
        # Convert combined_input to Float if necessary
        if combined_input.dtype != torch.float32:
            combined_input = combined_input.float()
        out = self.shared_layers(combined_input)
        obs_out = self.state_head(out).reshape(B, C, H, W)
        if orginal_dim == 3:
            obs_out = obs_out.squeeze(0)
        return obs_out, None
    
    
    # def extract_topk_regions_padding(self, state, attention_weights, topk=9):
    #     """
    #     according to attention weights, extract top-k regions from state
    #     :param state: state features, shape = (batch_size, seq_len, state_dim)
    #     :param attention_weights: attention weights, shape = (batch_size, seq_len)
    #     :param topk: number of regions to extract
    #     :return: 
    #         extracted_regions: extracted regions, shape = (batch_size, topk, state_dim)
    #         topk_indices: selected indices, shape = (batch_size, topk)
    #     """
    #     # acquire top-k indices
    #     batch_size = state.size(0)
    #     _, topk_indices = torch.topk(attention_weights, topk, dim=1)  # (batch_size, topk)
    #     state = state.permute(0, 2, 3, 1)
    #     attention_weights = attention_weights.view(batch_size, state.size(1), state.size(2))
    #     state = state.reshape(batch_size, -1, 3)
    #     output_data = torch.zeros_like(state)
    #     for i in range(batch_size):
    #         for idx in topk_indices[i]:
    #             output_data[i, idx] = state[i, idx]
    #     return output_data


    # def extract_topk_regions_without_padding(state, attention_weights, topk=5):
    #     """
    #     according to attention weights, extract top-k regions from state
    #     :param state: state features, shape = (batch_size, seq_len, state_dim)
    #     :param attention_weights: attention weights, shape = (batch_size, seq_len)
    #     :param topk: number of regions to extract
    #     :return: 
    #         extracted_regions: extracted regions, shape = (batch_size, topk, state_dim)
    #         topk_indices: selected indices, shape = (batch_size, topk)
    #     """
    #     # acquire top-k indices
    #     topk_values, topk_indices = torch.topk(attention_weights, topk, dim=1)  # (batch_size, topk)

    #     # extract top-k regions
    #     extracted_regions = torch.gather(state, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, state.size(-1)))

    #     return extracted_regions, topk_indices




