import torch
from torch import nn
from torch import nn
import torch.nn.functional as F


class EmbeddingModule(nn.Module):
    def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads):
        super().__init__()
        self.data_type = data_type
        if data_type == 'discrete':
            self.input_channel = 21
            self.action_embedding = nn.Embedding(7, embed_dim)
        else:
            self.input_channel = grid_shape[0]
            self.action_fc = nn.Linear(1, embed_dim)

        self.mask_size = mask_size
        self.y, self.x = mask_size // 2, mask_size // 2
        self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # Flatten spatial dimensions from (B, embed_dim, H, W) to (B, embed_dim, H*W).
        self.flatten = nn.Flatten(2)
        # Learn one positional embedding per patch with shape (1, H*W, embed_dim).
        self.pos_embedding = nn.Parameter(torch.randn(1, mask_size * mask_size, embed_dim))

        # Action fusion projects action information into the same embedding space.
        self.fuse_fc = nn.Linear(embed_dim * 2, embed_dim)

        self.pre_fc1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.pre_fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.pre_fc3 = nn.Linear(embed_dim, 3)

    def forward(self, state, action):
        orginal_dim = state.ndim
        if orginal_dim == 3:  # Single sample
            state = state.unsqueeze(0)  # Expand to (1, C, H, W).
            action = torch.tensor([action]).to(state.device)
        B, C, H, W = state.size()
        
        if self.data_type == 'discrete':
            obj = state[:, 0, :, :]
            color = state[:, 1, :, :]
            dir = state[:, 2, :, :]
            obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
            color = F.one_hot(state[:, 1, :, :].reshape(B, -1).long(), num_classes=6)
            dir = F.one_hot(state[:, 2, :, :].reshape(B, -1).long(), num_classes=4)
            state_emb = torch.cat([obj, color, dir], dim=-1).float()
            state_emb = state_emb.transpose(1,2).reshape(B, self.input_channel, H, W)
            action_emb = self.action_embedding(action)

        else:
            action_emb = self.action_fc(action.unsqueeze(1))  # (B, embed_dim)
            state_emb = state

        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        # Flatten spatial dimensions from (B, embed_dim, H, W) to (B, embed_dim, H*W).
        x = self.flatten(x)
        # Transpose to (B, H*W, embed_dim) for transformer-style token processing.
        x = x.transpose(1, 2)
        # Add positional embeddings.
        x = x + self.pos_embedding  # (B, 25, embed_dim)

        # Fuse action information.
        # `action` is assumed to be discrete with shape (B,).
        # Broadcast `action_emb` to every spatial token.
        action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        fused = torch.cat([x, action_emb], dim=-1)  # (B, 25, embed_dim*2)
        x = self.fuse_fc(fused)  # (B, 25, embed_dim)

        # Prediction head.
        x = self.pre_fc1(x)
        x = self.pre_fc2(x)
        x = self.pre_fc3(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        if orginal_dim == 3:
            x = x.squeeze(0)
        return x, None
    

    



    

