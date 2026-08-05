import torch
from torch import nn
from torch import nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return x + self.fc2(self.dropout(self.relu(self.fc1(x))))

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        :param d_model: feature dimension
        :param nhead: number of attention heads
        :param dropout: dropout ratio
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        # Use `nn.MultiheadAttention` with `batch_first=True` for (B, seq_len, d_model).
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward network.
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        # Two LayerNorm layers.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: input tensor with shape (B, seq_len, d_model)
        :return:
            - src: transformer-encoder output with shape (B, seq_len, d_model)
            - attn_weights: attention weights with shape (B, num_heads, seq_len, seq_len)
        """
        # Compute self-attention and return the attention weights.
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        # Residual connection + LayerNorm.
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # Feed-forward block.
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


# class AttentionModule(nn.Module):
#     def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads):
#         super().__init__()
#         self.data_type = data_type
#         if data_type == 'discrete':
#             self.input_channel = 21
#             self.action_embedding = nn.Embedding(5, embed_dim)
#             self.key_embedding    = nn.Embedding(2, embed_dim)
#         else:
#             self.input_channel = grid_shape[0]
#             self.action_fc = nn.Linear(1, embed_dim)
 
#         self.mask_size = mask_size
#         self.y, self.x = mask_size // 2, mask_size // 2
#         self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
#         self.bn1 = nn.GroupNorm(8, embed_dim)
#         self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
#         self.bn2 = nn.GroupNorm(8, embed_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)

#         # Flatten spatial dimensions from (B, embed_dim, H, W) to (B, embed_dim, H*W).
#         self.flatten = nn.Flatten(2)
#         # Learn one positional embedding per patch with shape (1, H*W, embed_dim).
#         # self.pos_embedding = nn.Parameter(torch.randn(1, mask_size * mask_size, embed_dim))
#         self.pos_embedding = nn.Parameter(torch.zeros(1, mask_size * mask_size, embed_dim))
#         nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # More stable initialization.

#         # Project action information into the same embedding space.
#         self.fuse_fc = nn.Linear(embed_dim * 2, embed_dim)

#         # Stack custom transformer encoder layers.
#         self.transformer_layers = nn.ModuleList([
#             CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#             for _ in range(1)
#         ])
#         self.fc = nn.Linear(embed_dim, 3)
#         self.act_key_fc = nn.Linear(embed_dim * 2, embed_dim)


#     def forward(self, state, action, info):
#         orginal_dim = state.ndim
#         if orginal_dim == 3:  # Single sample
#             state = state.unsqueeze(0)  # Expand to (1, C, H, W).
#             action = torch.tensor([action]).to(state.device)
#         B, C, H, W = state.size()
        
#         if self.data_type == 'discrete':
#             obj = state[:, 0, :, :]
#             color = state[:, 1, :, :]
#             dir = state[:, 2, :, :]
#             obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
#             color = F.one_hot(color.reshape(B, -1).long(), num_classes=6)
#             dir = F.one_hot(dir.reshape(B, -1).long(), num_classes=4)
#             state_emb = torch.cat([obj, color, dir], dim=-1).float()
#             state_emb = state_emb.transpose(1,2).reshape(B, self.input_channel, H, W)
#             action_emb = self.action_embedding(action)
#             if info is not None and 'carrying_key' in info:
#                 has_key = info['carrying_key']
#                 if not torch.is_tensor(has_key):                 # plain bool / int
#                     has_key = torch.tensor(has_key, device=state.device)
#                 else:                                            # already a tensor
#                     has_key = has_key.to(state.device)
#                 key_emb = self.key_embedding(has_key.long())     # (B, D)
#                 if key_emb.ndim == 1: 
#                     key_emb = key_emb.unsqueeze(0)  
#                 ak = torch.cat([action_emb, key_emb], dim=-1)      # (B, 2D)
#                 action_emb = self.act_key_fc(ak)   
#         else:
#             action_emb = self.action_fc(action.unsqueeze(1))  # (B, embed_dim)
#             state_emb = state

#         x = self.relu(self.bn1(self.conv1(state_emb)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         # Flatten spatial dimensions from (B, embed_dim, H, W) to (B, embed_dim, H*W).
#         x = self.flatten(x)
#         # Transpose to (B, H*W, embed_dim) for transformer processing.
#         x = x.transpose(1, 2)
#         # Add positional embeddings.
#         x = x + self.pos_embedding  # (B, 25, embed_dim)


#         # Fuse action information.
#         # `action` is assumed to be discrete with shape (B,).
#         # Broadcast `action_emb` to every token.
#         action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)
#         fused = torch.cat([x, action_emb], dim=-1)  # (B, 25, embed_dim*2)
#         x = self.fuse_fc(fused)  # (B, 25, embed_dim)

#         # Pass through the transformer encoder layers.
#         attn_weights = None
#         for layer in self.transformer_layers:
#             x, attn_weights = layer(x)

#         # Final output projection.
#         x = self.fc(x)
#         x = x.transpose(1, 2).reshape(B, C, H, W)

#         if orginal_dim == 3:
#             x = x.squeeze(0)
#         return x, attn_weights
        


class AttentionModule(nn.Module):
    def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads, env_type="minigrid", frame_stack=1):
        super().__init__()
        self.data_type = data_type
        self.env_type = env_type
        self.frame_stack = frame_stack
        self.is_bipedal = (env_type == "bipedalwalker")
        self.embed_dim = embed_dim
        if data_type == 'discrete':
            if env_type == 'crafter':
                # 20 object classes (0-19) + 5 direction classes = 25 channels per frame
                self.input_channel = (20 + 5) * frame_stack
                self.action_embedding = nn.Embedding(17, embed_dim) # 17 actions in crafter
                self.inv_fc = nn.Linear(16, embed_dim)
                self.inv_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 16)
                )
            else:
                self.input_channel = (11 + 6 + 4) * frame_stack
                self.action_embedding = nn.Embedding(7, embed_dim)
            self.key_embedding = nn.Embedding(2, embed_dim)
        else:
            if self.is_bipedal:
                self.state_dim = int(grid_shape[-1]) if len(grid_shape) > 0 else 24
                self.action_dim = 4
                if self.state_dim != 24:
                    raise ValueError(
                        f"Bipedal state_dim must be 24, got {self.state_dim}"
                    )
                self.bipedal_token_specs = [
                    ("hull_pose", [0, 1]),
                    ("hull_vel", [2, 3]),
                    ("leg1_hip", [4, 5]),
                    ("leg1_knee", [6, 7]),
                    ("leg1_contact", [8]),
                    ("leg2_hip", [9, 10]),
                    ("leg2_knee", [11, 12]),
                    ("leg2_contact", [13]),
                    ("lidar_near", [14, 15, 16, 17, 18]),
                    ("lidar_far", [19, 20, 21, 22, 23]),
                ]
                self.contact_token_names = {"leg1_contact", "leg2_contact"}
                self.contact_indices = [8, 13]
                self.num_tokens = len(self.bipedal_token_specs)
                self.token_name_to_idx = {
                    name: idx for idx, (name, _) in enumerate(self.bipedal_token_specs)
                }
                self.token_encoders = nn.ModuleDict({
                    name: nn.Linear(len(indices), embed_dim)
                    for name, indices in self.bipedal_token_specs
                })
                self.action_fc = nn.Linear(self.action_dim, embed_dim)
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
                nn.init.trunc_normal_(self.pos_embedding, std=0.02)
                self.token_type_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
                nn.init.trunc_normal_(self.token_type_embedding, std=0.02)
                self.context_fc = nn.Linear(self.state_dim, embed_dim)
                self.token_heads = nn.ModuleDict({
                    name: nn.Linear(embed_dim, len(indices))
                    for name, indices in self.bipedal_token_specs
                    if name not in self.contact_token_names
                })
                self.contact_context_specs = {
                    "leg1_contact": [
                        "leg1_contact",
                        "leg1_hip",
                        "leg1_knee",
                        "hull_vel",
                        "lidar_near",
                    ],
                    "leg2_contact": [
                        "leg2_contact",
                        "leg2_hip",
                        "leg2_knee",
                        "hull_vel",
                        "lidar_near",
                    ],
                }
                self.contact_heads = nn.ModuleDict({
                    name: nn.Sequential(
                        nn.Linear(embed_dim * len(self.contact_context_specs[name]), embed_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(embed_dim, 1),
                    )
                    for name, indices in self.bipedal_token_specs
                    if name in self.contact_token_names
                })
            else:
                self.input_channel = grid_shape[0] * frame_stack
                self.action_fc = nn.Linear(1, embed_dim)

        self.mask_size = mask_size
        self.y, self.x = mask_size // 2, mask_size // 2
        if not self.is_bipedal:
            self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
            self.bn1 = nn.GroupNorm(8, embed_dim)
            self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
            self.bn2 = nn.GroupNorm(8, embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)

        if not self.is_bipedal:
            self.flatten = nn.Flatten(2)
            self.pos_embedding = nn.Parameter(torch.zeros(1, mask_size * mask_size, embed_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.fuse_fc = nn.Linear(embed_dim * 3, embed_dim)
        self.res_mlp = ResidualMLP(embed_dim, embed_dim * 2, dropout=0.1)


        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(2)
        ])
        
        if env_type == 'crafter':
            self.out_channel = 20 + 5  # 20 obj classes (0-19) + 5 dir classes
        elif self.is_bipedal:
            self.out_channel = self.state_dim
        else:
            self.out_channel = 3
        if not self.is_bipedal:
            self.fc = nn.Linear(embed_dim, self.out_channel)
        
        self.dropout_conv = nn.Dropout(p=0.1)

    def tokenize_bipedal_state(self, state):
        # Ensure state is (Batch, 24) even if it comes as (Batch, 1, 24)
        if state.ndim == 3:
            state = state.squeeze(1)
            
        token_feats = []
        for name, indices in self.bipedal_token_specs:
            token_x = state[..., indices]
            token_x = self.token_encoders[name](token_x)
            token_feats.append(token_x)

        x = torch.stack(token_feats, dim=1) # (Batch, NumTokens, EmbedDim)
        x = x + self.pos_embedding + self.token_type_embedding
        return x

    def decode_bipedal_tokens(self, token_features, state):
        batch_size = token_features.size(0)
        out = state.new_zeros(batch_size, self.state_dim)
        contact_logits = {}
        for token_idx, (name, indices) in enumerate(self.bipedal_token_specs):
            if name in self.contact_token_names:
                context_names = self.contact_context_specs[name]
                context_features = [
                    token_features[:, self.token_name_to_idx[token_name], :]
                    for token_name in context_names
                ]
                contact_context = torch.cat(context_features, dim=-1)
                logits = self.contact_heads[name](contact_context)
                contact_logits[name] = logits
            else:
                pred = self.token_heads[name](token_features[:, token_idx, :])
                out[:, indices] = pred
        return out, contact_logits


    def forward(self, state, action, info, inv=None):
        orginal_dim = state.ndim
        if self.is_bipedal:
            if orginal_dim == 1:
                state = state.unsqueeze(0)
                action = torch.as_tensor(action, device=state.device).view(1, -1)
            elif orginal_dim == 2 and not torch.is_tensor(action):
                action = torch.as_tensor(action, device=state.device)

            state = state.float()
            action = action.float()
            B = state.size(0)
            x = self.tokenize_bipedal_state(state)

            action_emb = self.action_fc(action).unsqueeze(1).expand(-1, self.num_tokens, -1)
            context_emb = self.context_fc(state).unsqueeze(1).expand(-1, self.num_tokens, -1)

            fused = torch.cat([x, action_emb, context_emb], dim=-1)
            x = self.fuse_fc(fused)

            attn_weights = None
            for layer in self.transformer_layers:
                x, attn_weights = layer(x)

            x = self.res_mlp(x)
            x_out, contact_logits = self.decode_bipedal_tokens(x, state)

            if orginal_dim == 1:
                x_out = x_out.squeeze(0)
            return x_out, attn_weights, {"contact_logits": contact_logits}

        if orginal_dim == 3:  # Single sample
            state = state.unsqueeze(0)
            action = torch.tensor([action]).to(state.device)
        B, TotalC, H, W = state.size()
        K = self.frame_stack
        C_base = TotalC // K

        # ==== State encoding ====
        if self.data_type == 'discrete':
            all_frames_emb = []
            for k in range(K):
                frame = state[:, k*C_base:(k+1)*C_base]
                if self.env_type == 'crafter':
                    obj = frame[:, 0]
                    dir_id = frame[:, 1]
                    obj_oh = F.one_hot(obj.reshape(B, -1).long(), num_classes=20)  # IDs 0-19
                    dir_oh = F.one_hot(dir_id.reshape(B, -1).long(), num_classes=5)
                    frame_emb = torch.cat([obj_oh, dir_oh], dim=-1).float()
                else:
                    obj = frame[:, 0]
                    color = frame[:, 1]
                    dir_id = frame[:, 2]
                    obj_oh = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
                    color_oh = F.one_hot(color.reshape(B, -1).long(), num_classes=6)
                    dir_oh = F.one_hot(dir_id.reshape(B, -1).long(), num_classes=4)
                    frame_emb = torch.cat([obj_oh, color_oh, dir_oh], dim=-1).float()
                all_frames_emb.append(frame_emb)
            
            # Combine all stacked frames' embeddings
            state_emb = torch.cat(all_frames_emb, dim=-1)
            state_emb = state_emb.transpose(1, 2).reshape(B, self.input_channel, H, W)
        else:
            state_emb = state

        # ==== Convolutional feature extraction ====
        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.flatten(x).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embedding  # Add positional encoding.

        # ==== Prepare action embedding ====
        if self.data_type == 'discrete':
            action_emb = self.action_embedding(action)  # (B, D)
        else:
            action_emb = self.action_fc(action.unsqueeze(1))  # (B, D)

        action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, D)

        # ==== Embed and broadcast context information (key/inventory) ====
        if self.env_type == 'crafter':
            if inv is not None:
                context_emb = self.inv_fc(inv)  # (B, D)
            else:
                context_emb = torch.zeros_like(action_emb[:, 0, :])
            if context_emb.ndim == 1:
                context_emb = context_emb.unsqueeze(0)
        else:
            if info is not None and 'carrying_key' in info:
                has_key = info['carrying_key']
                if not torch.is_tensor(has_key):
                    has_key = torch.tensor(has_key, device=state.device)
                else:
                    has_key = has_key.to(state.device)
                context_emb = self.key_embedding(has_key.long())  # (B, D)
                if context_emb.ndim == 1:
                    context_emb = context_emb.unsqueeze(0)
            else:
                context_emb = torch.zeros_like(action_emb[:, 0, :])  # (B, D)

        context_emb = context_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, D)

        # ==== Fuse patch, action, and context features ====
        fused = torch.cat([x, action_emb, context_emb], dim=-1)  # (B, N, 3D)
        x = self.fuse_fc(fused)  # (B, N, D)

        # ==== Transformer ====
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)

        # ==== Residual MLP before FC ====
        x = self.res_mlp(x)  # shape: (B, N, D)

        # ==== Output head ====
        x_out = self.fc(x)
        x_out = x_out.transpose(1, 2).reshape(B, self.out_channel, H, W)

        if self.env_type == 'crafter':
            # Mean pool over spatial patches to predict inventory
            x_pooled = x.mean(dim=1)  # (B, D)
            inv_pred = self.inv_head(x_pooled) # (B, 16)
        else:
            inv_pred = None

        if orginal_dim == 3:
            x_out = x_out.squeeze(0)
            if inv_pred is not None:
                inv_pred = inv_pred.squeeze(0)
        return x_out, attn_weights, inv_pred
