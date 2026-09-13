import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from domain.minigrid.transition_codec import minigrid_contract

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
    def __init__(
        self,
        data_type,
        grid_shape,
        mask_size,
        embed_dim,
        num_heads,
        env_type="minigrid",
        frame_stack=1,
        crafter_output_mode="effect",
        crafter_inventory_classes=10,
        crafter_inventory_output_mode="categorical_gate",
        minigrid_transition_mode="effect",
        stochastic_outcome=False,
        stochastic_model="none",
        latent_num_factors=1,
        latent_num_classes=2,
    ):
        super().__init__()
        self.data_type = data_type
        self.env_type = env_type
        self.frame_stack = frame_stack
        self.crafter_output_mode = str(crafter_output_mode)
        self.crafter_inventory_classes = int(crafter_inventory_classes)
        self.crafter_inventory_output_mode = str(crafter_inventory_output_mode)
        self.minigrid_transition_mode = str(minigrid_transition_mode)
        self.stochastic_model = str(stochastic_model).lower()
        # ``stochastic_outcome`` is the v1 ablation flag.  Keep accepting it
        # so existing checkpoints/configurations retain their exact contract.
        self.stochastic_outcome = bool(stochastic_outcome) or self.stochastic_model == "outcome_v1"
        self.stochastic_latent_v2 = self.stochastic_model == "latent_v2"
        self.latent_num_factors = int(latent_num_factors)
        self.latent_num_classes = int(latent_num_classes)
        if self.stochastic_outcome and env_type != "minigrid":
            raise ValueError("stochastic_outcome is currently supported only for MiniGrid")
        if self.stochastic_latent_v2 and env_type != "minigrid":
            raise ValueError("stochastic latent v2 is currently supported only for MiniGrid")
        if self.stochastic_latent_v2 and (self.latent_num_factors < 1 or self.latent_num_classes < 2):
            raise ValueError("latent_v2 requires num_factors >= 1 and num_classes >= 2")
        self.is_bipedal = (env_type == "bipedalwalker")
        self.embed_dim = embed_dim
        if data_type == 'discrete':
            if env_type == 'crafter':
                # 20 object classes (0-19) + 5 direction classes = 25 channels per frame
                self.input_channel = (20 + 5) * frame_stack
                self.action_embedding = nn.Embedding(17, embed_dim) # 17 actions in crafter
                self.inv_fc = nn.Linear(16, embed_dim)
                if self.crafter_inventory_output_mode != "categorical_gate":
                    raise ValueError(
                        "Crafter inventory must use categorical_gate output, got "
                        f"{self.crafter_inventory_output_mode!r}"
                    )
                # Survival: 4 x (KEEP + SET_TO_0..9).
                # Items: 12 x KEEP/CHANGE gate + 12 x next value 0..9.
                inventory_output_width = (
                    4 * (self.crafter_inventory_classes + 1)
                    + 12 * 2
                    + 12 * self.crafter_inventory_classes
                )
                self.inv_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, inventory_output_width)
                )
            else:
                contract = minigrid_contract(self.minigrid_transition_mode)
                self.input_channel = (11 + 6 + 4) * frame_stack
                # The old absolute checkpoint used seven rows, including the
                # native ``done`` row. Keep that shape only for compatibility;
                # the semantic dataset/policy action space is always six.
                action_embedding_size = 7 if contract.mode == "absolute" else contract.action_count
                self.action_embedding = nn.Embedding(action_embedding_size, embed_dim)
                # 0 = empty hands; 1..6 = key colour id + 1.
                self.key_embedding = nn.Embedding(7, embed_dim)
                inventory_output_classes = contract.inventory_output_classes
                self.inv_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, inventory_output_classes),
                )
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
            if self.crafter_output_mode == "effect":
                # KEEP + SET_TO for 20 object and 5 direction categories.
                self.out_channel = (20 + 1) + (5 + 1)
            elif self.crafter_output_mode == "absolute":
                # Read-only compatibility with checkpoints created before the
                # Crafter WM returned to its original change-model semantics.
                self.out_channel = 20 + 5
            else:
                raise ValueError(
                    f"Unsupported Crafter output mode: {self.crafter_output_mode}"
                )
        elif self.is_bipedal:
            self.out_channel = self.state_dim
        else:
            self.out_channel = minigrid_contract(self.minigrid_transition_mode).output_channels
        if not self.is_bipedal:
            self.fc = nn.Linear(embed_dim, self.out_channel)

        if self.env_type == "minigrid":
            contract = minigrid_contract(self.minigrid_transition_mode)
            self.checkpoint_contract = {
                "domain": "minigrid",
                "transition_mode": contract.mode,
                "input_channels": 21,
                "output_channels": contract.output_channels,
                "inventory_output_classes": contract.inventory_output_classes,
                "semantic_action_count": contract.action_count,
            }
            if self.stochastic_outcome:
                # 0 = requested action executes; 1 = navigation action drops.
                self.outcome_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 2),
                )
                self.checkpoint_contract["stochastic_outcome"] = {
                    "version": "minigrid_action_outcome_v1",
                    "classes": ["execute", "noop_failure"],
                    "supervised_field": "action_failed",
                }
            if self.stochastic_latent_v2:
                # A factorised categorical latent is deliberately separate
                # from v1's labelled outcome classifier.  Its embedding is
                # injected before *both* spatial and inventory decoders.
                self.prior_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                    nn.Linear(embed_dim, self.latent_num_factors * self.latent_num_classes),
                )
                self.posterior_head = nn.Sequential(
                    # Current-state/action features, next-state features, and
                    # an optional next inventory token.
                    nn.Linear(embed_dim * 3, embed_dim), nn.ReLU(),
                    nn.Linear(embed_dim, self.latent_num_factors * self.latent_num_classes),
                )
                self.latent_embedding = nn.Parameter(
                    torch.empty(self.latent_num_factors, self.latent_num_classes, embed_dim)
                )
                nn.init.normal_(self.latent_embedding, std=0.02)
                self.checkpoint_contract["stochastic_latent"] = {
                    "version": "one_step_discrete_latent_v2",
                    "num_factors": self.latent_num_factors,
                    "num_classes": self.latent_num_classes,
                }
        
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

    @torch.no_grad()
    def encode_map_features(self, state):
        """Encode a full discrete MiniGrid map for generator novelty scoring.

        This intentionally stops before action/context fusion.  The transition
        model's convolutional representation is therefore reusable for maps
        of different spatial sizes and does not depend on the local attention
        mask or a positional-embedding length.
        """
        if self.env_type != "minigrid" or self.data_type != "discrete":
            raise ValueError("encode_map_features is only available for discrete MiniGrid")
        if state.ndim == 3:
            state = state.unsqueeze(0)
        if state.ndim != 4 or state.size(1) != 3:
            raise ValueError(
                "MiniGrid map features expect state [B,3,H,W], got "
                f"{tuple(state.shape)}"
            )

        state = state.to(next(self.parameters()).device)
        obj = state[:, 0].long().clamp(0, 10)
        color = state[:, 1].long().clamp(0, 5)
        direction = state[:, 2].long().clamp(0, 3)
        state_emb = torch.cat(
            [
                F.one_hot(obj, num_classes=11),
                F.one_hot(color, num_classes=6),
                F.one_hot(direction, num_classes=4),
            ],
            dim=-1,
        ).float().permute(0, 3, 1, 2)

        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(x, (1, 1)).flatten(1),
                F.adaptive_max_pool2d(x, (1, 1)).flatten(1),
            ],
            dim=1,
        )
        return F.normalize(pooled, p=2, dim=1)

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


    def _encode_discrete_tokens(self, state):
        """Shared state-only encoder used by the v2 posterior."""
        B, TotalC, H, W = state.size()
        K = self.frame_stack
        C_base = TotalC // K
        all_frames_emb = []
        for k in range(K):
            frame = state[:, k*C_base:(k+1)*C_base]
            if self.env_type == 'crafter':
                obj_oh = F.one_hot(frame[:, 0].reshape(B, -1).long(), num_classes=20)
                dir_oh = F.one_hot(frame[:, 1].reshape(B, -1).long(), num_classes=5)
                frame_emb = torch.cat([obj_oh, dir_oh], dim=-1).float()
            else:
                obj_oh = F.one_hot(frame[:, 0].reshape(B, -1).long(), num_classes=11)
                color_oh = F.one_hot(frame[:, 1].reshape(B, -1).long(), num_classes=6)
                dir_oh = F.one_hot(frame[:, 2].reshape(B, -1).long(), num_classes=4)
                frame_emb = torch.cat([obj_oh, color_oh, dir_oh], dim=-1).float()
            all_frames_emb.append(frame_emb)
        state_emb = torch.cat(all_frames_emb, dim=-1).transpose(1, 2)
        state_emb = state_emb.reshape(B, self.input_channel, H, W)
        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        return self.flatten(x).transpose(1, 2) + self.pos_embedding

    def _sample_latent(self, logits, sample_mode="sample", generator=None):
        if sample_mode not in {"sample", "mode"}:
            raise ValueError("sample_mode must be 'sample' or 'mode'")
        if sample_mode == "mode":
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, generator=generator
        ).reshape(logits.shape[:-1])

    def forward(self, state, action, info, inv=None, return_outcome=False,
                return_distribution=False, next_state=None, next_inventory=None,
                latent_labels=None,
                sample_mode="sample", generator=None):
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
            x = self._encode_discrete_tokens(state)
        else:
            state_emb = state
            x = self.relu(self.bn1(self.conv1(state_emb)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.dropout_conv(x)
            x = self.flatten(x).transpose(1, 2) + self.pos_embedding

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
            if inv is not None:
                carrying_token = torch.as_tensor(inv, device=state.device).long().reshape(-1)
                context_emb = self.key_embedding(carrying_token)  # (B, D)
            elif info is not None and 'carrying_key' in info:
                # Compatibility fallback for callers that have not migrated to
                # the explicit categorical inventory tensor yet.
                has_key = torch.as_tensor(
                    info['carrying_key'], device=state.device
                ).long().reshape(-1)
                context_emb = self.key_embedding(has_key)
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

        prior_logits = posterior_logits = latent_sample = None
        if return_distribution:
            if not self.stochastic_latent_v2:
                raise ValueError("Distribution prediction requires stochastic_model='latent_v2'")
            x_pooled_before_latent = x.mean(dim=1)
            prior_logits = self.prior_head(x_pooled_before_latent).reshape(
                B, self.latent_num_factors, self.latent_num_classes
            )
            posterior_logits = None
            selection_logits = prior_logits
            if next_state is not None:
                if next_state.ndim == 3:
                    next_state = next_state.unsqueeze(0)
                if next_state.shape[0] != B:
                    raise ValueError("next_state batch size must match state")
                next_tokens = self._encode_discrete_tokens(next_state)
                next_pooled = next_tokens.mean(dim=1)
                if next_inventory is not None:
                    next_inventory_token = torch.as_tensor(
                        next_inventory, device=x.device
                    ).long().reshape(-1)
                    if next_inventory_token.numel() != B:
                        raise ValueError("next_inventory batch size must match state")
                    next_inventory_emb = self.key_embedding(next_inventory_token)
                else:
                    next_inventory_emb = torch.zeros_like(next_pooled)
                posterior_logits = self.posterior_head(
                    torch.cat(
                        [x_pooled_before_latent, next_pooled, next_inventory_emb],
                        dim=-1,
                    )
                ).reshape(B, self.latent_num_factors, self.latent_num_classes)
                selection_logits = posterior_logits
            latent_sample = self._sample_latent(selection_logits, sample_mode, generator)
            # During training the posterior's sampled categorical assignment
            # must influence reconstruction; a straight-through Gumbel sample
            # keeps the decoder input discrete while passing that gradient.
            # Inference remains an explicitly generator-controlled categorical
            # draw from the prior.
            if self.training and posterior_logits is not None:
                assignments = F.gumbel_softmax(selection_logits, tau=1.0, hard=True, dim=-1)
                latent_sample = assignments.argmax(dim=-1)
            else:
                assignments = F.one_hot(latent_sample, self.latent_num_classes).float()
            # Observed MiniGrid failures supervise factor 0 without making the
            # rest of the factorised interface environment-specific.
            if latent_labels is not None:
                labels = torch.as_tensor(latent_labels, device=x.device).reshape(-1).long()
                if labels.numel() != B or ((labels < 0) | (labels >= self.latent_num_classes)).any():
                    raise ValueError("latent_labels must contain one valid class per batch row")
                assignments[:, 0] = F.one_hot(labels, self.latent_num_classes).float()
                latent_sample = latent_sample.clone()
                latent_sample[:, 0] = labels
            latent_context = torch.einsum("bfc,fcd->bd", assignments, self.latent_embedding)
            x = x + latent_context.unsqueeze(1)

        # ==== Output head ====
        x_out = self.fc(x)
        x_out = x_out.transpose(1, 2).reshape(B, self.out_channel, H, W)

        if self.env_type in ('crafter', 'minigrid'):
            # Mean pool over spatial patches to predict the inventory effect.
            x_pooled = x.mean(dim=1)  # (B, D)
            inv_pred = self.inv_head(x_pooled)
            if self.env_type == 'crafter':
                effect_width = 4 * (self.crafter_inventory_classes + 1)
                gate_width = 12 * 2
                survival_raw = inv_pred[:, :effect_width]
                gate_raw = inv_pred[:, effect_width:effect_width + gate_width]
                value_raw = inv_pred[:, effect_width + gate_width:]
                inv_pred = {
                    "survival_effect_logits": survival_raw.reshape(
                        B, 4, self.crafter_inventory_classes + 1
                    ).transpose(1, 2).contiguous(),
                    "item_gate_logits": gate_raw.reshape(B, 12, 2)
                    .transpose(1, 2).contiguous(),
                    "item_value_logits": value_raw.reshape(
                        B, 12, self.crafter_inventory_classes
                    ).transpose(1, 2).contiguous(),
                }
            outcome_logits = (
                self.outcome_head(x_pooled)
                if return_outcome and self.stochastic_outcome
                else None
            )
        else:
            inv_pred = None
            outcome_logits = None

        if return_outcome and outcome_logits is None:
            raise ValueError(
                "Stochastic outcome logits were requested from a model without "
                "the MiniGrid stochastic outcome head"
            )

        if orginal_dim == 3:
            x_out = x_out.squeeze(0)
            if inv_pred is not None:
                if isinstance(inv_pred, dict):
                    inv_pred = {key: value.squeeze(0) for key, value in inv_pred.items()}
                else:
                    inv_pred = inv_pred.squeeze(0)
            if outcome_logits is not None:
                outcome_logits = outcome_logits.squeeze(0)
        if return_outcome:
            return x_out, attn_weights, inv_pred, outcome_logits
        if return_distribution:
            return {
                "state_logits": x_out,
                "attention_weights": attn_weights,
                "inventory_logits": inv_pred,
                "prior_logits": prior_logits,
                "posterior_logits": posterior_logits,
                "latent_sample": latent_sample,
            }
        return x_out, attn_weights, inv_pred
