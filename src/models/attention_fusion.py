import torch
import torch.nn as nn


class AttentionWeightedFusion(nn.Module):
    def __init__(self, latent_dim: int, dropout_rate: float=0.1):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(latent_dim, latent_dim // 2), nn.Tanh(), nn.Linear(latent_dim // 2, 1), nn.Softmax(dim=2))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_img, z_text, z_shared, return_weights=False):
        stacked = torch.stack([z_img, z_text, z_shared], dim=2)
        weights = self.attention_net(stacked)
        fused_z = torch.sum(stacked * weights, dim=2)
        fused_z = self.layer_norm(fused_z)
        fused_z = self.dropout(fused_z)
        if return_weights:
            return fused_z, weights
        return fused_z
