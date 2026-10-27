from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, n_samples: int=1) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    if n_samples == 1:
        return mu + std * torch.randn_like(std)

    shape = (n_samples,) + mu.shape
    eps = torch.randn(shape, device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(0).expand(shape) + std.unsqueeze(0).expand(shape) * eps


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent_dim: int):
        super().__init__()
        self.layer_norm_input = nn.LayerNorm(input_dim)
        self.fc_mu = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.GELU(), nn.Linear(input_dim // 2, input_dim // 4), nn.GELU(), nn.Linear(input_dim // 4, latent_dim), nn.GELU())
        self.fc_log_var = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.GELU(), nn.Linear(input_dim // 2, input_dim // 4), nn.GELU(), nn.Linear(input_dim // 4, latent_dim))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer_norm_input(x)
        mu = self.fc_mu(x)
        logvar = -F.softplus(self.fc_log_var(x), beta=1)
        return mu, logvar

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def poe_fusion(mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logvar1 = torch.clamp(logvar1, -10, 10)
    logvar2 = torch.clamp(logvar2, -10, 10)
    prec1 = torch.exp(-logvar1)
    prec2 = torch.exp(-logvar2)
    prec_fused = prec1 + prec2
    mu_fused = (mu1 * prec1 + mu2 * prec2) / (prec_fused + 1e-08)
    var_fused = 1.0 / (prec_fused + 1e-08)
    logvar_fused = torch.log(var_fused + 1e-08)
    return mu_fused, logvar_fused


class MultimodalVAE(nn.Module):
    """VAE-style Gaussian encoders with PoE fusion."""

    def __init__(self, emb_dim: int, enc_hidden: int=256, latent_dim: int=64, fusion_hidden: int=256):
        super().__init__()
        self.enc_text = SimpleEncoder(emb_dim, enc_hidden, latent_dim)
        self.enc_img = SimpleEncoder(emb_dim, enc_hidden, latent_dim)
        self.latent_dim = latent_dim

    def encode_batch(self, text: torch.Tensor, img: torch.Tensor, mask: torch.Tensor=None, fusion_method='poe') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = text.shape
        text_aggregated = text.mean(dim=2)
        img_aggregated = img.mean(dim=2)
        emb_dim = text_aggregated.shape[-1]

        text_flat = text_aggregated.reshape(batch_size * seq_len, emb_dim)
        img_flat = img_aggregated.reshape(batch_size * seq_len, emb_dim)
        mu1_flat, logvar1_flat = self.enc_text(text_flat)
        mu2_flat, logvar2_flat = self.enc_img(img_flat)

        mu1 = mu1_flat.view(batch_size, seq_len, -1)
        logvar1 = logvar1_flat.view(batch_size, seq_len, -1)
        mu2 = mu2_flat.view(batch_size, seq_len, -1)
        logvar2 = logvar2_flat.view(batch_size, seq_len, -1)
        z_text = reparameterize(mu1, logvar1)
        z_img = reparameterize(mu2, logvar2)

        if fusion_method != 'poe':
            raise NotImplementedError("Only 'poe' fusion is supported.")

        mu_fused, logvar_fused = poe_fusion(mu1, logvar1, mu2, logvar2)
        z_shared = reparameterize(mu_fused, logvar_fused)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            z_text = z_text * mask_expanded
            z_img = z_img * mask_expanded
            mu1 = mu1 * mask_expanded
            logvar1 = logvar1 * mask_expanded
            mu2 = mu2 * mask_expanded
            logvar2 = logvar2 * mask_expanded
            z_shared = z_shared * mask_expanded
            mu_fused = mu_fused * mask_expanded
            logvar_fused = logvar_fused * mask_expanded

        return z_text, z_img, mu1, logvar1, mu2, logvar2, z_shared, mu_fused, logvar_fused


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    kl_per_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    if mask is not None:
        kl_masked = kl_per_item * mask.float()
        return kl_masked.sum() / (mask.sum() + 1e-08)
    return kl_per_item.mean()


def compute_vae_loss(mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor, mu_fused: torch.Tensor, logvar_fused: torch.Tensor, mask: torch.Tensor=None, beta: float=1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    kl_text = compute_kl_divergence(mu1, logvar1, mask)
    kl_img = compute_kl_divergence(mu2, logvar2, mask)
    kl_fused = compute_kl_divergence(mu_fused, logvar_fused, mask)
    kl_loss = kl_text + kl_img + kl_fused
    recon_loss = torch.tensor(0.0, device=mu1.device)
    total_vae_loss = recon_loss + beta * kl_loss
    diagnostics = {
        'kl_loss_total': kl_loss.item(),
        'kl_loss_text': kl_text.item(),
        'kl_loss_img': kl_img.item(),
        'kl_loss_fused': kl_fused.item(),
        'recon_loss': recon_loss.item(),
    }
    return total_vae_loss, diagnostics
