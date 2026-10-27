import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .attention_fusion import AttentionWeightedFusion
from .mamba_block import CrossModalTransformerLayer
from .moe_fusion import MoEVecFusionLayer
from .mutual_information import DisentangleMILoss
from .sasrec import SASRec
from .vae import MultimodalVAE, compute_vae_loss


class IntegratedMultimodalModel(nn.Module):
    """Integrated DiMMRec model."""

    def __init__(self, emb_dim: int, enc_hidden: int=256, latent_dim: int=64, fusion_hidden: int=256, num_heads: int=8, transformer_layers: int=1, drop_path: float=0.0, dropout_rate: float=0.2, max_seq_len: int=50, num_blocks: int=2, sasrec_hidden_units: int=128, img_cont: int=4, text_cont: int=9, mamba_d_state: int=16, mamba_d_conv: int=2, mamba_expand: int=1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vae_model = MultimodalVAE(emb_dim=emb_dim, enc_hidden=enc_hidden, latent_dim=latent_dim, fusion_hidden=fusion_hidden)
        self.transformer_layers = nn.ModuleList([CrossModalTransformerLayer(dim=latent_dim, num_heads=num_heads, drop_path=drop_path, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for _ in range(transformer_layers)])
        self.multimodal_fusion = AttentionWeightedFusion(latent_dim=latent_dim, dropout_rate=0.1)
        self.proj_text_to_latent = nn.Sequential(nn.Linear(emb_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.proj_img_to_latent = nn.Sequential(nn.Linear(emb_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.sasrec = SASRec(hidden_units=sasrec_hidden_units, max_seq_len=max_seq_len, num_blocks=num_blocks, dropout_rate=dropout_rate)
        self.moe_text = MoEVecFusionLayer(input_dim=latent_dim, n_tokens=text_cont, latent_dim=latent_dim, num_experts=4, top_k=4)
        self.moe_img = MoEVecFusionLayer(input_dim=latent_dim, n_tokens=img_cont, latent_dim=latent_dim, num_experts=4, top_k=4)
        self.mi_loss_module = DisentangleMILoss(dim=latent_dim)
        self.latent_dim = latent_dim
        self.sasrec_hidden_units = sasrec_hidden_units
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=1.0)

    def forward(self, text, img, mask=None, fusion_method='poe', neg_text=None, neg_img=None):
        _, _, mu1, logvar1, mu2, logvar2, z_shared_init, mu_fused, logvar_fused = self.vae_model.encode_batch(text, img, mask, fusion_method=fusion_method)

        batch_size, total_len, _, _ = text.shape
        text_proj = self.proj_text_to_latent(text)
        img_proj = self.proj_img_to_latent(img)

        bsz, seq_len = text_proj.shape[:2]
        num_text, dim_text = text_proj.shape[2], text_proj.shape[3]
        num_img, dim_img = img_proj.shape[2], img_proj.shape[3]
        assert dim_text == dim_img, f'latent dim mismatch: text C={dim_text}, img C={dim_img}'

        z_text_flat = text_proj.reshape(bsz * seq_len, num_text, dim_text)
        z_img_flat = img_proj.reshape(bsz * seq_len, num_img, dim_text)
        z_shared_flat = z_shared_init.reshape(bsz * seq_len, 1, dim_text)

        for mamba_layer in self.transformer_layers:
            z_img_flat, z_text_flat, z_shared_flat = mamba_layer(z_img_flat, z_text_flat, z_shared_flat, mask=None)

        z_img = z_img_flat.reshape(bsz, seq_len, num_img, dim_text)
        z_text = z_text_flat.reshape(bsz, seq_len, num_text, dim_text)
        z_shared = z_shared_flat.reshape(bsz, seq_len, dim_text)

        z_img_fused, _ = self.moe_img(z_img)
        z_text_fused, _ = self.moe_text(z_text)
        z_shared_fused = z_shared
        multimodal_emb = self.multimodal_fusion(z_img_fused, z_text_fused, z_shared_fused, return_weights=False)

        sas_output = None
        sas_shift_labels = None
        if mask is not None:
            seq_len_max = self.max_seq_len
            if total_len == seq_len_max + 1:
                sas_output, sas_shift_labels = self._train_logits(multimodal_emb, mask, neg_text, neg_img, fusion_method)
            else:
                sas_output = self._eval_logits(multimodal_emb, mask)

        return {
            'z_img': z_img_fused,
            'z_text': z_text_fused,
            'z_shared': z_shared_fused,
            'mu1': mu1,
            'logvar1': logvar1,
            'mu2': mu2,
            'logvar2': logvar2,
            'mu_fused': mu_fused,
            'logvar_fused': logvar_fused,
            'sas_output': sas_output,
            'sas_shift_labels': sas_shift_labels,
        }

    def _train_logits(self, multimodal_emb, mask, neg_text, neg_img, fusion_method):
        seq_input = multimodal_emb[:, :-1, :]
        seq_target = multimodal_emb[:, 1:, :]
        mask_input = mask[:, :-1].bool()
        mask_target = mask[:, 1:].bool()
        valid_mask = mask_input & mask_target
        b_idx, t_idx = valid_mask.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return None, None

        hist_batch, hist_mask = self._pack_shifted_histories(seq_input, mask_input, b_idx, t_idx)
        query = self.sasrec.forward_direct_batch(hist_batch, masks=hist_mask)
        pos_emb = seq_target[b_idx, t_idx]

        if neg_text is None or neg_img is None:
            logits = torch.matmul(query, pos_emb.transpose(0, 1)) / query.size(-1) ** 0.5
            labels = torch.arange(query.size(0), device=query.device, dtype=torch.long)
            return logits, labels

        neg_emb = self._encode_negative_items(neg_text, neg_img, b_idx, t_idx, fusion_method)
        pos_logits = (query * pos_emb).sum(dim=-1, keepdim=True)
        neg_logits = torch.einsum('nd,nkd->nk', query, neg_emb)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
        return logits, labels

    def _pack_shifted_histories(self, seq_input, mask_input, b_idx, t_idx):
        seq_len = seq_input.size(1)
        dim = seq_input.size(2)
        num_samples = b_idx.numel()
        device = seq_input.device

        selected_seq = seq_input[b_idx]
        selected_mask = mask_input[b_idx]
        dest = torch.arange(seq_len, device=device).unsqueeze(0).expand(num_samples, seq_len)
        shift_start = (seq_len - (t_idx + 1)).unsqueeze(1)
        src_idx = dest - shift_start
        src_idx_clamped = src_idx.clamp(min=0, max=seq_len - 1)
        gather_idx = src_idx_clamped.unsqueeze(-1).expand(num_samples, seq_len, dim)

        hist_batch = selected_seq.gather(1, gather_idx)
        src_valid = (src_idx >= 0) & (src_idx <= t_idx.unsqueeze(1))
        hist_mask = src_valid & selected_mask.gather(1, src_idx_clamped)
        hist_batch = hist_batch * hist_mask.unsqueeze(-1)
        return hist_batch, hist_mask

    def _encode_negative_items(self, neg_text, neg_img, b_idx, t_idx, fusion_method):
        batch_size, seq_len, num_neg, num_text, text_dim = neg_text.shape
        _, _, _, num_img, img_dim = neg_img.shape
        assert text_dim == img_dim, 'negative text/image feature dim mismatch'
        assert seq_len == self.max_seq_len, 'negative sample shape mismatch with max_seq_len'

        neg_text_flat = neg_text.reshape(batch_size * seq_len * num_neg, num_text, text_dim)
        neg_img_flat = neg_img.reshape(batch_size * seq_len * num_neg, num_img, img_dim)
        neg_mask = torch.ones(batch_size * seq_len * num_neg, 1, dtype=torch.bool, device=neg_text.device)

        _, _, _, _, _, _, z_shared_neg, _, _ = self.vae_model.encode_batch(neg_text_flat.unsqueeze(1), neg_img_flat.unsqueeze(1), neg_mask, fusion_method=fusion_method)
        text_neg = self.proj_text_to_latent(neg_text_flat.unsqueeze(1))
        img_neg = self.proj_img_to_latent(neg_img_flat.unsqueeze(1))

        num_text_neg, dim_neg = text_neg.shape[2], text_neg.shape[3]
        num_img_neg = img_neg.shape[2]
        z_text_neg = text_neg.reshape(batch_size * seq_len * num_neg, num_text_neg, dim_neg)
        z_img_neg = img_neg.reshape(batch_size * seq_len * num_neg, num_img_neg, dim_neg)
        z_shared_neg = z_shared_neg.reshape(batch_size * seq_len * num_neg, 1, dim_neg)

        for mamba_layer in self.transformer_layers:
            z_img_neg, z_text_neg, z_shared_neg = mamba_layer(z_img_neg, z_text_neg, z_shared_neg, mask=None)

        z_img_neg = z_img_neg.reshape(batch_size * seq_len * num_neg, 1, num_img_neg, dim_neg)
        z_text_neg = z_text_neg.reshape(batch_size * seq_len * num_neg, 1, num_text_neg, dim_neg)
        z_shared_neg = z_shared_neg.reshape(batch_size * seq_len * num_neg, 1, dim_neg)
        z_img_fused_neg, _ = self.moe_img(z_img_neg)
        z_text_fused_neg, _ = self.moe_text(z_text_neg)
        neg_emb = self.multimodal_fusion(z_img_fused_neg, z_text_fused_neg, z_shared_neg, return_weights=False)
        return neg_emb.reshape(batch_size, seq_len, num_neg, -1)[b_idx, t_idx]

    def _eval_logits(self, multimodal_emb, mask):
        seq_len_max = self.max_seq_len
        seq_embs = multimodal_emb[:, :seq_len_max, :]
        seq_masks = mask[:, :seq_len_max].bool()
        candidates_emb = multimodal_emb[:, seq_len_max:, :]
        user_repr = self.sasrec.forward_direct_batch(seq_embs, masks=seq_masks)

        zero_mask = user_repr.abs().sum(dim=-1) == 0
        if zero_mask.any():
            for idx in zero_mask.nonzero(as_tuple=True)[0]:
                valid_items_emb = seq_embs[idx, seq_masks[idx]]
                if valid_items_emb.shape[0] > 0:
                    user_repr[idx] = valid_items_emb.mean(dim=0)
                else:
                    user_repr[idx] = torch.zeros_like(user_repr[idx])

        return torch.bmm(user_repr.unsqueeze(1), candidates_emb.transpose(1, 2)).squeeze(1)

    def compute_mi_loss(self, model_output: Dict[str, torch.Tensor], mask: torch.Tensor=None) -> torch.Tensor:
        return self.mi_loss_module(
            model_output['z_img'],
            model_output['z_text'],
            model_output['z_shared'],
            mask,
        )


def compute_losses_from_output(model, model_output: Dict[str, torch.Tensor], device: torch.device, mask: torch.Tensor=None, alpha_mi: float=1.0, alpha_vae: float=1.0, alpha_sas: float=1.0):
    for key, value in model_output.items():
        if isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
            print(f'[WARN] NaN/Inf detected in model_output[{key}]')
            total_loss = torch.tensor(float('inf'), device=device, requires_grad=True)
            loss_dict = {'total_loss': float('inf'), 'vae_loss': float('inf'), 'mi_loss': 0.0, 'sas_loss': 0.0, 'weighted_vae': float('inf'), 'weighted_mi': 0.0, 'weighted_sas': 0.0}
            return total_loss, loss_dict

    vae_loss, _ = compute_vae_loss(
        mu1=model_output['mu1'],
        logvar1=model_output['logvar1'],
        mu2=model_output['mu2'],
        logvar2=model_output['logvar2'],
        mu_fused=model_output['mu_fused'],
        logvar_fused=model_output['logvar_fused'],
        mask=mask,
        beta=1.0,
    )
    mi_loss = model.compute_mi_loss(model_output, mask)
    sas_loss = torch.tensor(0.0, device=device)

    if model_output['sas_output'] is not None and model_output.get('sas_shift_labels') is not None:
        sas_logits = model_output['sas_output']
        sas_shift_labels = model_output['sas_shift_labels']
        if torch.isnan(sas_logits).any() or torch.isinf(sas_logits).any():
            print('[WARN] NaN/Inf detected in SAS logits')
            sas_loss = torch.tensor(float('inf'), device=device)
        else:
            sas_loss = F.cross_entropy(sas_logits, sas_shift_labels)

    total_loss = alpha_vae * vae_loss + alpha_mi * mi_loss + alpha_sas * sas_loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print('[WARN] Loss is NaN or Inf')
        total_loss = torch.tensor(float('inf'), device=device, requires_grad=True)

    loss_dict = {
        'total_loss': total_loss.item() if not (torch.isnan(total_loss) or torch.isinf(total_loss)) else float('inf'),
        'vae_loss': vae_loss.item() if not (torch.isnan(vae_loss) or torch.isinf(vae_loss)) else float('inf'),
        'mi_loss': mi_loss.item(),
        'sas_loss': sas_loss.item() if not (torch.isnan(sas_loss) or torch.isinf(sas_loss)) else float('inf'),
        'weighted_vae': (alpha_vae * vae_loss).item() if not (torch.isnan(vae_loss) or torch.isinf(vae_loss)) else float('inf'),
        'weighted_mi': (alpha_mi * mi_loss).item(),
        'weighted_sas': (alpha_sas * sas_loss).item() if not (torch.isnan(sas_loss) or torch.isinf(sas_loss)) else float('inf'),
    }
    return total_loss, loss_dict
