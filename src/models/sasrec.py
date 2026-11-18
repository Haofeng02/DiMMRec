import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, hidden_units: int, max_seq_len: int=50, num_blocks: int=2, num_heads: int=4, dropout_rate: float=0.2):
        super().__init__()
        self.hidden_units = hidden_units
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads, dim_feedforward=hidden_units * 4, dropout=dropout_rate, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_units)

    @staticmethod
    def _right_pad_valid_steps(seq_embs, masks):
        """Compact valid steps to the left while preserving their order."""
        _, seq_len, _ = seq_embs.shape
        device = seq_embs.device
        lengths = masks.sum(dim=1).long()
        working_masks = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        working_embs = torch.zeros_like(seq_embs)

        batch_indices, source_indices = masks.nonzero(as_tuple=True)
        logical_positions = masks.long().cumsum(dim=1) - 1
        target_indices = logical_positions[batch_indices, source_indices]
        working_embs[batch_indices, target_indices] = seq_embs[batch_indices, source_indices]
        return working_embs, working_masks, batch_indices, source_indices, target_indices

    def forward_all_steps(self, seq_embs, masks=None):
        """Encode every sequence step with causal and padding attention masks."""
        batch_size, seq_len, _ = seq_embs.shape
        device = seq_embs.device
        restore_indices = None

        if masks is not None:
            masks = masks.to(device=device, dtype=torch.bool)
            (
                seq_embs,
                working_masks,
                batch_indices,
                source_indices,
                target_indices,
            ) = self._right_pad_valid_steps(seq_embs, masks)
            restore_indices = (batch_indices, source_indices, target_indices)

            # A fully padded row needs one harmless working position so every
            # attention query has at least one visible key. It restores to zero.
            safe_masks = working_masks.clone()
            empty_rows = ~safe_masks.any(dim=1)
            safe_masks[empty_rows, 0] = True
            key_padding_mask = ~safe_masks
        else:
            key_padding_mask = None

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        seq_embs = seq_embs + self.position_embeddings(position_ids)
        seq_embs = self.layer_norm(seq_embs)
        seq_embs = self.dropout(seq_embs)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        working_out = self.encoder(
            seq_embs,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        if restore_indices is None:
            return working_out

        batch_indices, source_indices, target_indices = restore_indices
        seq_out = torch.zeros_like(working_out)
        seq_out[batch_indices, source_indices] = working_out[batch_indices, target_indices]
        return seq_out

    def forward(self, seq_items, multimodal_emb, unique_items_flat=None, item_to_index=None):
        seq_embs = multimodal_emb[seq_items]
        masks = seq_items.ne(0)
        return self.forward_direct_batch(seq_embs, masks=masks)

    def forward_direct_batch(self, seq_embs, masks=None):
        batch_size, seq_len, _ = seq_embs.shape
        device = seq_embs.device
        seq_out = self.forward_all_steps(seq_embs, masks=masks)

        if masks is None:
            return seq_out[:, -1, :]

        masks = masks.to(device=device, dtype=torch.bool)
        physical_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        last_valid_positions = physical_positions.masked_fill(~masks, -1).max(dim=1).values
        valid_rows = last_valid_positions.ge(0)
        safe_positions = last_valid_positions.clamp_min(0)
        batch_indices = torch.arange(batch_size, device=device)
        user_repr = seq_out[batch_indices, safe_positions]
        return user_repr.masked_fill(~valid_rows.unsqueeze(-1), 0.0)
