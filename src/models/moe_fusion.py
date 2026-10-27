from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionExpert(nn.Module):
    def __init__(self, input_dim: int, n_tokens: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.input_features = input_dim * n_tokens
        self.net = nn.Sequential(nn.Linear(self.input_features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.net(x.view(batch_size, -1))


class GroupGatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x.mean(dim=1)), dim=-1)


class MoEVecFusionLayer(nn.Module):
    def __init__(self, input_dim: int, n_tokens: int, latent_dim: int, num_experts: int, top_k: int=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.gating = GroupGatingNetwork(input_dim, num_experts)
        self.experts = nn.ModuleList([FusionExpert(input_dim=input_dim, n_tokens=n_tokens, output_dim=latent_dim, hidden_dim=input_dim * n_tokens // 2) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_tokens, dim = x.shape
        assert n_tokens == self.n_tokens, f'Input N ({n_tokens}) must match initialized n_tokens ({self.n_tokens})'
        assert self.top_k <= self.num_experts, f'top_k ({self.top_k}) must <= num_experts ({self.num_experts})'

        x_reshaped = x.reshape(-1, n_tokens, dim)
        gate_weights = self.gating(x_reshaped)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-09)

        expert_results = torch.stack([expert(x_reshaped) for expert in self.experts], dim=1)
        output_dim = expert_results.size(-1)
        gather_index = top_k_indices.unsqueeze(-1).expand(-1, -1, output_dim)
        selected = expert_results.gather(dim=1, index=gather_index)
        final_fused = (selected * top_k_weights.unsqueeze(-1)).sum(dim=1)
        final_output = final_fused.view(batch_size, seq_len, output_dim)

        p_expert_mean = gate_weights.mean(dim=0)
        uniform = gate_weights.new_full((self.num_experts,), 1.0 / self.num_experts)
        moe_loss = F.mse_loss(p_expert_mean, uniform)
        return final_output, moe_loss
