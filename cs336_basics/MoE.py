"""Correctness-oriented top-k sparse dispatch; no capacity dropping or fused kernels."""
import torch
from torch import nn
from .Linear import Linear
from .SwiGLU import SwiGLU

class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError('Require 1 <= top_k <= num_experts')
        self.num_experts, self.top_k = num_experts, top_k
        self.router = Linear(d_model, num_experts)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        probabilities = self.router(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel():
                values = expert(flat[token_ids]) * weights[token_ids, slots, None].to(flat.dtype)
                out.index_add_(0, token_ids, values)
        frequency = torch.bincount(indices.flatten(), minlength=self.num_experts).float()
        frequency = frequency / indices.numel()
        aux = self.num_experts * (frequency.detach() * probabilities.mean(0)).sum()
        return out.reshape_as(x), aux
