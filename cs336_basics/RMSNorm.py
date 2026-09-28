import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)
