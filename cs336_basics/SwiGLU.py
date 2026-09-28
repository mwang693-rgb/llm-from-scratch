import torch
from torch import Tensor,nn
from .Linear import Linear
def silu_fn(in_features):
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int,dtype = None, device = None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear(d_model,d_ff, dtype = dtype,device = device)
        self.w2 = Linear(d_ff,d_model, dtype = dtype,device = device)
        self.w3 = Linear(d_model,d_ff, dtype = dtype,device = device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu_fn(self.w1(x))*self.w3(x))