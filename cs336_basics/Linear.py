import torch
from torch import Tensor, nn
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, mean=0., std=std, a=-3 * std, b=3 * std)
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights.T