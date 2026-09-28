import torch
from torch import Tensor, nn
class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean = 0, std = 1,a = -2,b = 2)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]



    