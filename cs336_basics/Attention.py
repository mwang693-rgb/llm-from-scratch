import torch
import math
from .Softmax import softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None  
) -> torch.Tensor:

    d_k = Q.size(-1)
    scores = torch.einsum('...nk, ...mk -> ...nm', Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    probs = softmax(scores, -1)

    output = torch.einsum('...nm,...mk -> ...nk', probs, V)

    return output