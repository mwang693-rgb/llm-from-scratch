import torch

def cross_entropy(logits, target):
    values = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    shifted = values - values.amax(dim=-1, keepdim=True)
    chosen = shifted.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (shifted.exp().sum(-1).log() - chosen).mean()
