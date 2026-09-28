import math
import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    if dataset.ndim != 1 or len(dataset) <= context_length or min(batch_size, context_length) <= 0:
        raise ValueError('Need a 1D dataset longer than the positive context length')
    starts = np.random.randint(0, len(dataset)-context_length, size=batch_size)
    positions = starts[:, None] + np.arange(context_length)[None, :]
    return (torch.as_tensor(np.asarray(dataset[positions], dtype=np.int64), device=device),
            torch.as_tensor(np.asarray(dataset[positions+1], dtype=np.int64), device=device))

@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if max_l2_norm < 0:
        raise ValueError('max_l2_norm must be nonnegative')
    if grads:
        norm = torch.stack([g.float().norm() for g in grads]).norm()
        scale = (max_l2_norm / (norm + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(scale.to(g.device))

def cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if not 0 <= warmup_iters < cosine_cycle_iters:
        raise ValueError('Require 0 <= warmup < cycle')
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)

def save_checkpoint(model, optimizer, iteration, out):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration), out)

def load_checkpoint(src, model, optimizer):
    state = torch.load(src, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['iteration']
