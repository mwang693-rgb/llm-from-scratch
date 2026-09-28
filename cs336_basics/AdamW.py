import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError('Invalid optimizer hyperparameters')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state['step'] += 1
                t, m, v = state['step'], state['m'], state['v']
                m.mul_(b1).add_(p.grad, alpha=1-b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1-b2)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                denominator = v.sqrt().div_(math.sqrt(1-b2**t)).add_(group['eps'])
                p.addcdiv_(m, denominator, value=-group['lr']/(1-b1**t))
        return loss
