"""Small reproducible training CLI. Defaults are a synthetic smoke test, not a benchmark."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .Transformer_lm import Transformer_lm
from .AdamW import AdamW
from .cross_entropy import cross_entropy
from .training import get_batch, gradient_clipping, cosine_schedule, save_checkpoint


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    data = np.load(args.data, mmap_mode='r') if args.data else np.tile(np.arange(32), 256)
    vocab_size = args.vocab_size if args.data else 32
    if data.min() < 0 or data.max() >= vocab_size:
        raise ValueError('Dataset token IDs outside vocabulary')
    model = Transformer_lm(args.d_model, args.heads, args.d_ff, vocab_size,
                           args.context, args.layers, 10000, tie_embeddings=args.tie_embeddings,
                           num_experts=args.experts, top_k=args.top_k).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    fixed_x, fixed_y = get_batch(data, args.batch_size, args.context, args.device)
    def evaluate():
        with torch.no_grad():
            return cross_entropy(model(fixed_x), fixed_y).item()
    initial = evaluate()
    start = time.perf_counter()
    for step in range(args.steps):
        x, y = get_batch(data, args.batch_size, args.context, args.device)
        for group in optimizer.param_groups:
            group['lr'] = cosine_schedule(step, args.lr, args.lr*0.1, min(5,args.steps-1), args.steps)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(x, return_aux_loss=True)
        loss = cross_entropy(logits, y) + args.aux_weight * aux
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
    result = dict(seed=args.seed, steps=args.steps, device=args.device,
                  dataset=str(args.data) if args.data else 'synthetic repeating integers 0..31',
                  evaluation='fixed training-distribution batch; no held-out quality claim',
                  tied_embeddings=args.tie_embeddings, experts=args.experts, top_k=args.top_k,
                  parameters=sum(p.numel() for p in model.parameters()),
                  initial_loss=initial, final_loss=evaluate(), seconds=time.perf_counter()-start,
                  torch_version=torch.__version__)
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, args.steps, args.output/'checkpoint.pt')
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, help='1D .npy file of integer token IDs; otherwise synthetic smoke data')
    p.add_argument('--vocab-size', type=int, default=32000)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--d-ff', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--device', default='cpu')
    p.add_argument('--tie-embeddings', action='store_true')
    p.add_argument('--experts', type=int, default=0)
    p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--aux-weight', type=float, default=0.01)
    p.add_argument('--output', type=Path, default=Path('artifacts/smoke'))
    args=p.parse_args()
    if args.steps <= 0: p.error('--steps must be positive')
    run(args)
