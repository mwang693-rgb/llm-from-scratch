# Language Model from Scratch

A compact PyTorch implementation of a decoder-only language model, with dense and sparse mixture-of-experts variants.

This is Moyang Wang's independent study project based on [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics). It is not a Stanford affiliation or an official course submission. The September 2026 portfolio release builds on an existing learning implementation and includes AI-assisted repairs, completion, tests, and extensions. See [PROVENANCE.md](PROVENANCE.md).

## Architecture

```text
token IDs → byte/token embeddings
  → [RMSNorm → causal multi-head attention with RoPE → residual
     RMSNorm → SwiGLU or top-k sparse MoE → residual] × layers
  → RMSNorm → vocabulary projection → next-token logits
```

- Bias-free linear projections and RMSNorm; RoPE on queries and keys.
- **Optional weight tying:** input embeddings and output projection share the same Parameter.
- **Optional sparse MoE:** token-level top-k routing over independent SwiGLU experts, normalized routing weights, and a load-balancing auxiliary loss.
- Byte-level BPE training and tokenization, including protected special tokens.
- Stable cross-entropy, AdamW, cosine learning-rate scheduling, clipping, checkpointing, and sampling.

The sparse dispatch is written for clarity and correctness. It is not a fused or distributed MoE implementation; no throughput improvement is claimed. The model does not yet implement incremental KV-cached decoding.

## Quick start

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python -X utf8 -m pytest -q
uv run python -m cs336_basics.train \
  --steps 80 \
  --tie-embeddings \
  --experts 4 \
  --top-k 2 \
  --output artifacts/moe
```

Without `--data`, training uses a tiny synthetic repeating sequence to verify that optimization works. Supply a one-dimensional `.npy` token-ID array and `--vocab-size` for your own corpus. No training data or pretrained checkpoints are bundled.

```python
from cs336_basics.Transformer_lm import Transformer_lm

model = Transformer_lm(
    d_model=256, num_head=8, d_ff=688,
    vocab_size=10000, context_length=512,
    num_layer=4, rope_theta=10000,
    tie_embeddings=True, num_experts=4, top_k=2,
)
logits, balance_loss = model(token_ids, return_aux_loss=True)
# Combine token cross-entropy with a small coefficient times balance_loss.
```

## Validation

On Windows CPU / PyTorch 2.11.0: **52 tests passed, 2 skipped**. The skipped tests require POSIX resource limits; no memory-bound guarantee is inferred from them. Reference tests cover the dense model, tokenization, BPE training, numerical utilities, AdamW, batching, and checkpoints. Extension tests check MoE equivalence to a dense reference and its gradients, causal masking, tied parameters, and serialization.

Three 80-step synthetic training runs completed for dense, tied-dense, and tied-MoE configurations. [VALIDATION.md](VALIDATION.md) records exact parameters, losses, and limitations. These runs demonstrate trainability only, not useful language quality or comparative model performance.

## Repository guide

- `cs336_basics/Transformer_lm.py`: model and sampling.
- `cs336_basics/Transformerblock.py`: pre-norm residual block.
- `cs336_basics/MoE.py`: sparse routing and auxiliary loss.
- `cs336_basics/Tokenizer.py`, `bpe_training.py`: byte-level BPE.
- `cs336_basics/train.py`: reproducible training CLI.
- `tests/`: course-derived reference tests plus extension tests.

## Attribution

Course scaffolding and reference tests are derived from Stanford CS336 and retain the MIT license in `LICENSE`. Architectural extensions and integration are documented separately in `PROVENANCE.md`.
