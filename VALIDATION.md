# Validation — September 28, 2026

## Correctness

`python -X utf8 -m pytest -q`: **52 passed, 2 skipped** on Windows CPU with PyTorch 2.11.0.
The two skipped tests require POSIX resource limits; the reference test already skips one, and the release adds an explicit Windows skip to the other. No assertions or expected snapshots were weakened.

## Synthetic trainability checks

80 optimizer steps, seed 42, 2 layers, model width 32, 4 attention heads, FFN width 64, vocabulary 32, context 16, batch size 8. Inputs repeat integer token IDs 0 through 31. Evaluation uses a fixed batch from the same synthetic distribution.

| Configuration | Parameters | Initial loss | Final loss |
|---|---:|---:|---:|
| dense | 22,688 | 3.657898 | 0.061253 |
| tied | 21,664 | 18.661610 | 0.002393 |
| moe | 58,784 | 19.896587 | 0.000749 |

Commands: run `python -m cs336_basics.train --steps 80`, adding `--tie-embeddings` for tied dense, and `--tie-embeddings --experts 4 --top-k 2` for MoE. Use separate `--output` paths.

These are trainability checks, not a controlled quality comparison: the parameter counts and initializations differ, and the task is trivial. The measurements do not establish language modeling quality, generalization, routing balance, GPU throughput, or MoE superiority. Weight tying removes exactly one vocabulary-by-model-width matrix (1,024 parameters here).

Raw metrics are in `validation/`. Large-corpus training, held-out perplexity, GPU profiling, mixed-precision stress tests, and distributed expert parallelism remain future work.
