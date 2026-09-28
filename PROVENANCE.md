# Implementation provenance

## Existing learning implementation

Before the portfolio release, the local study checkout contained embedding and linear layers, RMSNorm, RoPE, attention, SwiGLU, Transformer blocks, a language-model skeleton with tied output embeddings, and partial tokenizer/optimizer/loss work. These files had accumulated independent study and prior assistance. No claim is made that every line was written without tools.

## September 28, 2026 portfolio release

With explicit authorization, an AI coding assistant worked in a separate copy and:

- fixed repeated attention evaluation in the residual block, initialization, and unfinished loss/training utilities;
- made tied versus untied embeddings configurable while retaining dense reference compatibility;
- added top-k sparse-MoE dispatch, balancing loss, tests, and training integration;
- completed byte-level BPE/tokenizer behavior, AdamW, scheduling, clipping, and checkpointing;
- ran reference and extension tests and synthetic trainability checks;
- prepared documentation and a public portfolio release.

The original local learning checkout was not edited. The public release is an educational implementation, not a trained foundation model or a novel architecture claim.

## Test portability

The tokenizer reference test has a Windows guard that explicitly skips POSIX resource-limit checks when `resource` is unavailable. Run Python with `-X utf8` on Windows to read the upstream UTF-8 fixtures correctly. Numerical expected outputs were not changed.
