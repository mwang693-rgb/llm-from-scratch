"""Decoder-only language model, optional weight tying and sparse MoE."""
import torch
from torch import nn
from .Embedding import embedding
from .Linear import Linear
from .RMSNorm import RMSNorm
from .Transformerblock import Transformerblock

class Transformer_lm(nn.Module):
    def __init__(self, d_model, num_head, d_ff, vocab_size, context_length,
                 num_layer, rope_theta, *, tie_embeddings=False, num_experts=0, top_k=2):
        super().__init__()
        if min(vocab_size, context_length, num_layer, d_model, d_ff) <= 0:
            raise ValueError('Model dimensions must be positive')
        self.Embedding = embedding(vocab_size, d_model)
        self.layer = nn.ModuleList([
            Transformerblock(d_model, num_head, d_ff, context_length, rope_theta,
                             num_experts=num_experts, top_k=top_k)
            for _ in range(num_layer)])
        self.Norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if tie_embeddings:
            self.lm_head.weights = self.Embedding.weight
        self.context_length = context_length

    def forward(self, sentence, *, return_aux_loss=False):
        if not 0 < sentence.shape[-1] <= self.context_length:
            raise ValueError('Sequence length must be in [1, context_length]')
        x = self.Embedding(sentence)
        aux = x.new_zeros(())
        for block in self.layer:
            x, block_aux = block(x, return_aux_loss=True)
            aux = aux + block_aux
        logits = self.lm_head(self.Norm(x))
        return (logits, aux / len(self.layer)) if return_aux_loss else logits

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        if temperature < 0 or (top_k is not None and top_k <= 0):
            raise ValueError('Invalid sampling parameters')
        for _ in range(max_new_tokens):
            logits = self(tokens[..., -self.context_length:])[..., -1, :]
            if temperature == 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                shape = logits.shape[:-1]
                next_token = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]), 1)
                next_token = next_token.reshape(*shape, 1)
            tokens = torch.cat((tokens, next_token), dim=-1)
        return tokens
