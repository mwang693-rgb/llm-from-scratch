from torch import nn
from .multihead_self_attention import multihead_self_attention
from .RMSNorm import RMSNorm
from .SwiGLU import SwiGLU
from .MoE import SparseMoE

class Transformerblock(nn.Module):
    def __init__(self, d_model, num_head, d_ff, max_seq_len, theta, *, num_experts=0, top_k=2):
        super().__init__()
        self.RMSNorm1 = RMSNorm(d_model)
        self.RMSNorm2 = RMSNorm(d_model)
        self.multilayer = multihead_self_attention(d_model, num_head, theta, max_seq_len)
        self.FFNlayer = SparseMoE(d_model, d_ff, num_experts, top_k) if num_experts else SwiGLU(d_model, d_ff)

    def forward(self, x, *, return_aux_loss=False):
        x = x + self.multilayer(self.RMSNorm1(x))
        normalized = self.RMSNorm2(x)
        if isinstance(self.FFNlayer, SparseMoE):
            update, aux = self.FFNlayer(normalized)
        else:
            update, aux = self.FFNlayer(normalized), x.new_zeros(())
        x = x + update
        return (x, aux) if return_aux_loss else x
