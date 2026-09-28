import torch
from einops import rearrange
from torch import Tensor, nn

from .Attention import scaled_dot_product_attention
from .Linear import Linear


class multihead_self_attention(nn.Module):
    """输入与输出均为 (..., sequence, d_model)，支持没有 batch 维的输入。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta=None,
        max_seq=None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if (theta is None) != (max_seq is None):
            raise ValueError("Provide both theta and max_seq, or neither")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 子层保存在 self 上；Q/K/V 等本次计算结果则留在 forward 内。
        self.WQ = Linear(d_model, d_model, dtype=dtype, device=device)
        self.WK = Linear(d_model, d_model, dtype=dtype, device=device)
        self.WV = Linear(d_model, d_model, dtype=dtype, device=device)
        self.WO = Linear(d_model, d_model, dtype=dtype, device=device)

        self.rope = None
        if theta is not None:
            if self.d_k % 2 != 0:
                raise ValueError("RoPE requires an even head dimension")
            # 约定该模块接收 (..., sequence, d_k) 及对应的位置编号。
            from .RoPE import RotaryPositionalEmbedding

            self.rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len=max_seq, device=device
            )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("Expected input shape (..., sequence, d_model)")
        seq_len = x.shape[-2]

        # 投影后仍为 (..., sequence, d_model)。
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        # 拆最后的特征轴，并让 Attention 的最后两轴为 sequence、head_dim。
        pattern = "... sequence (heads head_dim) -> ... heads sequence head_dim"
        q = rearrange(Q, pattern, heads=self.num_heads)
        k = rearrange(K, pattern, heads=self.num_heads)
        v = rearrange(V, pattern, heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # 先匹配输入的 (..., sequence)，再插入长度为 1 的头轴。
            # (..., 1, sequence) 使每个头使用相同的位置编号。
            token_positions = token_positions.to(device=x.device)
            token_positions = torch.broadcast_to(token_positions, x.shape[:-1])
            positions_per_head = token_positions.unsqueeze(-2)
            q = self.rope(q, positions_per_head)
            k = self.rope(k, positions_per_head)

        # True 表示允许关注；下三角包含当前位置及之前的 token。
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn = scaled_dot_product_attention(q, k, v, mask=mask)

        # (..., heads, sequence, head_dim) -> (..., sequence, d_model)
        attn = rearrange(attn, "... heads sequence head_dim -> ... sequence (heads head_dim)")
        return self.WO(attn)


