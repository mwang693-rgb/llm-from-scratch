import torch
from torch import nn
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        assert d_k > 0 and d_k % 2 == 0
        self.d_k = d_k

        # 每对特征只需要一个角度，所以列数是 d_k // 2。
        finaltheta = torch.zeros(
            max_seq_len, d_k // 2,
            dtype=torch.float32,
            device=device,
        )

        for i in range(max_seq_len):
            for d in range(d_k // 2):
                finaltheta[i, d] = i / (theta ** ((2 * d) / d_k))

        # 实际使用的是这两张表，所以把它们注册为 buffer。
        self.register_buffer("cosangle", torch.cos(finaltheta))
        self.register_buffer("sinangle", torch.sin(finaltheta))

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # 约定：
        # x：(..., sequence, d_k)
        # token_positions：(..., sequence)
        # 前面的维度需要能互相广播。
        token_positions = token_positions.to(
            device=x.device, dtype=torch.long
        )

        # 只取本次 token 对应的位置，而不是使用整张缓存表。
        cos = self.cosangle[token_positions].to(dtype=x.dtype)
        sin = self.sinangle[token_positions].to(dtype=x.dtype)

        inputx = rearrange(x, "... (a b) -> ... a b", b=2)

        new_first = cos * inputx[..., 0] - sin * inputx[..., 1]
        new_second = sin * inputx[..., 0] + cos * inputx[..., 1]

        rotated_pairs = torch.stack(
            (new_first, new_second), dim=-1
        )

        output = rearrange(
            rotated_pairs,
            "... pairs components -> ... (pairs components)",
        )
        return output




