import numpy as np
import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        self.stddev = np.sqrt(2.0 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

    def init_weight(self):
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=self.stddev,
            a=-3.0*self.stddev,
            b=-3.0*self.stddev,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W.T)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings:int,
        embedding_dim:int,
        device:torch.device|None=None,
        dtype:torch.dtype|None=None,
    ):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.init_emb()
    
    def init_emb(self):
        torch.nn.init.trunc_normal_(
            self.emb,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb[x]


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * g
    """
    def __init__(
        self,
        d_model:int,
        eps:float=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        
        # Mean of square over d_model dimension
        first_term = 1 / self.d_model * torch.sum(x ** 2, dim=-1, keepdim=True)
        denom = torch.sqrt(first_term + self.eps)
        result = (x / denom) * self.g
        return result.to(in_type)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,
    ):
        super().__init__()
        self.std = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

        self.init_weight()
    
    def init_weight(self):
        torch.nn.init.trunc_normal_(
            self.W1,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3,
        )
        torch.nn.init.trunc_normal_(
            self.W2,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3
        )
        torch.nn.init.trunc_normal_(
            self.W3,
            mean=0.0,
            std=self.std,
            a=-3,
            b=3
        )

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        first_term = self.SiLU(torch.matmul(x, self.W1.T))
        second_term = torch.matmul(x, self.W3.T)
        return torch.matmul(torch.mul(first_term, second_term), self.W2.T)



# ROPE
def rotate_angle(theta: float, d_k: int, max_seq_len: int) -> torch.Tensor:
    # theta ^ (-2i/d_k) for i = 0, 1, ..., d_k/2-1
    i = torch.arange(0, d_k // 2, dtype=torch.float32)
    frequencies = theta ** (-2 * i / d_k)

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    return torch.outer(positions, frequencies)  # shape: (max_seq_len, d_k // 2)


def get_rotary_embedding(theta: float, d_k: int, max_seq_len: int):
    angles = rotate_angle(theta, d_k, max_seq_len)
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of x by swapping and negating pairs of elements
    (x1, x2) -> (-x2, x1)
    """
    x1 = x[..., ::2]  # even indices (0, 2, 4, ...)
    x2 = x[..., 1::2]  # odd indices (1, 3, 5, ...)
    # interleave -x2 and x1
    rotated = torch.stack((-x2, x1), dim=-1)  # shape (..., d_k // 2, 2)
    return rotated.flatten(start_dim=-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin:torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class ROPE(nn.Module):
    def __init__(
        self,
        theta:float,
        d_k:int,
        max_seq_len:int,
        device=None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        cos, sin = get_rotary_embedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions, ...]
        sin = self.sin_cached[token_positions, ...]
        return apply_rotary_pos_emb(x, cos, sin)


if __name__ == "__main__":
    d_k = 64
    max_seq_len = 512
    angles = rotate_angle(10000.0, d_k, max_seq_len)
    x = torch.rand(2, 6)
    print(x)
    print(get_half_seq(x))