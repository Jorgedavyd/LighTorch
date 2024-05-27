"""
Positional encodings:
- Rotary Positional Encoding (Not Learnable)
- Default Positional Encoding (Varswani, 2017)
- Taylor Encoding (Not Learnable)

"""

import torch
from torch import nn, Tensor
from datetime import timedelta
import math

"""
Rotary Positional Encoder [source](https://arxiv.org/pdf/2104.09864.pdf)

RoFormer (applied to transformer architecture to enhance performance)
Llama (applied to keys and queries before MultiQueryAttention)

I'll probably make my own implementation for manual headed attention with
directed variable attention analysis
"""


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        theta: int = 10000,
        dtype=torch.float32,
        device="cuda",
    ) -> None:
        super(RotaryPositionalEncoding, self).__init__()
        """
        Creating rotary transformation matrix

        Given the embedding space V, a linear space in R^n , there is the finite sucession {x_n}_{n=1}^N where N 
        is the number of samples onto a single sequence (sequence_length), where implicitly x_i /from V for i .
        We want to introduce a linear transformation onto this so it makes a learnable rotation into the 
        embedding space 
        """
        self.device = device

        # embedding size must be even
        assert d_model % 2 == 0, "d_model must be div by 2"
        # Create all thetas (theta_i) for i in range(0,ndim/2) theta^(-(2i)/ndim)
        theta_j = torch.tensor(
            [1 / theta ** ((2 * i) / d_model) for i in range(d_model / 2)],
            dtype=dtype,
            device=self.device,
        )
        # creates absolute position based on seq_len
        m_i = torch.arange(
            seq_len,
        )
        # creates (m_i,theta_j) matrix
        function_inputs = torch.outer(m_i, theta_j)
        # translated into polar
        self.rotary_transformation = (
            torch.polar(torch.ones_like(function_inputs), function_inputs)
            .unsqueeze(0)
            .unsqueeze(2)
        )

    def forward(
        self,
        x_n: Tensor,
    ) -> Tensor:
        # resampling input from embedding space into (batch_size, seq_len, embedding_size/2)
        # (B, N, d_model) -> (B,N,d_model/2) polar transformation
        resampled_input = torch.view_as_complex(
            x_n.float().reshape(*x_n.shape[:-1], -1, 2)
        )
        # F: ((1, N, 1, d_model/2), (B,N,H,d_model/2)) -> (B,N,H,d_model/2)
        rotated_batch = self.rotary_transformation * resampled_input
        # (B,N,H,d_model/2) -> (B,N,H, d_model/2, 2)
        rot_out = torch.view_as_real(rotated_batch)
        # (B,N,H,d_model/2, 2) -> (B,N,H,d_model)
        rot_out = rot_out.reshape(*x_n.shape)
        return rot_out.type_as(x_n).to(self.device)


"""
Dn Positional Encoding
Adds the first n-degree derivatives of the samples, creates lineal time dependence.
"""


class DnPositionalEncoding(nn.Module):
    def __init__(
        self, delta_t: timedelta, degree: int = 1, edge_order: int = 1
    ) -> None:
        super().__init__()
        self.delta_t = delta_t.total_seconds()
        self.degree = degree
        self.edge_order = edge_order

    def forward(self, x_n: Tensor) -> Tensor:
        out = x_n.clone()
        for _ in range(1, self.degree + 1):
            x_n = torch.gradient(
                x_n, spacing=(self.delta_t,), dim=-1, edge_order=self.edge_order
            )
            out += x_n
        return out


class AbsoluteSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dropout):
        super(AbsoluteSinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        # create positional encoding
        pos_embedding = torch.empty(seq_len, embed_dim)
        # change the pos_embedding to fit the functions
        for i in range(seq_len):
            for j in range(embed_dim // 2):
                pos_embedding[i, 2 * j] = math.sin(i / pow(10000, (2 * j) / embed_dim))
                pos_embedding[i, 2 * j + 1] = math.cos(
                    i / pow(10000, (2 * j) / embed_dim)
                )
        x += pos_embedding.unsqueeze(0)
        return self.dropout(x)


__all__ = [
    "AbsoluteSinusoidalPositionalEncoding",
    "RotaryPositionalEncoding",
    "DnPositionalEncoding",
]
