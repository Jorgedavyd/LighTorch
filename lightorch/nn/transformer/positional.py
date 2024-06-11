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
    ) -> None:
        super(RotaryPositionalEncoding, self).__init__()
        """
        Creating rotary transformation matrix

        Given the embedding space V, a linear space in R^n , there is the finite sucession {x_n}_{n=1}^N where N 
        is the number of samples onto a single sequence (sequence_length), where implicitly x_i /from V for i .
        We want to introduce a linear transformation onto this so it makes a learnable rotation into the 
        embedding space 
        """
        # embedding size must be even
        assert d_model % 2 == 0, "d_model must be div by 2"
        self.theta_numerator = torch.arange(0, d_model, 2).float()
        self.theta_j = 1.0 / (theta ** (self.theta_numerator / d_model))  # (Dim / 2)
        # creates absolute position based on seq_len
        self.m_i = torch.arange(seq_len)
        # creates (m_i,theta_j) matrix
        function_inputs = torch.outer(self.m_i, self.theta_j).float()
        # translated into polar
        self.freqs_complex = torch.polar(
            torch.ones_like(function_inputs), function_inputs
        )

    def forward(self, x) -> Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        freqs_complex = self.freqs_complex.unsqueeze(0)
        # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        x_rotated = x_complex * freqs_complex
        # Convert the complex number back to the real number
        # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        x_out = torch.view_as_real(x_rotated)
        # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x)


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
            (x_n,) = torch.gradient(
                x_n, spacing=(self.delta_t,), dim=-1, edge_order=self.edge_order
            )
            out += x_n
        return out


class AbsoluteSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dropout):
        super(AbsoluteSinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
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
