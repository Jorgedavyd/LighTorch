from torch import nn, Tensor
import torch

"""
Root Mean Squared Normalization (https://arxiv.org/pdf/1910.07467.pdf)

from summary:

Extensive experiments on several tasks using diverse network architectures 
show that RMSNorm achieves comparable performanceagainst LayerNorm but reduces 
the running time by 7%-64% on different models
"""


class RootMeanSquaredNormalization(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(RootMeanSquaredNormalization, self).__init__()
        self.eps = eps
        self.g_i = nn.Parameter(torch.ones(dim))

    def forward(self, input: Tensor) -> Tensor:
        # RMSN(x_i) = g_i*(x_i/(RMSE(x_i) + eps))
        return self.g_i * (
            input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + self.eps)
        )
