from torch import nn, Tensor
from typing import Optional, List, Sequence, Tuple
from ..functional import residual_connection
from .attention import SelfAttention, CrossAttention

"""
# Base transformer:
SelfAttention: SelfAttention module from attention (both work for decoder and encoder like architectures)
CrossAttention: CrossAttention module from attention (both work for decoder and encoder like architectures)
FFN: A feed forward network (both work for decoder and encoder like architectures)

"""


class _Transformer(nn.Module):
    def __init__(
        self,
        self_attention: SelfAttention,
        cross_attention: CrossAttention,
        ffn: nn.Module,
        postnorm: nn.Module,
        prenorm: nn.Module,
    ) -> None:
        super().__init__()
        self._self_attention = self_attention
        self._cross_attention = cross_attention
        self._ffn = ffn
        self.postnorm = postnorm if postnorm is not None else nn.Identity()
        self.prenorm = prenorm if prenorm is not None else nn.Identity()

    def _apply_sublayer(self, input: Tensor, sublayer: nn.Module, *args) -> Tensor:
        return residual_connection(
            input, lambda x: self.postnorm(sublayer(self.prenorm(x), *args))
        )

    def ffn(self, input: Tensor) -> Tensor:
        return self._apply_sublayer(input, self._ffn)

    def cross_attention(self, input: Tensor, cross: Tensor) -> Tensor:
        return self._apply_sublayer(input, self._cross_attention, cross)

    def self_attention(self, input: Tensor) -> Tensor:
        return self._apply_sublayer(input, self._self_attention)


class TransformerCell(_Transformer):
    def __init__(
        self,
        *,
        self_attention: SelfAttention = None,
        cross_attention: CrossAttention = None,
        ffn: nn.Module = None,
        prenorm: nn.Module = None,
        postnorm: nn.Module = None
    ) -> None:
        super().__init__(self_attention, cross_attention, ffn, postnorm, prenorm)

    def forward(self, x: Tensor, cross: Optional[Tensor] = None) -> Tensor:
        if self._self_attention is not None:
            x = self.self_attention(x)

        if self._cross_attention is not None and cross is not None:
            x = self.cross_attention(x, cross)

        if self._ffn is not None:
            x = self.ffn(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_layer: Optional[nn.Module] = None,
        positional_encoding: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        fc: Optional[nn.Module] = None,
        n_layers: int = 1,
    ) -> None:
        assert (
            encoder is not None or decoder is not None
        ), "Not valid parameters, must be at least one encoder or decoder."
        super().__init__()
        self.embedding = embedding_layer
        self.pe = positional_encoding
        if encoder is not None:
            self.encoder = nn.ModuleList([encoder for _ in range(n_layers)])
        else:
            self.encoder = False
        if decoder is not None:
            self.decoder = nn.ModuleList([decoder for _ in range(n_layers)])
        else:
            self.decoder = False
        self.n_layers = n_layers
        self.fc = fc

    def forward(self, x: Tensor) -> Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        if self.pe is not None:
            x = self.pe(x)

        if self.encoder and self.decoder:
            for encoder, decoder in zip(self.encoder, self.decoder):
                x = encoder(x)
                out = decoder(x)

        elif self.encoder:
            for encoder in self.encoder:
                out = encoder(x)

        else:
            for decoder in self.decoder:
                out = decoder(x)
        if self.fc:
            x = self.fc(out)

        return x


class CrossTransformer(nn.Module):
    def __init__(
        self,
        cell_1: TransformerCell,
        cell_2: TransformerCell,
        n_layers: int,
        fc: nn.Module,
    ) -> None:
        super().__init__()
        self.cell_1 = nn.ModuleList([cell_1 for _ in range(n_layers)])
        self.cell_2 = nn.ModuleList([cell_2 for _ in range(n_layers)])
        self.fc = fc
        self.n_layers = n_layers

    def _single_forward(
        self,
        cell_1: TransformerCell,
        cell_2: TransformerCell,
        head_1: Tensor,
        head_2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out0 = cell_1.self_attention(head_1)
        out1 = cell_2.self_attention(head_2)

        out0 = cell_1.cross_attention(out0, out1)
        out1 = cell_2.cross_attention(out1, out0)

        out0 = cell_1.ffn(out0)
        out1 = cell_2.ffn(out1)

        return out0, out1

    def forward(self, head_1: Tensor, head_2: Tensor) -> Tuple[Tensor, Tensor]:
        for cell_1, cell_2 in zip(self.cell_1, self.cell_2):
            head_1, head_2 = self._single_forward(cell_1, cell_2, head_1, head_2)

        return head_1, head_2


__all__ = ["Transformer", "TransformerCell", "CrossTransformer"]
