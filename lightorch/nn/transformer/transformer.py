from torch import nn, Tensor
from typing import Optional, List, Sequence
from ..functional import residual_connection

"""
# Base transformer:
SelfAttention: SelfAttention module from attention (both work for decoder and encoder like architectures)
CrossAttention: CrossAttention module from attention (both work for decoder and encoder like architectures)
FFN: A feed forward network (both work for decoder and encoder like architectures)

"""

class _Transformer(nn.Module):
    def __init__(self, self_attention, cross_attention, ffn, postnorm, prenorm) -> None:
        super().__init__()
        self._self_attention = self_attention
        self._cross_attention = cross_attention
        self._ffn = ffn
        self.postnorm = postnorm if postnorm is not None else nn.Identity()
        self.prenorm = prenorm if prenorm is not None else nn.Identity()

    def _apply_sublayer(self, input: Tensor, sublayer: nn.Module, *args) -> Tensor:
        return residual_connection(input, lambda x: self.postnorm(sublayer(self.prenorm(x), *args)))

    def ffn(self, input: Tensor) -> Tensor:
        return self._apply_sublayer(input, self._ffn)
    
    def cross_attention(self, input: Tensor, cross: Tensor, is_causal) -> Tensor:
        return self._apply_sublayer(input, self._cross_attention, cross, is_causal)
        
    def self_attention(self, input: Tensor, is_causal: bool = False) -> Tensor:
        return self._apply_sublayer(input, self._self_attention, is_causal)
        

class TransformerCell(_Transformer):
    def __init__(
            self,
            *,
            self_attention: nn.Module = None,
            cross_attention: nn.Module = None,
            ffn: nn.Module = None,
            prenorm: nn.Module = None,
            postnorm: nn.Module = None
    ) -> None:
        super().__init__(self_attention, cross_attention, ffn, postnorm, prenorm)

class Transformer(nn.Module):
    def __init__(
            self, 
            embedding_layer: Optional[nn.Module] = None, 
            positional_encoding: Optional[nn.Module] = None, 
            encoder: Optional[nn.Module] = None, 
            decoder: Optional[nn.Module] = None, 
            fc: Optional[nn.Module] = None,
            n_layers: int = 1
        ) -> None:
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
        
    def forward(self, **kwargs) -> Tensor:
        if self.embedding is not None:
            out = self.embedding(**kwargs)
        if self.pe is not None:
            out = self.pe(out)
        
        if self.encoder and self.decoder:
            hist: List = []
            for encoder in self.encoder:
                out = encoder(out)
                hist.append(out)

            for cross, decoder in zip(hist, self.decoder):
                out = decoder(**kwargs, cross)
            
            out = self.fc(out)

        elif self.encoder:
            for encoder in self.encoder:
                out = encoder(out)
            
        else:
            for decoder in self.decoder:
                out = decoder(out)
            
        return out

class CrossTransformer(nn.Module):
    def __init__(self, *cells, n_layers: int, fc: nn.Module) -> None:
        assert (len(cells) == 2), 'Must be 2 transformer cells'
        self.cells = nn.ModuleList([cells for _ in range(n_layers)])
        self.fc = fc
        self.n_layers = n_layers

    def _single_forward(self, cells: Sequence[TransformerCell], layer: int, first_args: Sequence, second_args: Sequence) -> Tensor:
        out0 = cells[0].self_attention(*first_args)
        out1 = cells[1].self_attention(*second_args)

        out0 = cells[0].cross_attention(out0, out1)
        out1 = cells[1].cross_attention(out1, out0)
        
        out0 = cells[0].ffn(*out0)
        out1 = cells[1].ffn(*out1)

        return (out0, ), (out1, )
    def forward(self, first_inputs: Sequence, second_inputs: Sequence) -> Tensor:
        for layer, cells in enumerate(self.cells):
            first_inputs, second_inputs = self._single_forward(cells, layer, first_inputs, second_inputs)
        
        


__all__ = [
    'Transformer',
    'TransformerCell',
    'CrossTransformer'
]