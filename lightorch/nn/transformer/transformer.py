from torch import nn, Tensor
from typing import Optional, List

class GeneralTransformer(nn.Module):
    def __init__(self, *, encoder: Optional[nn.Module], decoder: Optional[nn.Module], n_layers: int) -> None:
        super().__init__()
        if encoder is not None:

            self.encoders = nn.Sequential(
                *[
                    encoder for _ in range(n_layers)
                ]
            )
        else:
            self.encoders = False

        if decoder is not None:
            self.decoders = nn.Sequential(
                *[
                    decoder for _ in range(n_layers)
                ]
            )
        else:
            self.decoders = False
    def encoder_forward(self, x: Tensor) -> Tensor:
        hist: List[nn.Module] = []
        for module in self.encoders:
            x = module(x)
            hist.append(x)
        return hist
    def forward(self, x: Tensor) -> Tensor:
        if self.encoders:
            hist = self.encoder_forward(x)
        # Continue
            
        

class CrossTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, n_layers: int) -> None:
        pass