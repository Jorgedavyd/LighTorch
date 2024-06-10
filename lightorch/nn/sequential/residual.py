from torch import nn, Tensor
from typing import Union, Tuple, Callable, Any
from ..functional import residual_connection

class _Residual(nn.Module):
    def __init__(self, module: nn.Module | Callable[[int, int], nn.Module], input_size: int, hidden_size: int, n_layers: int):
        self.model = nn.ModuleList([module(input_size, hidden_size) for _ in range(n_layers)])
    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor], Tensor]:
        for layer in self.model:
            x, _ = residual_connection(x, lambda x: layer(x))
        
        return x

class LSTM(_Residual):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lstm_layers: int,
        res_layers: int,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        super().__init__(lambda input_size, hidden_size: nn.LSTM(
            input_size,
            hidden_size,
            lstm_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype
        ), input_size, hidden_size, res_layers)        
        
class GRU(_Residual):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gru_layers: int,
        res_layers: int,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        super().__init__(lambda input_size, hidden_size: nn.GRU(
            input_size,
            hidden_size,
            gru_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device,
            dtype
        ), input_size, hidden_size, res_layers)        
        
__all__ = ['LSTM', 'GRU']