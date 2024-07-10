from torch import nn, Tensor
from typing import Union, Any, Union
from ..functional import residual_connection


class _Residual(nn.Module):
    def __init__(self, module: nn.Module, n_layers: int):
        super().__init__()
        self.model = nn.ModuleList([module for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
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
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        super().__init__(
            nn.LSTM(
                input_size,
                hidden_size,
                lstm_layers,
                bias,
                batch_first,
                dropout,
                bidirectional,
                proj_size,
                device,
                dtype,
            ),
            res_layers,
        )


class GRU(_Residual):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gru_layers: int,
        res_layers: int,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        super().__init__(
            nn.GRU(
                input_size,
                hidden_size,
                gru_layers,
                bias,
                batch_first,
                dropout,
                bidirectional,
                device,
                dtype,
            ),
            res_layers,
        )


__all__ = ["LSTM", "GRU"]
