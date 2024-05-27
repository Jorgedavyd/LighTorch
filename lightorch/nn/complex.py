from torch import nn, Tensor


class Complex(nn.Module):
    """
    # Complex
    Module to transform non-complex operators.
    """

    def __init__(self, module: nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.Re_mod = module(*args, **kwargs)
        self.Im_mod = module(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.Re_mod(x.real) + 1j * self.Im_mod(x.imag)


__all__ = ["Complex"]
