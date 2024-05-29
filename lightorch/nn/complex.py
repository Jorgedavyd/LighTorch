from torch import nn, Tensor
from copy import deepcopy


class Complex(nn.Module):
    """
    # Complex
    Module to transform non-complex operators.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.Re_mod = module
        self.Im_mod = deepcopy(module)

    def forward(self, x: Tensor) -> Tensor:
        return self.Re_mod(x.real) + 1j * self.Im_mod(x.imag)


__all__ = ["Complex"]
