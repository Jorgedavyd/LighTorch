from torch import nn, Tensor
from typing import Callable


class _DefaultFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        k_multiplier: int,
        out_features: int,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, in_features * k_multiplier, False)
        self.w2 = nn.Linear(in_features * k_multiplier, out_features, False)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)))


class FFN_ReLU(_DefaultFFN):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_Sigmoid(_DefaultFFN):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_Swish(_DefaultFFN):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_GELU(_DefaultFFN):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_Bilinear(_DefaultFFN):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class _GLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, out_features, True)
        self.w2 = nn.Linear(in_features, out_features, True)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.w1(x)) * self.w2(x)


class BiGLU(_GLU):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward()


class GLU(_GLU):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class ReGLU(_GLU):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class GEGLU(_GLU):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class SiGLU(_GLU):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class _GLU_variants(nn.Module):
    def __init__(
        self,
        in_features: int,
        k_multiplier: int,
        out_features: int,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, in_features * k_multiplier, bias=False)
        self.v = nn.Linear(in_features, in_features * k_multiplier, bias=False)
        self.w2 = nn.Linear(in_features * k_multiplier, out_features)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.v(x))


class FFN_SwiGLU(_GLU_variants):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_ReGLU(_GLU_variants):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_GEGLU(_GLU_variants):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_Bilinear(_GLU_variants):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FFN_GLU(_GLU_variants):
    def __init__(self, in_features: int, k_multiplier: int, out_features: int) -> None:
        super().__init__(in_features, k_multiplier, out_features, nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


__all__ = [
    "FFN_ReLU",
    "FFN_Bilinear",
    "FFN_Sigmoid",
    "FFN_Swish",
    "FFN_GELU",
    "BiGLU",
    "GLU",
    "ReGLU",
    "GEGLU",
    "SiGLU",
    "FFN_SwiGLU",
    "FFN_ReGLU",
    "FFN_GEGLU",
    "FFN_Bilinear",
    "FFN_GLU",
]
