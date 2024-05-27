from torch import nn, Tensor


# Revise
class KAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        for i in range(1, in_features):
            for o in range(1, out_features):
                setattr(
                    self, f"phi{o}_{i}", nn.Linear(1, 1, bias)
                )  # Add other learnable function

    def next_step(self, x: Tensor, i: int) -> Tensor:
        out = 0
        for o in range(self.out_features):
            out += getattr(self, f"phi{o}_{i}")(x)
        return out

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor([self.next_step(x, i) for i in range(self.in_features)])
        return out
