from torch import nn, Tensor
import torch
import torch.nn.functional as F


class MonteCarloFC(nn.Module):
    def __init__(
        self, fc_layer: nn.Module, dropout: float = 0.2, n_sampling: int = 5
    ) -> None:
        super().__init__()
        self.n_sampling = n_sampling
        self.fc = fc_layer
        self.dropout = lambda x: F.dropout(x, dropout, True)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for _ in range(self.n_sampling):
            x = self.dropout(x)
            outputs.append(self.fc(x))
        out = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return out
