from torch import nn
from typing import Sequence, Dict


class _Loss(nn.Module):
    def __init__(
            self,
            labels: Sequence[str] | str,
            factors: Dict[str, float] | Sequence[Dict[str, float]]
    ) -> None:
        super().__init__()
        self.labels = labels.append('Overall')
        self.factors = factors

