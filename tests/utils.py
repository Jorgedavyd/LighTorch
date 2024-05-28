import torch
from torch import Tensor

def create_inputs(*size) -> Tensor:
    return torch.randn(*size)