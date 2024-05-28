import pytest
from torch import Tensor
import torch

input: Tensor = torch.randn(1, 32)
target: Tensor = torch.randn(1, 32)

## Test automation for criterions

def tv()