import pytest
from torch import Tensor
import torch
from ...lightorch.nn.functional import *
from ..utils import create_inputs


# Test automation for functionals


class PartialConv


def partialconv() -> None:
    
    # 1D- Partial Convolutional Neural Network
    input: Tensor = create_inputs(1, 30, 10) # batch size, channel size, sequence length 
    mask_in: Tensor = create_inputs(1, 30, 10).clamp(0, 1) # batch size, channel size, sequence length 
    weight: Tensor = create_inputs(40, 30, 5) # out channels, in channels, kernel size
    one: Tensor = torch.ones_like()
    
    out, mask_in = partialconv1d(input, mask_in, weight, one, bias, stride = 1, padding = 1, update_mask = True)
