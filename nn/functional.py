import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple
import torch.nn.functional as F
    
def fourierconvNd(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    out = (x.real * weight.real) + 1j * (x.imag * weight.imag)
    if bias is not None:
        out += bias
    return out

def partialconv3d(input: Tensor, mask_in: Tensor, weight: Tensor, one: Tensor, bias: Optional[Tensor], stride, padding, dilation, update_mask: bool = True) -> Union[Tuple[Tensor, Tensor], Tensor]:
    
    with torch.no_grad():
        
        sum = F.conv3d(mask_in, torch.ones_like(weight, requires_grad=False), stride = stride, padding = padding, dilation = dilation)
        
        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)
    
    out = F.conv3d(input*mask_in, weight, None, stride, padding, dilation)
    
    out = out * (one/sum) + bias

    return (out, updated_mask) if update_mask else out


def partialconv2d(input: Tensor, mask_in: Tensor, weight: Tensor, one: Tensor, ones_weight: Tensor, bias: Optional[Tensor], stride, padding, dilation, update_mask: bool = True) -> Union[Tuple[Tensor, Tensor], Tensor]:
    
    with torch.no_grad():
        
        sum = F.conv2d(mask_in, ones_weight, stride = stride, padding = padding, dilation = dilation)
        
        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)

    out = F.conv2d(input, weight, None, stride, padding, dilation)

    out = out * (one/sum) + bias

    return (out, updated_mask) if update_mask else out

def partialconv1d(input: Tensor, mask_in: Tensor, weight: Tensor, one: Tensor, bias: Optional[Tensor], stride, padding, dilation, update_mask: bool = True) -> Union[Tuple[Tensor, Tensor], Tensor]:
    
    with torch.no_grad():
        
        sum = F.conv1d(mask_in, torch.ones_like(weight, requires_grad=False), stride = stride, padding = padding, dilation = dilation)
        
        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)
    
    out = F.conv1d(input*mask_in, weight, None, stride, padding, dilation)
    
    out = out * (one/sum) + bias

    return (out, updated_mask) if update_mask else out
