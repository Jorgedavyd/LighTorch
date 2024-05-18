from torch import nn, Tensor
import functional as F
from torch.fft import fftn, ifftn
import torch
from torch.nn import init
from math import sqrt

class _FourierConvNd(nn.Module):
    def __init__(
            self,
            *kernel_size,
            bias: bool = True,
            pre_fft: bool = True,
            post_ifft: bool = False,
            device = None,
            dtype = None,
    ) -> None:
        super().__init__()
        
        self.factory_kwargs = {
            'device': device,
            'dtype': dtype
        }

        if pre_fft:
            self.fft = fftn

        if post_ifft:
            self.ifft = ifftn

        self.weight = nn.Parameter(torch.empty(*kernel_size, **self.factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(kernel_size[0], **self.factory_kwargs))
        else:
            self.bias = None

        self._init_parameters()
        self._fourier_space(len(kernel_size))
        
    def _fourier_space(self, dims: int) -> Tensor:
        if self.bias is not None:
            self.bias = self.fft(self.bias, dim = (-i for i in range(1, dims)))
        self.weight = self.fft(self.weight, dim = (-i for i in range(1, dims)))
    
    def _init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input: Tensor) -> Tensor:
        return F.fourierconvNd(input, self.weight, self.bias)

class FourierConv1d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input, dim = -1)
        out = super(FourierConv1d, self).forward(input)
        if self.ifft:
            out = self.ifft(out, dim = -1)
        return out
    
class FourierConv2d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input, dim = (-2, -1))
        out = super(FourierConv1d, self).forward(input)
        if self.ifft:
            out = self.ifft(out, dim = (-2, -1))
        return out
    

class FourierConv3d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input, dim = (-3, -2, -1))
        out = super(FourierConv1d, self).forward(input)
        if self.ifft:
            out = self.ifft(out, dim = (-3, -2, -1))
        return out


__all__ = ['FourierConv1d', 'FourierConv2d', 'FourierConv3d']

