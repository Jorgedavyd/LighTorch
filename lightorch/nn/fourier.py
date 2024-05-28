from torch import nn, Tensor
from . import functional as F
from torch.fft import fftn, ifftn
import torch
from torch.nn import init
from math import sqrt
import torch.nn.functional as f
from typing import Tuple

class _FourierConvNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *kernel_size,
        padding: Tuple[int],
        bias: bool = True,
        eps: float = 1e-5,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.padding = padding
        if pre_fft:
            self.fft = lambda x: fftn(x, dim=(-i for i in range(1, len(kernel_size))))
        else:
            self.fft = False
        if post_ifft:
            self.ifft = lambda x: ifftn(x, dim=(-i for i in range(1, len(kernel_size))))
        else:
            self.ifft = False
        
        if out_channels == in_channels:
            self.one = None
        else:
            self.one = torch.ones(out_channels, in_channels, 1, 1) + 1j*0
        
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_size, **self.factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **self.factory_kwargs))
        else:
            self.bias = None

        self._init_parameters()
        self._fourier_space(len(kernel_size))

    def _fourier_space(self, dims: int) -> Tensor:
        if self.bias is not None:
            self.bias = self.fft(self.bias, dim=(-i for i in range(1, dims)))
        self.weight = self.fft(self.weight, dim=(-i for i in range(1, dims)))

    def _init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)


class _FourierDeconvNd(_FourierConvNd):
    def __init__(self, in_channels: int, out_channels: int, *kernel_size, bias: bool = True, eps: float = 0.00001, pre_fft: bool = True, post_ifft: bool = False, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, *kernel_size, bias=bias, eps=eps, pre_fft=pre_fft, post_ifft=post_ifft, device=device, dtype=dtype)

class FourierConv1d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierconv1d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierconv1d(f.pad(
                input, self.padding, mode = 'constant', value = 0
            ), self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierConv2d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierconv2d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierconv2d(f.pad(input, self.padding, 'constant', value = 0), self.one, self.weight, self.bias)
            
        if self.ifft:
            return self.ifft(out)
        return out


class FourierConv3d(_FourierConvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierconv3d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierconv3d(f.pad(input, self.padding, 'constant', value = 0), self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out

class FourierDeconv1d(_FourierDeconvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv1d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierdeconv1d(f.pad(input, self.padding, 'constant', value = 0), self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierDeconv2d(_FourierDeconvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv2d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierdeconv2d(f.pad(input, self.padding, 'constant', value = 0), self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierDeconv3d(_FourierDeconvNd):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv3d(input, self.one, self.weight, self.bias)
        else:
            out = F.fourierdeconv3d(f.pad(input, self.padding, 'constant', value = 0), self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


__all__ = [
    "FourierConv1d",
    "FourierConv2d",
    "FourierConv3d",
    "FourierDeconv1d",
    "FourierDeconv2d",
    "FourierDeconv3d",
]
