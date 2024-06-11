from torch.fft import fftn, ifftn
from torch import nn, Tensor
from torch.nn import init
from . import functional as F
import torch
from math import sqrt
import torch.nn.functional as f
from typing import Tuple, Sequence, Union
from itertools import chain


class _FourierConvNd(nn.Module):
    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 1e-5,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        self.n = n
        if isinstance(kernel_size, tuple):
            assert n == n, f"Not valid kernel size for {n}-convolution"
        else:
            kernel_size = (kernel_size,) * n

        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}

        if padding is not None:
            self.padding = self.get_padding(padding)
        else:
            self.padding = padding

        if pre_fft:
            self.fft = lambda x: fftn(x, dim=[-i for i in range(1, n + 1)])
        else:
            self.fft = False
        if post_ifft:
            self.ifft = lambda x: ifftn(x, dim=[-i for i in range(1, n + 1)])
        else:
            self.ifft = False

        if out_channels == in_channels:
            self.one = None
        else:
            self.one = (
                torch.ones(out_channels, in_channels, *[1 for _ in range(n)]) + 1j * 0
            )

        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(out_channels, *kernel_size, **self.factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **self.factory_kwargs))
        else:
            self.bias = None

        self._init_parameters()

    def get_padding(self, padding: Union[Tuple[int], int]) -> Sequence[int]:
        if isinstance(padding, tuple):
            assert (
                len(padding) == self.n
            ), f"Not valid padding scheme for {self.n}-convolution"
            return [*chain.from_iterable([(i,) * 2 for i in reversed(padding)])]
        else:
            return [*chain.from_iterable([(padding,) * 2 for _ in range(self.n)])]

    def _fourier_space(self) -> Tensor:
        # probably deprecated
        if self.bias is not None:
            self.bias = self.fft(self.bias, dim=[-i for i in range(1, self.n + 1)])
        self.weight = self.fft(self.weight, dim=[-i for i in range(1, self.n + 1)])

    def _init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)


class _FourierDeconvNd(_FourierConvNd):
    def __init__(
        self,
        n: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Tuple[int],
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            n,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )


class FourierConv1d(_FourierConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierconv1d(
                f.pad(input, self.padding, mode="constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierconv1d(input, self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierConv2d(_FourierConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierconv2d(
                f.pad(input, self.padding, "constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierconv2d(input, self.one, self.weight, self.bias)

        if self.ifft:
            return self.ifft(out)
        return out


class FourierConv3d(_FourierConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)

        if self.padding is not None:
            out = F.fourierconv3d(
                f.pad(input, self.padding, "constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierconv3d(input, self.one, self.weight, self.bias)

        if self.ifft:
            return self.ifft(out)
        return out


class FourierDeconv1d(_FourierDeconvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv1d(
                f.pad(input, self.padding, "constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierdeconv1d(input, self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierDeconv2d(_FourierDeconvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv2d(
                f.pad(input, self.padding, "constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierdeconv2d(input, self.one, self.weight, self.bias)
        if self.ifft:
            return self.ifft(out)
        return out


class FourierDeconv3d(_FourierDeconvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int], int],
        padding: Union[Tuple[int], int] = None,
        bias: bool = True,
        eps: float = 0.00001,
        pre_fft: bool = True,
        post_ifft: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            bias,
            eps,
            pre_fft,
            post_ifft,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fft:
            input = self.fft(input)
        if self.padding is not None:
            out = F.fourierdeconv3d(
                f.pad(input, self.padding, "constant", value=0),
                self.one,
                self.weight,
                self.bias,
            )
        else:
            out = F.fourierdeconv3d(input, self.one, self.weight, self.bias)
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
