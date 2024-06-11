import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Callable, List
import torch.nn.functional as F
from einops import rearrange
from torch.fft import fftn
from .utils import FeatureExtractor2D


def _fourierconvNd(
    n: int, x: Tensor, weight: Tensor, bias: Union[Tensor, None]
) -> Tensor:
    # To fourier space
    weight = fftn(weight, dim=[-i for i in range(1, n + 1)])

    # weight -> 1, 1, out channels, *kernel_size
    x *= weight  # Convolution in the fourier space

    if bias is not None:
        bias = fftn(bias, dim=-1)
        return x + bias.reshape(-1, *[1 for _ in range(n)])

    return x


def _fourierdeconvNd(
    n: int, x: Tensor, weight: Tensor, bias: Union[Tensor, None], eps: float = 1e-5
) -> Tensor:
    # To fourier space
    weight = fftn(weight, dim=[-i for i in range(1, n + 1)])

    # weight -> 1, out channels, *kernel_size
    x /= weight + eps  # Convolution in the fourier space

    if bias is not None:
        bias = fftn(bias, dim=-1)
        return x + bias.reshape(-1, *[1 for _ in range(n)])

    return x


def fourierconv3d(x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None]):
    """
    x (Tensor): batch size, channels, height, width
    weight (Tensor): out channels, *kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    stride: (int)
    padding: (int)
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv3d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    input_shape = x.shape
    weight_shape = weight.shape
    # Rearrange tensors for Fourier convolution
    out = rearrange(
        out,
        "B C (f kd) (h kh) (w kw) -> B (f h w) C kd kh kw",
        kd=weight_shape[-3],
        kh=weight_shape[-2],
        kw=weight_shape[-1],
    )

    out = _fourierconvNd(3, out, weight, bias)

    out = rearrange(
        out,
        "B (f h w) C kd kh kw -> B C (f kd) (h kh) (w kw)",
        f=int(input_shape[-3] / weight_shape[-3]),
        h=int(input_shape[-2] / weight_shape[-2]),
        w=int(input_shape[-1] / weight_shape[-1]),
    )

    return out


def fourierconv2d(x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None]):
    """
    x (Tensor): batch size, channels, height, width
    weight (Tensor): out channels, *kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv2d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    input_shape = x.shape
    weight_shape = weight.shape

    out = rearrange(
        out,
        "B C (h kh) (w kw) -> B (h w) C kh kw",
        kh=weight_shape[-2],
        kw=weight_shape[-1],
    )

    out = _fourierconvNd(2, out, weight, bias)

    out = rearrange(
        out,
        "B (h w) C k1 k2 -> B C (h k1) (w k2)",
        h=int(input_shape[-2] / weight_shape[-2]),
        w=int(input_shape[-1] / weight_shape[-1]),
    )

    return out


def fourierconv1d(x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None]):
    """
    x (Tensor): batch size, channels, sequence length
    weight (Tensor): out channels, kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv1d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    weight_shape = weight.shape

    out = rearrange(out, "B C (l k) -> B l C k", k=weight_shape[-1])

    out = _fourierconvNd(1, out, weight, bias)

    out = rearrange(out, "B l C k -> B C (l k)")

    return out


def fourierdeconv3d(
    x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None], eps: float = 1e-5
):
    """
    x (Tensor): batch size, channels, height, width
    weight (Tensor): out channels, *kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv3d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    # Rearrange tensors for Fourier convolution
    input_shape = x.shape
    weight_shape = weight.shape

    # Rearrange tensors for Fourier convolution
    out = rearrange(
        out,
        "B C (f kd) (h kh) (w kw) -> B (f h w) C kd kh kw",
        kd=weight_shape[-3],
        kh=weight_shape[-2],
        kw=weight_shape[-1],
    )

    out = _fourierdeconvNd(3, out, weight, bias, eps)

    out = rearrange(
        out,
        "B (f h w) C kd kh kw -> B C (f kd) (h kh) (w kw)",
        f=int(input_shape[-3] / weight_shape[-3]),
        h=int(input_shape[-2] / weight_shape[-2]),
        w=int(input_shape[-1] / weight_shape[-1]),
    )

    return out


def fourierdeconv2d(
    x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None], eps: float = 1e-5
):
    """
    x (Tensor): batch size, channels, height, width
    weight (Tensor): out channels, *kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv2d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    input_shape = x.shape
    weight_shape = weight.shape

    out = rearrange(
        out,
        "B C (h kh) (w kw) -> B (h w) C kh kw",
        kh=weight_shape[-2],
        kw=weight_shape[-1],
    )

    out = _fourierdeconvNd(2, out, weight, bias, eps)

    out = rearrange(
        out,
        "B (h w) C k1 k2 -> B C (h k1) (w k2)",
        h=int(input_shape[-2] / weight_shape[-2]),
        w=int(input_shape[-1] / weight_shape[-1]),
    )

    return out


def fourierdeconv1d(
    x: Tensor, one: Tensor, weight: Tensor, bias: Union[Tensor, None], eps: float = 1e-5
):
    """
    x (Tensor): batch size, channels, sequence length
    weight (Tensor): out channels, kernel_size
    one (Tensor): out channels, in channels, *1 #Paralelization game
    bias (Tensor): out channels
    """
    if one is not None:
        # Augment the channel dimension of the input
        out = F.conv1d(x, one, None, 1)  # one: (out_channel, in_channel, *kernel_size)

    out = rearrange(out, "B C (l k) -> B l C k", k=weight.shape[-1])

    out = _fourierdeconvNd(1, out, weight, bias, eps)

    out = rearrange(out, "B l C k -> B C (l k)")

    return out


def _partialconvnd(
    conv: F,
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one_sum: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    with torch.no_grad():
        sum_m: Tensor = conv(
            mask_in,
            torch.ones_like(weight, requires_grad=False),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if update_mask:
            updated_mask = sum_m.clamp_max(1)

    out = conv(input * mask_in, weight, None, stride, padding, dilation)

    out *= one_sum / sum_m
    out += bias

    return (out, updated_mask) if update_mask else out


def partialconv3d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one_sum: int,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    return _partialconvnd(
        F.conv3d,
        input,
        mask_in,
        weight,
        one_sum,
        bias,
        stride,
        padding,
        dilation,
        update_mask,
    )


def partialconv2d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one_sum: int,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    return _partialconvnd(
        F.conv2d,
        input,
        mask_in,
        weight,
        one_sum,
        bias,
        stride,
        padding,
        dilation,
        update_mask,
    )


def partialconv1d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one_sum: int,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    return _partialconvnd(
        F.conv1d,
        input,
        mask_in,
        weight,
        one_sum,
        bias,
        stride,
        padding,
        dilation,
        update_mask,
    )


def residual_connection(
    x: Tensor,
    sublayer: Callable[[Tensor], Tensor],
    to_dim_layer: Callable[[Tensor], Tensor] = nn.Identity(),
) -> Tensor:
    return to_dim_layer(x) + sublayer(x)


# Criterion functionals


def psnr(input: Tensor, target: Tensor, max: float) -> Tensor:
    max = Tensor([max])
    return 10 * torch.log10(
        torch.div(torch.pow(max, 2), torch.nn.functional.mse_loss(input, target))
    )


def style_loss(
    input: Tensor,
    target: Tensor,
    F_p: Tensor,
    feature_extractor: FeatureExtractor2D = None,
) -> Tensor:
    if feature_extractor is not None:
        phi_input: Tensor = feature_extractor(input)
        phi_output: Tensor = feature_extractor(target)
    else:
        phi_input: Tensor = input
        phi_output: Tensor = target

    phi_input: List[Tensor] = change_dim(phi_input)
    phi_output: List[Tensor] = change_dim(phi_output)

    return ((_style_forward(phi_input, phi_output)) / F_p).sum()


def perceptual_loss(
    input: Tensor,
    target: Tensor,
    N_phi_p: Tensor,
    feature_extractor: FeatureExtractor2D = None,
) -> Tensor:
    if feature_extractor is not None:
        phi_input: Tensor = feature_extractor(input)
        phi_output: Tensor = feature_extractor(target)
    else:
        phi_input: Tensor = input
        phi_output: Tensor = target

    return (
        Tensor(
            [
                torch.norm(phi_out - phi_gt, p=1)
                for phi_out, phi_gt in zip(
                    phi_input,
                    phi_output,
                )
            ]
        )
        / N_phi_p
    ).sum()


def change_dim(P: List[Tensor]) -> List[Tensor]:
    return [tensor.view(tensor.shape[0], tensor.shape[1], -1) for tensor in P]


def _style_forward(input_list: List[Tensor], gt_list: List[Tensor]) -> List[Tensor]:
    return Tensor(
        [
            torch.norm(out @ out.transpose(-2, -1) - gt @ gt.transpose(-2, -1), p=1)
            for out, gt in zip(input_list, gt_list)
        ]
    )


def total_variance(input: Tensor) -> Tensor:
    return (
        torch.norm(input[:, :, :, :-1] - input[:, :, :, 1:], p=1).sum()
        + torch.norm(input[:, :, :-1, :] - input[:, :, 1:, :], p=1).sum()
    )


def kl_div(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
