import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Callable, List
import torch.nn.functional as F
from lightning.pytorch import LightningModule


def fourierconvNd(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
    out = x * weight
    if bias is not None:
        return out + bias
    return out


def partialconv3d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    with torch.no_grad():

        sum = F.conv3d(
            mask_in,
            torch.ones_like(weight, requires_grad=False),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)

    out = F.conv3d(input * mask_in, weight, None, stride, padding, dilation)

    out = out * (one / sum) + bias

    return (out, updated_mask) if update_mask else out


def partialconv2d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one: Tensor,
    ones_weight: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    with torch.no_grad():

        sum = F.conv2d(
            mask_in, ones_weight, stride=stride, padding=padding, dilation=dilation
        )

        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)

    out = F.conv2d(input, weight, None, stride, padding, dilation)

    out = out * (one / sum) + bias

    return (out, updated_mask) if update_mask else out


def partialconv1d(
    input: Tensor,
    mask_in: Tensor,
    weight: Tensor,
    one: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    update_mask: bool = True,
) -> Union[Tuple[Tensor, Tensor], Tensor]:

    with torch.no_grad():

        sum = F.conv1d(
            mask_in,
            torch.ones_like(weight, requires_grad=False),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if update_mask:
            updated_mask = torch.clamp_max(sum, 1)

    out = F.conv1d(input * mask_in, weight, None, stride, padding, dilation)

    out = out * (one / sum) + bias

    return (out, updated_mask) if update_mask else out


def residual_connection(
    x: Tensor,
    sublayer: Callable[[Tensor], Tensor],
    to_dim_layer: Callable[[Tensor], Tensor] = nn.Identity(),
) -> Tensor:
    return to_dim_layer(x) + sublayer(x)


# Criterion functionals


def psnr(input: Tensor, target: Tensor, max: float) -> Tensor:
    return 10 * torch.log10(
        torch.div(torch.pow(max, 2), torch.nn.functional.mse_loss(input, target))
    )


def style_loss(
    input: Tensor,
    target: Tensor,
    F_p: Tensor,
    feature_extractor: nn.Module | LightningModule = None,
) -> Tensor:
    if feature_extractor is not None:
        phi_input: Tensor = feature_extractor(input)
        phi_output: Tensor = feature_extractor(target)

    phi_input: List[Tensor] = change_dim(phi_input)
    phi_output: List[Tensor] = change_dim(phi_output)

    return ((_style_forward(phi_input, phi_output)) / F_p).sum()


def perceptual_loss(
    input: Tensor,
    target: Tensor,
    N_phi_p: Tensor,
    feature_extractor: nn.Module | LightningModule = None,
) -> Tensor:
    if feature_extractor is not None:
        phi_input: Tensor = feature_extractor(input)
        phi_output: Tensor = feature_extractor(target)
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
    return torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + torch.mean(
        torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :])
    )


def KL_div(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
