import torch.nn.functional as F
from torch import Tensor, nn
from typing import Union, Tuple
import torch


class _PartialConvNd:
    def __init__(self) -> None:

        self.one = 1

        for dim in self.kernel_size[1:]:
            self.one *= dim

        self.mask_one = torch.ones_like(self.weight, requires_grad=False)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        self.update_mask = kwargs.pop("update_mask", True)
        super().__init__(*args, **kwargs)
        _PartialConvNd.__init__(self)

    def forward(
        self, input: Tensor, mask_in: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        with torch.no_grad():

            sum = F.conv2d(
                mask_in,
                self.mask_one,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

            if self.update_mask:
                updated_mask = torch.clamp_max(sum, 1)

        out = super(PartialConv2d, self)._conv_forward(input, self.weight, None)

        out = out * (self.one / sum) + self.bias.view(1, -1, 1, 1)

        return (out, updated_mask) if self.update_mask else out


class PartialConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs) -> None:
        self.update_mask = kwargs.pop("update_mask", True)
        super(PartialConv3d, self).__init__(*args, **kwargs)
        _PartialConvNd.__init__(self)

    def forward(
        self, input: Tensor, mask_in: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        with torch.no_grad():

            sum = F.conv3d(
                mask_in,
                self.mask_one,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

            if self.update_mask:
                updated_mask = torch.clamp_max(sum, 1)

        out = super(PartialConv3d, self)._conv_forward(input, self.weight, None)

        out = out * (self.one / sum) + self.bias.view(1, -1, 1, 1)

        return (out, updated_mask) if self.update_mask else out


class PartialConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        self.update_mask = kwargs.pop("update_mask", True)
        super(PartialConv1d, self).__init__(*args, **kwargs)
        _PartialConvNd.__init__(self)

    def forward(
        self, input: Tensor, mask_in: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        with torch.no_grad():

            sum = F.conv1d(
                mask_in,
                self.mask_one,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

            if self.update_mask:
                updated_mask = torch.clamp_max(sum, 1)

        out = super(PartialConv1d, self)._conv_forward(input, self.weight, None)

        out = out * (self.one / sum) + self.bias.view(1, -1, 1, 1)

        return (out, updated_mask) if self.update_mask else out


__all__ = ["PartialConv1d", "PartialConv2d", "PartialConv3d"]
