import torch
from torch import nn, Tensor
from typing import Tuple, Sequence
from ..nn.criterions import _Loss
from ..training import Module
from dataclasses import dataclass

"""
VAE

beta-VAE

Conditional VAE
"""


@dataclass
class ValidKwargs:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    encoder_lr: float
    encoder_wd: float
    decoder_lr: float
    decoder_wd: float

    grad_clip_value: float

    # Reconstruction criterion
    reconstruction_criterion: nn.Module
    recons_params: Sequence[float] | float | None


class Loss(_Loss):
    """
    # Variational Autoencoder Loss:
    \mathcal{L}_{total} = \mathcal{L}_{recons} - \beta \mathcal{L}_{KL}
    Given a beta parameter, it is converted into a \beta-VAE.
    """

    def __init__(self, beta: float, reconstruction_criterion: nn.Module) -> None:
        super().__init__(
            reconstruction_criterion.labels.append("KL Divergence"),
            reconstruction_criterion.factors.update({"KL Divergence": beta}),
        )

        self.L_recons = reconstruction_criterion(**self.recons_kwargs)
        self.beta = beta

    def forward(self, I_out, I_gt, mu, logvar) -> Tuple[Tensor, ...]:
        L_recons = self.L_recons(I_out, I_gt)

        L_kl = -0.5 * torch.sum(torch.log(logvar) - 1 + logvar + torch.pow(mu, 2))

        return (L_recons, L_kl, L_recons + self.beta * L_kl)


class VAE(Module):
    """ """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        conditional: bool = True,
        **hparams,
    ) -> None:
        super().__init__()
        # Defining the hyperparameters
        for hparam_name, hparam in hparams.items():
            setattr(self, hparam_name, hparam)
        # Defining the encoder-decoder architecture
        self.encoder = encoder
        self.decoder = decoder
        # Defining conditional VAE
        self.conditional = conditional
        # Defining the criterion
        self.criterion = Loss(self.beta, self.reconstruction_criterion)
        # latent variable characteristics
        assert (
            self.encoder.fc.out_features % 2 == 0
        ), f"Not valid encoder final layer output size {self.encoder.fc.out_features}, should be even."
        self._half = self.encoder.fc.out_features // 2

    def _reparametrization(self, encoder_output: Tensor) -> Tuple[Tensor, ...]:
        # encoder_output = encoder_output.view(b, _, 2 * self._half, 2*self._half)
        pass

    def training_step(self, batch, idx) -> Tensor:
        x, y = batch
        x, mu, std = self(x, y)
        return self.compute_loss(x, mu, std)

    def validation_step(self, batch, idx) -> None:
        pass

    def forward(self, x: Tensor) -> Tuple[Tensor, ...] | Tensor:
        x = self.encoder(x)
        x, mu, std = self._reparametrization(x)
        x = self.decoder(x)
        return x, mu, std
