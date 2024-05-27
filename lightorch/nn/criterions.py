from torch import nn, Tensor
from typing import Sequence, Dict, Tuple, Optional, Callable, Sequence, List
import torch
from . import functional as F


class _Base(nn.Module):
    def __init__(
        self,
        labels: Sequence[str] | str,
        factors: Dict[str, float] | Sequence[Dict[str, float]],
    ) -> None:
        super().__init__()
        self.labels = labels.append("Overall")
        self.factors = factors


class ELBO(_Base):
    """
    # Variational Autoencoder Loss:
    \mathcal{L}_{total} = \mathcal{L}_{recons} - \beta \mathcal{L}_{KL}
    Given a beta parameter, it is converted into a \beta-VAE.
    """

    def __init__(self, beta: float, reconstruction_criterion: nn.Module) -> None:
        super().__init__(
            ["KL Divergence"] + reconstruction_criterion.labels,
            {"KL Divergence": beta}.update(reconstruction_criterion.factors),
        )

        self.L_recons = reconstruction_criterion
        self.beta = beta

    def forward(self, I_out, I_gt, mu, logvar) -> Tuple[Tensor, ...]:
        L_recons = self.L_recons(I_out, I_gt)

        L_kl = -0.5 * torch.sum(torch.log(logvar) - 1 + logvar + torch.pow(mu, 2))

        return (L_recons, L_kl, L_recons + self.beta * L_kl)


# Gram matrix based loss
class StyleLoss(_Base):
    def __init__(self, feature_extractor) -> None:
        super().__init__(labels="Style Loss", factors=None)
        self.feature_extractor = feature_extractor

        sample_tensor: Tensor = torch.randn(32, 1, 1024, 1024)
        F_p: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            F_p.append(c**3 * h * w)

        self.F_p: Tensor = Tensor(F_p)

    def forward(self, input: Tensor, target: Tensor, feature_extractor: bool = True):
        return F.style_loss(
            input,
            target,
            self.F_p,
            self.feature_extractor if feature_extractor else None,
        )


# Perceptual loss for style features
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

        sample_tensor: Tensor = torch.randn(32, 1, 1024, 1024)
        N_phi_p: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            N_phi_p.append(c * h * w)

        self.N_phi_p: Tensor = Tensor(N_phi_p)

    def forward(self, input: Tensor, target: Tensor, feature_extractor: bool = True):
        return F.perceptual_loss(
            input,
            target,
            self.N_phi_p,
            self.feature_extractor if feature_extractor else None,
        )


# pnsr


class PeakNoiseSignalRatio(nn.Module):
    def __init__(self, max: float) -> None:
        super().__init__()
        self.max = max

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.pnsr(input, target, self.max)


# Total variance


class TV(nn.Module):
    """
    # Total Variance (TV)
    """

    def __init__():
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.total_variance(input)


# lambda
class LagrangianFunctional(_Base):
    """
    Creates a lagrangian function of the form:
    $\mathcal{F}(f, g; \lambda) = f(x) - \lambda \dot g(x)$
    given g a vector field representing constraints.
    """

    def __init__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        *g,
        lambd: Tensor,
        f_name: Optional[str] = None,
        g_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        if f_name is None:
            f_name = "f"
        if g_names is None:
            g_names = [f"g_{i}" for i in range(len(g))]
        labels = [f_name, *g_names]
        super().__init__(labels, {k: v for k, v in zip(labels[1:], lambd)})
        if "make_convex" in kwargs:
            self.make_convex = True
        self.lambd = lambd
        self.g = g
        self.f = f

    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        return self.f(out, target) - torch.dot(
            self.lambd, Tensor([func(out, target) for func in self.g])
        )


__all__ = [
    "LagrangianFunctional",
    "ELBO",
    "TV",
    "PeakNoiseSignalRatio",
    "StyleLoss",
    "PerceptualLoss",
]
