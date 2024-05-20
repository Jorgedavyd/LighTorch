from torch import nn, Tensor
from typing import Sequence, Dict, Tuple, Optional, Callable, Sequence
import torch


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
            reconstruction_criterion.labels.append("KL Divergence"),
            reconstruction_criterion.factors.update({"KL Divergence": beta}),
        )

        self.L_recons = reconstruction_criterion(**self.recons_kwargs)
        self.beta = beta

    def forward(self, I_out, I_gt, mu, logvar) -> Tuple[Tensor, ...]:
        L_recons = self.L_recons(I_out, I_gt)

        L_kl = -0.5 * torch.sum(torch.log(logvar) - 1 + logvar + torch.pow(mu, 2))

        return (L_recons, L_kl, L_recons + self.beta * L_kl)


# Gram matrix

# perceptual

# ssim

# pnsr


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
]
