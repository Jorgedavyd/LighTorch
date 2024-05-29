from torch import nn, Tensor
from typing import Sequence, Dict, Tuple, Optional, Callable, Sequence, List
import torch
from . import functional as F
from itertools import chain


def _merge_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out = dict()
    for dict_ in dicts:
        out.update(dict_)
    return out


class _Base(nn.Module):
    def __init__(
        self,
        labels: Sequence[str] | str,
        factors: Dict[str, float] | Sequence[Dict[str, float]],
    ) -> None:
        super().__init__()
        if "Overall" not in labels:
            self.labels = labels.append("Overall")
        self.factors = factors


class Loss(_Base):
    def __init__(self, *loss) -> None:
        super().__init__(
            list(set([*chain.from_iterable([i.labels for i in loss])])),
            _merge_dicts([i.factors for i in loss]),
        )
        assert len(self.loss) == len(
            self.factors
        ), "Must have the same length of losses as factors"
        self.loss = loss

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        loss = 0
        out_list = []

        for loss in self.loss:
            loss_, out_ = loss(**kwargs)
            out_list.append(loss_)
            loss += out_

        out_list.append(loss)

        return tuple(*out_list)


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

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        """
        input, target, logvar, mu
        """
        L_recons, L_recons_out = self.L_recons(kwargs["input"], kwargs["target"])

        L_kl = -0.5 * torch.sum(
            torch.log(kwargs["logvar"])
            - 1
            + kwargs["logvar"]
            + torch.pow(kwargs["mu"], 2)
        )

        return (L_recons, L_kl, L_recons_out + self.beta * L_kl)


# Gram matrix based loss
class StyleLoss(_Base):
    """
    forward (input, target, feature_extractor: bool = True)
    """

    def __init__(
        self, feature_extractor, sample_tensor: Tensor, factor: float = 1e-3
    ) -> None:
        super().__init__(
            labels=[self.__class__.__name__], factors={self.__class__.__name__: factor}
        )
        self.feature_extractor = feature_extractor

        F_p: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            F_p.append(c**3 * h * w)

        self.F_p: Tensor = Tensor(F_p)

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        out = F.style_loss(
            kwargs["input"],
            kwargs["target"],
            self.F_p,
            self.feature_extractor if kwargs.get("feature_extractor", True) else None,
        )
        return out, self.factors[self.__class__.__name__] * out


# Perceptual loss for style features
class PerceptualLoss(_Base):
    """
    forward (input, target, feature_extractor: bool = True)
    """

    def __init__(
        self, feature_extractor, sample_tensor: Tensor, factor: float = 1e-3
    ) -> None:
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})
        self.feature_extractor = feature_extractor
        N_phi_p: List[int] = []

        for feature_layer in self.fe(sample_tensor):
            c, h, w = feature_layer.shape[1:]
            N_phi_p.append(c * h * w)

        self.N_phi_p: Tensor = Tensor(N_phi_p)

    def forward(self, **kwargs) -> Tensor:
        out = F.perceptual_loss(
            kwargs["input"],
            kwargs["target"],
            self.N_phi_p,
            self.feature_extractor if kwargs.get("feature_extractor", True) else None,
        )
        return out, self.factors[self.__class__.__name__] * out


# pnsr


class PeakNoiseSignalRatio(_Base):
    """
    forward (input, target)
    """

    def __init__(self, max: float, factor: float = 1) -> None:
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})
        self.max = max

    def forward(self, **kwargs) -> Tensor:
        out = F.psnr(kwargs["input"], kwargs["target"], self.max)
        return out, out * self.factors[self.__class__.__name__]


# Total variance


class TV(nn.Module):
    """
    # Total Variance (TV)
    forward (input)
    """

    def __init__(self, factor: float = 1):
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})

    def forward(self, **kwargs) -> Tensor:
        out = F.total_variance(kwargs["input"])
        return out, out * self.factors[self.__class__.__name__]


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
    "Loss",
]
