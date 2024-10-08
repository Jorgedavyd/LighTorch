from torch import nn, Tensor
from typing import Optional, Sequence, Dict, Tuple, Sequence, List, Union
from . import functional as F
from itertools import chain


def _merge_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out = dict()
    for dict_ in dicts:
        out.update(dict_)
    return out


class LighTorchLoss(nn.Module):
    def __init__(
        self,
        labels: Union[List[str], str],
        factors: Union[Dict[str, float], Sequence[Dict[str, float]]],
    ) -> None:
        super().__init__()
        if isinstance(labels, str):
            labels = [labels]
        self.labels: List[str] = labels
        if "Overall" not in labels:
            self.labels.append("Overall")
        self.factors = factors


class Loss(LighTorchLoss):
    def __init__(self, *loss) -> None:
        assert len(set(map(type, loss))) == len(
            loss
        ), "Not valid input classes, each should be different."
        super().__init__(
            labels=list(set([*chain.from_iterable([i.labels for i in loss])])),
            factors=_merge_dicts([i.factors for i in loss]),
        )
        self.loss = loss

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        loss_ = Tensor([0.0])
        out_list = []
        for loss in self.loss:
            args = loss(**kwargs)
            out_list.extend(list(args[:-1]))
            loss_ += args[-1]
        out_list.append(loss_)
        out_list = tuple(out_list)
        return out_list


class MSELoss(nn.MSELoss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", factor: float = 1
    ) -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)
        self.factors = {self.__class__.__name__: factor}
        self.labels = [self.__class__.__name__]

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        out = super().forward(kwargs["input"], kwargs["target"])
        return out, out * self.factors[self.__class__.__name__]


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Union[Tensor, None] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
        factor: float = 1,
    ) -> None:
        super(CrossEntropyLoss, self).__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )
        self.factors = {self.__class__.__name__: factor}
        self.labels = [self.__class__.__name__]

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        out = super().forward(kwargs["input"], kwargs["target"])
        return out, out * self.factors[self.__class__.__name__]


class BinaryCrossEntropy(nn.BCELoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.factors = {self.__class__.__name__: factor}
        self.labels = [self.__class__.__name__]

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        out = super().forward(kwargs["input"], kwargs["target"])
        return out, out * self.factors[self.__class__.__name__]


class ELBO(LighTorchLoss):
    """
    # Variational Autoencoder Loss:
    \mathcal{L}_{total} = \mathcal{L}_{recons} - \beta \mathcal{L}_{KL}
    Given a beta parameter, it is converted into a \beta-VAE.
    """

    def __init__(self, beta: float, reconstruction_criterion: LighTorchLoss) -> None:
        factors = {"KL Divergence": beta}
        factors.update(reconstruction_criterion.factors)
        super().__init__(
            labels=["KL Divergence"] + reconstruction_criterion.labels, factors=factors
        )

        self.L_recons = reconstruction_criterion
        self.beta = beta

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        """
        input, target, logvar, mu
        """
        *L_recons, L_recons_out = self.L_recons(**kwargs)
        L_kl = F.kl_div(kwargs["mu"], kwargs["logvar"])
        return (*L_recons, L_kl, L_recons_out + self.beta * L_kl)


# Gram matrix based loss
class StyleLoss(LighTorchLoss):
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

        for feature_layer in self.feature_extractor(sample_tensor):
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
class PerceptualLoss(LighTorchLoss):
    """
    forward (input, target, feature_extractor: bool = True)
    """

    def __init__(
        self, feature_extractor, sample_tensor: Tensor, factor: float = 1e-3
    ) -> None:
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})
        self.feature_extractor = feature_extractor
        N_phi_p: List[int] = []

        for feature_layer in self.feature_extractor(sample_tensor):
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


class PeakSignalNoiseRatio(LighTorchLoss):
    """
    forward (input, target)
    """

    def __init__(self, max: float, factor: float = 1) -> None:
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})
        self.max = max

    def forward(self, **kwargs) -> Tensor:
        out = F.psnr(kwargs["input"], kwargs["target"], self.max)
        return out, out * self.factors[self.__class__.__name__]


class TV(LighTorchLoss):
    """
    # Total Variance (TV)
    forward (input)
    """

    def __init__(self, factor: float = 1):
        super().__init__([self.__class__.__name__], {self.__class__.__name__: factor})

    def forward(self, **kwargs) -> Tensor:
        out = F.total_variance(kwargs["input"])
        return out, out * self.factors[self.__class__.__name__]


class LagrangianFunctional(LighTorchLoss):
    """
    Creates a lagrangian function of the form:
    $\mathcal{F}(f, g; \lambda) = f(x) - \lambda \dot g(x)$
    given g a vector field representing constraints.
    """

    def __init__(
        self,
        f: LighTorchLoss,
        g: Sequence[LighTorchLoss],
        **kwargs,
    ) -> None:
        if f_name := getattr(f, "labels", False):
            assert (
                len(f_name) == 1
            ), "Not valid f function, should consist on just one criterion."
        else:
            raise ValueError(
                "Not valid constraint, should belong to LighTorchLoss class"
            )

        g_names: List[str] = []
        for constraint in g:
            if g_name := getattr(constraint, "labels", False):
                assert (
                    len(g_name) == 1
                ), "Not valid constraint function, should consist on just one criterion each."
                g_names.append(*g_name)
            else:
                raise ValueError(
                    "Not valid constraint, should belong to LighTorchLoss class"
                )
        for func in g:
            assert (
                list(func.factors.values())[0] < 0
            ), "Not valid factor for g, should be negative"

        f_name = f_name[0]

        labels = [f_name, *g_names]

        factors = {}

        for idx, func in enumerate([f, *g]):
            if idx < 1:
                factors.update(
                    {
                        f"f_{func.__class__.__name__}": func.factors[
                            func.__class__.__name__
                        ]
                    }
                )
            else:
                factors.update(
                    {
                        f"g_{idx}_{func.__class__.__name__}": func.factors[
                            func.__class__.__name__
                        ]
                    }
                )

        super().__init__(labels, factors)

        if "make_convex" in kwargs:
            self.make_convex = True
        else:
            self.make_convex = False

        self.g = g
        self.f = f

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        g_out_list: List[float] = []
        g_out_fact: List[float] = []
        for constraint in self.g:
            out, out_fact = constraint(**kwargs)
            g_out_list.append(out)
            g_out_fact.append(out_fact)

        f_out, f_fact = self.f(**kwargs)

        return f_out, *g_out_list, f_fact - sum(g_out_fact)


__all__ = [
    "LagrangianFunctional",
    "ELBO",
    "TV",
    "PeakSignalNoiseRatio",
    "StyleLoss",
    "PerceptualLoss",
    "Loss",
    "LighTorchLoss",
    "MSELoss",
    "CrossEntropyLoss",
]
