from ..lightorch.nn.criterions import *
from torch import Tensor, nn
from .utils import *
import random

## Test automation for criterions

# Unit tests
input: Tensor = create_inputs(1, 3, 256, 256)
target: Tensor = create_inputs(1, 3, 256, 256)

mu: Tensor = create_inputs(1, 32)
logvar: Tensor = create_inputs(1, 32)

model: nn.Module = Model(32)
randint: int = random.randint(-100, 100)


def test_tv() -> None:
    loss = TV(randint)
    loss(input=input)


def test_style() -> None:
    loss = StyleLoss(model, input, randint)
    loss(input=input, target=target, feature_extractor=False)


def test_perc() -> None:
    loss = PerceptualLoss(model, input, randint)
    loss(input=input, target=target, feature_extractor=False)


def test_psnr() -> None:
    loss = PeakNoiseSignalRatio(1, randint)
    loss(input=input, target=target)


# Integration tests
def test_lagrange() -> None:
    loss = LagrangianFunctional(
        nn.MSELoss(),
        nn.MSELoss(),
        nn.MSELoss(),
        lambd=Tensor([randint for _ in range(2)]),
    )
    loss(input=input, target=target)


def test_loss() -> None:
    loss = Loss(TV(randint), PeakNoiseSignalRatio(1, randint))
    loss(input=input, target=target)


def test_elbo() -> None:
    loss = ELBO(randint, PeakNoiseSignalRatio(1, randint))
    loss(input=input, target=target, mu=mu, logvar=logvar)


def test_fourier() -> None:
    input: Tensor = create_inputs(1, 3, 256, 256)


def test_partial() -> None:
    raise NotImplementedError


def test_kan() -> None:
    raise NotImplementedError
