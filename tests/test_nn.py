import torch
from torch import Tensor, nn
import random
from ..lightorch.nn import *
from .utils import *

random.seed(42)
torch.manual_seed(42)

input: Tensor = create_inputs(1, 3, 256, 256)
target: Tensor = create_inputs(1, 3, 256, 256)

model: nn.Module = Model(32)
randint: int = random.randint(-100, 100)


def test_complex() -> None:
    sample_input = torch.randn(32, 10) + 1j * torch.randn(32, 10)
    layer = Complex(nn.Linear(10, 20))
    result = layer(sample_input)
    assert result is not None, "Complex failed"


def test_tv() -> None:
    loss = TV(randint)
    result = loss(input=input)
    assert result is not None, "TV loss failed"


# Integrated
def test_style() -> None:
    feature: nn.Module = FeatureExtractor([8, 12], "vgg16")
    loss = StyleLoss(feature, input, randint)
    result = loss(input=input, target=target, feature_extractor=False)
    assert result is not None, "StyleLoss failed"


# Integrated
def test_perc() -> None:
    feature: nn.Module = FeatureExtractor([8, 12], "vgg16")
    loss = PerceptualLoss(feature, input, randint)
    result = loss(input=input, target=target, feature_extractor=False)
    assert result is not None, "PerceptualLoss failed"


def test_psnr() -> None:
    loss = MSELoss(factor=randint)
    result = loss(input=input, target=target)
    assert result is not None, "MSE failed"


def test_psnr() -> None:
    loss = CrossEntropyLoss(factor=randint)
    result = loss(input=input, target=target)
    assert result is not None, "CrossEntropy failed"


def test_psnr() -> None:
    loss = PeakNoiseSignalRatio(1, randint)
    result = loss(input=input, target=target)
    assert result is not None, "PeakNoiseSignalRatio failed"


def test_lagrange() -> None:
    lambd = torch.tensor([randint for _ in range(2)], dtype=torch.float32)
    loss = LagrangianFunctional(
        MSELoss(factor=1), MSELoss(factor=1), MSELoss(factor=1), lambd=lambd
    )
    result = loss(input=input, target=target)
    assert result is not None, "LagrangianFunctional failed"


def test_loss() -> None:
    loss = Loss(TV(randint), PeakNoiseSignalRatio(1, randint))
    result = loss(input=input, target=target)
    assert result is not None, "Combined Loss failed"


def test_elbo() -> None:
    mu: Tensor = create_inputs(1, 32)
    logvar: Tensor = create_inputs(1, 32)
    loss = ELBO(randint, PeakNoiseSignalRatio(1, randint))
    result = loss(input=input, target=target, mu=mu, logvar=logvar)
    assert result is not None, "ELBO loss failed"


def test_fourier2d() -> None:
    sample_input: Tensor = torch.randn(32, 3, 256, 256)  # batch size, input_size
    model = nn.Sequential(
        FourierConv2d(3, 10, 5, 1, pre_fft=True),
        FourierConv2d(10, 20, 5, 1, post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 20, 256, 256), "FourierConv2d failed"

    model = nn.Sequential(
        FourierDeconv2d(3, 10, 5, 1, pre_fft=True),
        FourierDeconv2d(10, 20, 5, 1, post_ifft=True),
    )

    output = model(sample_input)
    assert output.shape == (32, 20, 256, 256), "FourierDeconv2d failed"


def test_fourier1d() -> None:
    sample_input: Tensor = torch.randn(32, 1, 10)  # batch size, channels, input_size
    model = nn.Sequential(
        FourierConv1d(1, 3, 5, 1, pre_fft=True),
        FourierConv1d(3, 5, 5, 1, post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 5, 10), "FourierConv1d failed"

    model = nn.Sequential(
        FourierDeconv1d(1, 3, 5, 1, pre_fft=True),
        FourierDeconv1d(3, 5, 5, 1, post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 74, 10), "FourierDeconv1d failed"


def test_fourier3d() -> None:
    sample_input: Tensor = torch.randn(
        32, 3, 5, 256, 256
    )  # batch size, channels, frames, height, width
    model = nn.Sequential(
        FourierConv3d(3, 2, 8, pre_fft=True),
        FourierConv3d(2, 1, 8, post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 1, 5, 256, 256), "FourierConv3d failed"

    model = nn.Sequential(
        FourierDeconv3d(3, 2, 8, pre_fft=True),
        FourierDeconv3d(2, 1, 8, post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 20, 256, 256), "FourierDeconv3d failed"


def test_partial() -> None:
    sample_input: Tensor = torch.randn(
        32, 3, 256, 256
    )  # batch size, channels, height, width
    mask = create_mask()
    model = nn.Sequential(PartialConv2d(3, 5, 3, 1, 1), PartialConv2d(5, 5, 3, 1, 1))
    output = model(sample_input, mask)
    assert output.shape == (32, 5, 256, 256), "PartialConv2d failed"


def test_normalization() -> None:
    sample_input: Tensor = torch.randn(
        32, 20, 10
    )  # batch size, sequence_length, input_size
    norm = RootMeanSquaredNormalization(dim=10)
    output = norm(sample_input)
    assert output.shape == (32, 20, 10), "RootMeanSquaredNormalization failed"


# Integrated
def test_monte_carlo() -> None:
    sample_input: Tensor = torch.randn(32, 10)  # batch size, input_size
    model = MonteCarloFC(
        fc_layer=DeepNeuralNetwork(
            in_features=10,
            layers=(20, 20, 1),
            activations=(nn.ReLU(), nn.ReLU(), nn.Sigmoid()),
        ),
        dropout=0.5,
        n_sampling=50,
    )
    output = model(sample_input)
    assert output.shape == (32, 1), "MonteCarloFC failed"


def test_kan() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("KAN test not implemented")
