import torch
from torch import Tensor, nn
import random
from lightorch.nn import *
from .utils import *

random.seed(42)
torch.manual_seed(42)

randint: int = random.randint(0, 20)
assert isinstance(randint, int)


def test_complex() -> None:
    sample_input = torch.randn(32, 10) + 1j * torch.randn(32, 10)
    layer = Complex(nn.Linear(10, 20))
    result = layer(sample_input)
    assert result is not None, "Complex failed"


def test_tv() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    loss = TV(randint)
    result = loss(input=input)
    assert result is not None, "TV loss failed"


# Integrated
def test_style() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    feature_extractor: nn.Module = FeatureExtractor([8, 12], "vgg16")
    loss = StyleLoss(feature_extractor, input, randint)
    result = loss(input=input, target=target, feature_extractor=True)
    assert result is not None, "StyleLoss failed"


# Integrated
def test_perc() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    feature: nn.Module = FeatureExtractor([8, 12], "vgg16")
    loss = PerceptualLoss(feature, input, randint)
    result = loss(input=input, target=target, feature_extractor=False)
    assert result is not None, "PerceptualLoss failed"


def test_mse() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    loss = MSELoss(factor=randint)
    result = loss(input=input, target=target)
    assert result is not None, "MSE failed"


def test_entropy_loss() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256).clamp(0, 1)
    target: Tensor = torch.randn(1, 3, 256, 256).clamp(0, 1)
    loss = CrossEntropyLoss(factor=randint)
    result = loss(input=input, target=target)
    assert result is not None, "CrossEntropy failed"


def test_psnr() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    loss = PeakSignalNoiseRatio(1, randint)
    result = loss(input=input, target=target)
    assert result is not None, "PeakNoiseSignalRatio failed"


def test_lagrange() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    loss = LagrangianFunctional(
        MSELoss(factor=1),
        (MSELoss(factor=-1), MSELoss(factor=-1)),
    )
    result = loss(input=input, target=target)
    assert result is not None, "LagrangianFunctional failed"


def test_loss() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256).clamp_max(1)
    target: Tensor = torch.randn(1, 3, 256, 256).clamp_max(1)
    loss = Loss(TV(randint), PeakSignalNoiseRatio(1, randint))
    result = loss(input=input, target=target)
    assert result is not None, "Combined Loss failed"


def test_elbo() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    mu: Tensor = create_inputs(1, 32)
    logvar: Tensor = create_inputs(1, 32)
    loss = ELBO(randint, PeakSignalNoiseRatio(1, randint))
    result = loss(input=input, target=target, mu=mu, logvar=logvar)
    assert result is not None, "ELBO loss failed"


def test_fourier2d() -> None:
    sample_input: Tensor = torch.randn(32, 3, 256, 256)  # batch size, input_size
    model = nn.Sequential(
        FourierConv2d(3, 5, (8, 8), pre_fft=True),
        FourierConv2d(5, 3, (8, 8), post_ifft=True),
    )

    output = model(sample_input)

    assert output.shape == (32, 3, 256, 256), "FourierConv2d failed"

    model = nn.Sequential(
        FourierDeconv2d(3, 5, (8, 8), pre_fft=True),
        FourierDeconv2d(5, 3, (8, 8), post_ifft=True),
    )

    output = model(sample_input)
    assert output.shape == (32, 3, 256, 256), "FourierDeconv2d failed"


def test_fourier1d() -> None:
    sample_input: Tensor = torch.randn(32, 3, 10)  # batch size, channels, input_size
    model = nn.Sequential(
        FourierConv1d(3, 5, 2, pre_fft=True), FourierConv1d(5, 3, 2, post_ifft=True)
    )
    output = model(sample_input)
    assert output.shape == (32, 3, 10), "FourierConv1d failed"

    model = nn.Sequential(
        FourierDeconv1d(3, 5, 2, pre_fft=True), FourierDeconv1d(5, 3, 2, post_ifft=True)
    )
    output = model(sample_input)

    assert output.shape == (32, 3, 10), "FourierDeconv1d failed"


def test_fourier3d() -> None:
    sample_input: Tensor = torch.randn(
        32, 3, 5, 256, 256
    )  # batch size, channels, frames, height, width
    model = nn.Sequential(
        FourierConv3d(3, 5, (1, 8, 8), pre_fft=True),
        FourierConv3d(5, 3, (1, 8, 8), post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 3, 5, 256, 256), "FourierConv3d failed"

    model = nn.Sequential(
        FourierDeconv3d(3, 5, (1, 8, 8), pre_fft=True),
        FourierDeconv3d(5, 3, (1, 8, 8), post_ifft=True),
    )
    output = model(sample_input)
    assert output.shape == (32, 3, 5, 256, 256), "FourierDeconv3d failed"


def test_partial() -> None:
    sample_input: Tensor = torch.randn(
        1, 3, 256, 256
    )  # batch size, channels, height, width
    mask = create_mask().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    model = PartialConv2d(3, 5, 3, 1, 1)
    out, mask = model(sample_input, mask)
    assert out.shape == (1, 5, 256, 256), "PartialConv2d failed"
    assert out.shape == mask.shape, "PartialConv2d failed"


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
            activations=(nn.ReLU, nn.ReLU, nn.Sigmoid),
        ),
        dropout=0.5,
        n_sampling=50,
    )
    output = model(sample_input)
    assert output.shape == (32, 1), "MonteCarloFC failed"


def test_kan() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("KAN test not implemented")

def test_trans() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("Transformer test not implemented")

def test_att() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("Attention test not implemented")

def test_ffn() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("FFN test not implemented")

def test_pos_embed() -> None:
    # Placeholder for future implementation
    raise NotImplementedError("Positional Encoding and Embedding test not implemented")