import torch
from torch import Tensor, nn
import random
from lightorch.nn import *
from lightorch.nn.sequential.residual import *
from .utils import *
import pytest
from datetime import timedelta

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
    feature_extractor: nn.Module = FeatureExtractor2D([8, 12], "vgg16")
    loss = StyleLoss(feature_extractor, input, randint)
    result = loss(input=input, target=target, feature_extractor=True)
    assert result is not None, "StyleLoss failed"


# Integrated
def test_perc() -> None:
    input: Tensor = torch.randn(1, 3, 256, 256)
    target: Tensor = torch.randn(1, 3, 256, 256)
    feature: nn.Module = FeatureExtractor2D([8, 12], "vgg16")
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


def test_transformer_cell() -> None:
    batch_size = 32
    seq_length = 10
    embed_dim = 64
    n_heads = 8

    input_tensor = torch.randn(batch_size, seq_length, embed_dim)
    cross_tensor = torch.randn(batch_size, seq_length, embed_dim)

    attention = MultiHeadAttention(embed_dim, seq_length, n_heads)
    ffn = nn.Sequential(
        nn.Linear(embed_dim, 4 * embed_dim),
        nn.ReLU(),
        nn.Linear(4 * embed_dim, embed_dim),
    )
    prenorm = nn.LayerNorm(embed_dim)
    postnorm = nn.LayerNorm(embed_dim)

    transformer_cell = TransformerCell(
        self_attention=SelfAttention(attention),
        cross_attention=CrossAttention(attention),
        ffn=ffn,
        prenorm=prenorm,
        postnorm=postnorm,
    )

    output = transformer_cell.self_attention(input_tensor)
    assert output.shape == input_tensor.shape

    output = transformer_cell.cross_attention(input_tensor, cross_tensor)
    assert output.shape == input_tensor.shape

    output = transformer_cell.ffn(input_tensor)
    assert output.shape == input_tensor.shape


def test_transformer() -> None:
    batch_size = 32
    seq_length = 10
    embed_dim = 64
    n_heads = 8

    input_tensor = torch.randn(batch_size, seq_length, embed_dim)

    embedding_layer = nn.Linear(embed_dim, embed_dim)
    positional_encoding = AbsoluteSinusoidalPositionalEncoding(0.0)
    attention = MultiHeadAttention(embed_dim, seq_length, n_heads)
    ffn = nn.Sequential(
        nn.Linear(embed_dim, 4 * embed_dim),
        nn.ReLU(),
        nn.Linear(4 * embed_dim, embed_dim),
    )
    prenorm = nn.LayerNorm(embed_dim)
    postnorm = nn.LayerNorm(embed_dim)

    encoder = TransformerCell(
        self_attention=SelfAttention(attention),
        cross_attention=CrossAttention(attention),
        ffn=ffn,
        prenorm=prenorm,
        postnorm=postnorm,
    )

    transformer = Transformer(
        embedding_layer=embedding_layer,
        positional_encoding=positional_encoding,
        encoder=encoder,
        n_layers=1,
    )

    output = transformer(input_tensor)
    assert output.shape == (batch_size, seq_length, embed_dim)


def test_cross_transformer() -> None:
    batch_size = 32
    seq_length = 10
    embed_dim = 64
    n_heads = 8

    first_input = torch.randn(batch_size, seq_length, embed_dim)
    second_input = torch.randn(batch_size, seq_length, embed_dim)

    attention = MultiHeadAttention(embed_dim, seq_length, n_heads)
    ffn = nn.Sequential(
        nn.Linear(embed_dim, 4 * embed_dim),
        nn.ReLU(),
        nn.Linear(4 * embed_dim, embed_dim),
    )
    prenorm = nn.LayerNorm(embed_dim)
    postnorm = nn.LayerNorm(embed_dim)

    cell1 = TransformerCell(
        self_attention=SelfAttention(attention),
        cross_attention=CrossAttention(attention),
        ffn=ffn,
        prenorm=prenorm,
        postnorm=postnorm,
    )

    cell2 = TransformerCell(
        self_attention=SelfAttention(attention),
        cross_attention=CrossAttention(attention),
        ffn=ffn,
        prenorm=prenorm,
        postnorm=postnorm,
    )

    cross_transformer = CrossTransformer(
        cell1, cell2, n_layers=1, fc=nn.Linear(embed_dim, embed_dim)
    )

    output = cross_transformer(first_input, second_input)
    assert output[0].shape == (batch_size, seq_length, embed_dim)
    assert output[1].shape == (batch_size, seq_length, embed_dim)


def test_att() -> None:
    batch_size = 32
    seq_length = 10
    embed_dim = 64
    n_heads = 8
    n_queries = 8
    n_groups = 4

    input_tensor = torch.randn(batch_size, seq_length, embed_dim)
    cross_tensor = torch.randn(batch_size, seq_length, embed_dim)

    # Initialize attention mechanisms
    multi_head_attention = MultiHeadAttention(embed_dim, seq_length, n_heads)
    multi_query_attention = MultiQueryAttention(embed_dim, seq_length, n_queries)
    grouped_query_attention = GroupedQueryAttention(
        embed_dim, seq_length, n_queries, n_groups
    )

    # Wrap with SelfAttention and CrossAttention
    self_attention_mh = SelfAttention(multi_head_attention)
    self_attention_mq = SelfAttention(multi_query_attention)
    self_attention_gq = SelfAttention(grouped_query_attention)

    cross_attention_mh = CrossAttention(multi_head_attention, method="i i c")
    cross_attention_mq = CrossAttention(multi_query_attention, method="i i c")
    cross_attention_gq = CrossAttention(grouped_query_attention, method="i i c")

    output_mh_self = self_attention_mh(input_tensor)
    assert (
        output_mh_self.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_mh_self.shape}"

    output_mq_self = self_attention_mq(input_tensor)
    assert (
        output_mq_self.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_mq_self.shape}"

    output_gq_self = self_attention_gq(input_tensor)
    assert (
        output_gq_self.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_gq_self.shape}"

    output_mh_cross = cross_attention_mh(input_tensor, cross_tensor)
    assert (
        output_mh_cross.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_mh_cross.shape}"

    output_mq_cross = cross_attention_mq(input_tensor, cross_tensor)
    assert (
        output_mq_cross.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_mq_cross.shape}"

    output_gq_cross = cross_attention_gq(input_tensor, cross_tensor)
    assert (
        output_gq_cross.shape == input_tensor.shape
    ), f"Expected shape {input_tensor.shape}, got {output_gq_cross.shape}"


models_with_params = [
    (FFN_ReLU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_Bilinear, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_Sigmoid, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_Swish, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_GELU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (BiGLU, {"in_features": 64, "out_features": 128}),
    (GLU, {"in_features": 64, "out_features": 128}),
    (ReGLU, {"in_features": 64, "out_features": 128}),
    (GEGLU, {"in_features": 64, "out_features": 128}),
    (SiGLU, {"in_features": 64, "out_features": 128}),
    (FFN_SwiGLU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_ReGLU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_GEGLU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
    (FFN_GLU, {"in_features": 64, "k_multiplier": 2, "out_features": 128}),
]


@pytest.mark.parametrize("model_class, params", models_with_params)
def test_ffn(model_class, params) -> None:
    model = model_class(**params)

    in_features = params["in_features"]
    x = torch.randn(32, in_features)

    output = model(x)

    out_features = params["out_features"]
    assert output.shape == (32, out_features)


def test_pos() -> None:
    dropout = 0.1
    batch_size = 32
    seq_length = 10
    embed_dim = 64
    degree = 2
    edge_order = 1
    delta_t = timedelta(seconds=1)

    input_tensor = torch.randn(batch_size, seq_length, embed_dim)

    abs_pos_enc = AbsoluteSinusoidalPositionalEncoding(dropout=dropout)
    rot_pos_enc = RotaryPositionalEncoding(d_model=embed_dim, seq_len=seq_length)
    dn_pos_enc = DnPositionalEncoding(
        delta_t=delta_t, degree=degree, edge_order=edge_order
    )

    output = rot_pos_enc(input_tensor)
    assert output.shape == input_tensor.shape
    output = abs_pos_enc(input_tensor)
    assert output.shape == input_tensor.shape
    output = dn_pos_enc(input_tensor)
    assert output.shape == input_tensor.shape


# implementation on c++
# def test_patch_embedding_3dcnn():
#     batch_size = 2
#     frames = 8
#     channels = 3
#     height = 32
#     width = 32
#     h_div = 4
#     w_div = 4
#     d_model = 64
#     architecture = (channels,)
#     hidden_activations = (nn.ReLU(),)
#     dropout = 0.1

#     input_tensor = torch.randn(batch_size, frames, channels, height, width)

#     feature_extractor = FeatureExtractor3D() # Define
#     pe = AbsoluteSinusoidalPositionalEncoding()

#     patch_embed = PatchEmbeddding3DCNN(h_div=h_div, w_div=w_div, pe=pe, feature_extractor=feature_extractor, X=input_tensor)

#     output = patch_embed(input_tensor)

#     assert output.shape == (batch_size, h_div * w_div, d_model)

#     feature_extractor = FeatureExtractor2D()

#     patch_embed = PatchEmbedding2DCNN(d_model=d_model, pe=pe, feature_extractor=feature_extractor, architecture=architecture, hidden_activations=hidden_activations, dropout=dropout)

#     output = patch_embed(input_tensor)

#     assert output.shape == (batch_size, frames, d_model)

# def test_res() -> None:
#     input_size = 10
#     sequence_length = 16
#     batch_size = 2
#     hidden_size = 20
#     rnn_layers = 1
#     res_layers = 1
#     shape = (input_size, sequence_length, batch_size)
#     x = torch.randn(*shape) # batch_size, sequence, input_size

#     model = GRU(input_size, hidden_size, rnn_layers, res_layers)
#     out = model(x)
#     assert (out.shape == shape), 'Residual GRU failed'

#     model = LSTM(input_size, hidden_size, rnn_layers, res_layers)
#     out = model(x)
#     assert (out.shape == shape), 'Residual LSTM failed'
