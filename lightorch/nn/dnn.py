# Base Deep Neural Network
from torch import Tensor, nn
from typing import Sequence


def _SingularLayer(
    in_features: int, out_features: int, activation: nn.Module
) -> Tensor:
    out = nn.Sequential(nn.Linear(in_features, out_features), activation())
    return out


class DeepNeuralNetwork(nn.Sequential):
    """
    # Deep Neural Network
    in_features (int): Number of input features to the model.
    layers (Sequence[int]): A list/tuple/sequence of the hidden layers size.
    activations (Sequence[nn.Module, None]): A list/tuple/sequence of the activation functions for each layer (None is identity).
    """

    def __init__(
        self,
        in_features: int,
        layers: Sequence[int],
        activations: Sequence[nn.Module, None],
    ):
        assert len(layers) == len(
            activations
        ), "Must have the same amount of layers and activation functions"

        for activation in activations:
            if activation is None:
                activation = nn.Identity()

        self.dnn = nn.Sequential()

        for dim, activation in zip(layers, activations):
            self.dnn.append(_SingularLayer(in_features, dim, activation))
            in_features = dim

    def forward(self, input: Tensor) -> Tensor:
        return self.dnn(input)


__all__ = ["DeepNeuralNetwork"]
