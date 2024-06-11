from torch import nn, Tensor
from typing import List, Sequence
from torchvision.models import (
    VGG19_Weights,
    vgg19,
    VGG16_Weights,
    vgg16,
    ResNet50_Weights,
    resnet50,
    resnet101,
    ResNet101_Weights,
)


VALID_MODELS_2D = {
    "vgg19": {
        "model": vgg19,
        "weights": VGG19_Weights,
        "valid_layers": [i for i in range(37)],
    },
    "vgg16": {
        "model": vgg16,
        "weights": VGG16_Weights,
        "valid_layers": [i for i in range(31)],
    },
    "resnet50": {
        "model": resnet50,
        "weights": ResNet50_Weights,
        "valid_layers": [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ],
    },
    "resnet101": {
        "model": resnet101,
        "weights": ResNet101_Weights,
        "valid_layers": [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ],
    },
}


class FeatureExtractor2D(nn.Module):
    def __init__(
        self, layers: Sequence[int] = [4, 9, 18], model_str: str = "vgg19"
    ) -> None:
        assert model_str in VALID_MODELS_2D, f"Model not in {VALID_MODELS_2D.keys()}"
        assert len(list(set(layers))) == len(layers), "Not valid repeated inputs"
        hist: List = []
        for layer in layers:
            valid_models: List[str] = VALID_MODELS_2D[model_str]["valid_layers"]
            num = valid_models.index(layer)
            hist.append(num)
        assert sorted(hist) == hist, "Not ordered inputs"
        super().__init__()
        self.model_str: str = model_str
        self.layers = list(map(str, layers))
        self.model = VALID_MODELS_2D[model_str]["model"](
            weights=VALID_MODELS_2D[model_str]["weights"].IMAGENET1K_V1
        )
        for param in self.model.parameters():
            param.requires_grad = False
        # Setting the transformation
        self.transform = VALID_MODELS_2D[model_str]["weights"].IMAGENET1K_V1.transforms(
            antialias=True
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        features = []

        if "vgg" in self.model_str:
            for name, layer in self.model.features.named_children():
                input = layer(input)
                if name in self.layers:
                    features.append(input)
                    if name == self.layers[-1]:
                        return features
        else:
            for name, layer in self.model.named_children():
                input = layer(input)
                if name in self.layers:
                    features.append(input)
                    if name == self.layers[-1]:
                        return features


VALID_MODELS_3D = {
    "vgg19": {
        "model": vgg19,
        "weights": VGG19_Weights,
        "valid_layers": [i for i in range(37)],
    },
    "vgg16": {
        "model": vgg16,
        "weights": VGG16_Weights,
        "valid_layers": [i for i in range(31)],
    },
    "resnet50": {
        "model": resnet50,
        "weights": ResNet50_Weights,
        "valid_layers": [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ],
    },
    "resnet101": {
        "model": resnet101,
        "weights": ResNet101_Weights,
        "valid_layers": [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ],
    },
}


class FeatureExtractor3D(nn.Module):
    def __init__(
        self, layers: Sequence[int] = [4, 9, 18], model_str: str = "vgg19"
    ) -> None:
        assert model_str in VALID_MODELS_3D, f"Model not in {VALID_MODELS_3D.keys()}"
        assert list(set(layers)) == layers, "Not valid repeated inputs"
        hist: List = []
        for layer in layers:
            valid_models: List[str] = VALID_MODELS_3D[model_str]["valid_layers"]
            num = valid_models.index(layer)
            hist.append(num)
        assert sorted(hist) == hist, "Not ordered inputs"
        super().__init__()
        self.model_str: str = model_str
        self.layers = list(map(str, layers))
        self.model = VALID_MODELS_3D[model_str]["model"](
            weights=VALID_MODELS_3D[model_str]["weights"].IMAGENET1K_V1
        )
        for param in self.model.parameters():
            param.requires_grad = False
        # Setting the transformation
        self.transform = VALID_MODELS_3D[model_str]["weights"].IMAGENET1K_V1.transforms(
            antialias=True
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        features = []

        if "vgg" in self.model_str:
            for name, layer in self.model.features.named_children():
                input = layer(input)
                if name in self.layers:
                    features.append(input)
                    if name == self.layers[-1]:
                        return features
        else:
            for name, layer in self.model.named_children():
                input = layer(input)
                if name in self.layers:
                    features.append(input)
                    if name == self.layers[-1]:
                        return features


__all__ = [
    "FeatureExtractor2D",
    "FeatureExtractor3D",
]
