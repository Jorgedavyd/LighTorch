"""
Vision tasks:
Image:
- 2D-CNN patch embedding.
Video:
- 2D-CNN patch embedding.
- 3D-CNN patch embedding.
"""

from torch import Tensor, nn


class PatchEmbeddding3DCNN(nn.Module):
    def __init__(
        self,
        h_div: int,
        w_div: int,
        pe,
        feature_extractor,
        X: Tensor,
    ):
        super().__init__()
        """
        B: batch size
        F: Frames
        C: Channels
        H_div: number of vertical cuts
        W_div: number of horizontal cuts
        H_div*W_div: total patches
        h: H/H_div
        w: W/W_div
        """
        assert (
            X.size(-2) % self.h_div == 0 or X.size(-1) % self.w_div == 0
        ), "Patching size must be multiplier of dimention"
        self.H_div = h_div
        self.W_div = w_div
        self.feature_extractor = feature_extractor
        self.pe = pe

    def forward(self, X: Tensor):
        B, F, C, H, W = X.shape
        # -> (B*h_div*w_div, F,C,h, w)
        X = X.view(-1, F, C, H // self.H_div, W // self.W_div)
        # -> (B,h_div*w_div, embed_size)
        out = self.feature_extractor(X).view(B, X.size(0) / B, -1)
        # -> (B,h_div*w_div, embed_size)
        X = self.pe(out)
        return X


class PatchEmbedding2DCNN(nn.Module):
    def __init__(
        self,
        d_model: int,
        pe,
        feature_extractor,
        architecture: tuple,
        hidden_activations: tuple,
        dropout: float = 0.1,
    ):
        super().__init__()
        """
        B: batch size
        F: Frames
        C: Channels
        H_div: number of vertical cuts
        W_div: number of horizontal cuts
        H_div*W_div: total patches
        h: H/H_div
        w: W/W_div
        """
        self.d_model = d_model
        self.feature_extractor = feature_extractor(
            d_model, architecture, hidden_activations, dropout
        )
        self.pe = pe

    def forward(self, X: Tensor):
        B, F, C, H, W = X.shape
        # -> (B*F,C,H, W)
        X = X.view(B * F, C, H, W)
        # -> (B,F, embed_size)
        out = self.feature_extractor(X).view(B, F, -1)
        # -> (B,F, embed_size)
        X = self.pe(out)
        return X


__all__ = ["PatchEmbeddding3DCNN", "PatchEmbedding2DCNN"]
