from torch import nn, Tensor
import torch.nn.functional as F
import torch
from einops import rearrange

"""
Types:
- GroupedQueryAttention
- MultiQueryAttention
- DefaultMultiHeadAttention
- SingleHeadAttention

You can use it with:
- SelfAttention
- CrossAttention

"""


class _AttentionBase(nn.Module):
    def __init__(
        self, seq: bool = False, scale_factor: float = 1, flash: bool = True
    ) -> None:
        super().__init__()
        self.seq = seq
        self.scale_factor = scale_factor
        self.flash = flash

    def normal_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> Tensor:
        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v, mask, scale=self.scale_factor
                )
            return out

        energy = q @ k.transpose(-1, -2)

        if mask is not None:
            energy.masked_fill_(mask, -torch.inf)

        energy *= self.scale_factor

        return F.softmax(energy) @ v

    def seq_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> Tensor:
        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(
                    q.transpose(-1, -2),
                    k.transpose(-1, -2),
                    v.transpose(-1, -2),
                    mask.transpose(-1, -2),
                    scale=self.scale_factor,
                )
            return out

        energy = q.transpose(-1, -2) @ k

        if mask is not None:
            energy.masked_fill_(mask.transpose(-1, -2), -torch.inf)

        energy *= self.scale_factor

        return (F.softmax(energy) @ v.transpose(-1, -2)).transpose(-1, -2)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        if self.seq:
            return self.seq_attention(q, k, v, mask, self.scale_factor)
        return self.normal_attention(q, k, v, mask, self.scale_factor)


class SelfAttention(nn.Module):
    def __init__(self, type: _AttentionBase, *args, **kwargs) -> None:
        super().__init__()
        self.attention = type(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(input, input, input)


class CrossAttention(nn.Module):
    def __init__(
        self, type: _AttentionBase, method: str = "i i c", *args, **kwargs
    ) -> None:
        super().__init__()
        self.attention = type(*args, **kwargs)
        self.method = method.lower()

    def forward(self, input: Tensor, cross: Tensor) -> Tensor:
        match self.method:
            case "i i c":
                return self.attention(input, input, cross)
            case "c c i":
                return self.attention(cross, cross, input)
            case "i c c":
                return self.attention(input, cross, cross)

        raise ValueError(f"Not valid method: {self.method}")


class GroupedQueryAttention(_AttentionBase):
    def __init__(
        self,
        embed_dim: int,
        n_queries: int,
        n_groups: int,
        kdim: int = None,
        vdim: int = None,
        **kwargs,
    ) -> None:
        super(MultiQueryAttention, self).__init__(**kwargs)
        # Defining the hidden spaces
        if vdim is None:
            vdim = embed_dim
        if kdim is None:
            kdim = embed_dim

        assert (
            vdim % n_queries == 0
        ), "Not valid number of heads, not divisible by embedding dimension or values dimension"
        assert (
            n_queries % n_groups == 0
        ), "The number of query heads should be divisible by the number of groups"

        self.query_heads = n_queries
        self.query_dim = vdim // n_queries
        self.n_groups = n_groups

        # Projections
        ## (B, S, embed_dim) -> (B, S, embed_dim)
        self.Wq = nn.Linear(embed_dim, vdim)
        ## (B, S, embed_dim) -> (B, S, embed_dim // n_queries) for each head, we have a single key and value
        self.Wk = nn.Linear(kdim, n_groups * self.query_dim)
        self.Wv = nn.Linear(vdim, n_groups * self.query_dim)

        self.fc = nn.Linear(vdim, vdim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        # Reshaping for broadcasting
        q = rearrange(
            self.Wq(q), "b s (g q d) -> b g q s d", q=self.query_heads, g=self.n_groups
        )  # -> (B, S, vdim) -> (B, n_groups, n_queries, S, query_dim)
        k = rearrange(
            self.Wk(k), "b s (g d) -> b g s d", g=self.n_groups
        )  # -> (B, n_groups, S, query_dim)
        v = rearrange(
            self.Wv(v), "b s (g d) -> b g s d", g=self.n_groups
        )  # -> (B, n_groups, S, query_dim)

        out = rearrange(self.attention(q, k, v, mask), "b g q s d -> b s (g q d)")

        return self.fc(out)


class MultiHeadAttention(_AttentionBase):
    def __init__(
        self, embed_dim: int, n_heads: int, kdim: int = None, vdim: int = None, **kwargs
    ) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        # Defining hidden spaces
        if vdim is None:
            vdim = embed_dim
        if kdim is None:
            kdim = embed_dim

        assert (
            vdim % n_heads == 0
        ), "Not valid number of heads, not divisible by embedding dimension or values dimension"

        # Projections
        self.Wq = nn.Linear(embed_dim, vdim)
        self.Wk = nn.Linear(kdim, vdim)
        self.Wv = nn.Linear(vdim, vdim)
        self.fc = nn.Linear(vdim, vdim)

        self.vdim = vdim
        self.kdim = kdim
        self.qdim = embed_dim
        self.n_heads = n_heads
        self.head_dim = vdim // n_heads

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        # Defining the shapes
        input_shape = v.shape
        q_shape = (*q.shape[:2], self.n_heads, self.head_dim)
        k_shape = (*k.shape[:2], self.n_heads, self.head_dim)
        v_shape = (*v.shape[:2], self.n_heads, self.head_dim)

        # Projecting and reshaping for hidden attention -> (B,HEADS, S, HEAD_DIM)
        q = self.Wq(q).view(*q_shape).transpose(1, 2)
        k = self.Wk(k).view(*k_shape).transpose(1, 2)
        v = self.Wv(v).view(*v_shape).transpose(1, 2)

        out = self.attention(q, k, v, mask)

        out = self.vdim(v).transpose(1, 2).reshape(input_shape)

        return self.fc(out)


class MultiQueryAttention(_AttentionBase):
    def __init__(
        self,
        embed_dim: int,
        n_queries: int,
        kdim: int = None,
        vdim: int = None,
        **kwargs,
    ) -> None:
        super(MultiQueryAttention, self).__init__(**kwargs)
        if vdim is None:
            vdim = embed_dim
        if kdim is None:
            kdim = embed_dim

        assert (
            vdim % n_queries == 0
        ), "Not valid number of heads, not divisible by embedding dimension or values dimension"

        self.query_heads = n_queries
        self.query_dim = vdim // n_queries

        # Projections
        ## (B, S, embed_dim) -> (B, S, embed_dim)
        self.Wq = nn.Linear(embed_dim, vdim)
        ## (B, S, embed_dim) -> (B, S, embed_dim // n_queries) for each head, we have a single key and value
        self.Wk = nn.Linear(kdim, self.query_dim)
        self.Wv = nn.Linear(vdim, self.query_dim)

        self.fc = nn.Linear(vdim, vdim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        # Defining shapes
        input_shape = v.shape
        q_shape = (*q.shape[:2], self.query_heads, self.query_dim)
        k_shape = (*k.shape[:2], 1, self.query_dim)
        v_shape = (*v.shape[:2], 1, self.query_dim)
        # Reshaping for broadcasting
        q = (
            self.Wq(q).view(*q_shape).transpose(1, 2)
        )  # -> (B, s, embed_dim) -> (B, query_heads, S, query_dim)
        k = self.Wk(k).view(*k_shape).transpose(1, 2)  # -> (B, 1, S, query_dim)
        v = self.Wv(v).view(*v_shape).transpose(1, 2)  # -> (B, 1, S, query_dim)
        # Performing attention by broadcasting and getting back to the original shape
        out = self.attention(q, k, v, mask).transpose(1, 2).reshape(input_shape)
        # Last linear projection
        return self.fc(out)


__all__ = [
    "MultiQueryAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "SelfAttention",
    "CrossAttention",
]
