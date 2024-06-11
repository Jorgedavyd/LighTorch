from torch import nn, Tensor
import torch.nn.functional as F
import torch
from einops import rearrange
from math import sqrt

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
        self,
        sequence_length: int,
        sequence_attention: bool = False,
        scale_factor: float = 1.0,
        flash_attention: bool = True,
        is_causal: bool = False,
        attn_mask: Tensor = None,
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        if is_causal:
            assert attn_mask is None, "You defined both attn_mask and is_causal"
            self.attn_mask = torch.ones(sequence_length, sequence_length).tril()
        self.seq = sequence_attention
        self.scale_factor = scale_factor
        self.flash = flash_attention

    def normal_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v, self.attn_mask, scale=self.scale_factor
                )
                return out

        energy = q @ k.transpose(-1, -2)

        if self.attn_mask is not None:
            energy.masked_fill_(self.attn_mask, -torch.inf)

        energy *= self.scale_factor

        return F.softmax(energy, dim=-1) @ v

    def seq_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.flash:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(
                    q.transpose(-1, -2),
                    k.transpose(-1, -2),
                    v.transpose(-1, -2),
                    self.attn_mask.transpose(-1, -2),
                    scale=self.scale_factor,
                )
                return out

        energy = q.transpose(-1, -2) @ k

        if self.attn_mask is not None:
            energy.masked_fill_(self.attn_mask.transpose(-1, -2), -torch.inf)

        energy *= self.scale_factor

        return (F.softmax(energy, dim=-1) @ v.transpose(-1, -2)).transpose(-1, -2)

    def attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.seq:
            return self.seq_attention(q, k, v)
        return self.normal_attention(q, k, v)


class SelfAttention(nn.Module):
    def __init__(self, attention: _AttentionBase) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, input: Tensor) -> Tensor:
        return self.attention(input, input, input)


class CrossAttention(nn.Module):
    def __init__(self, attention: _AttentionBase, method: str = "i i c") -> None:
        super().__init__()
        self.attention = attention
        self.method = method.lower()
        self.valid = {
            "i i c": lambda input, cross: self.attention(input, input, cross),
            "c c i": lambda input, cross: self.attention(cross, cross, input),
            "i c c": lambda input, cross: self.attention(input, cross, cross),
        }
        assert method in self.valid, "Not valid method"

    def forward(self, input: Tensor, cross: Tensor) -> Tensor:
        return self.valid[self.method](input, cross)


class GroupedQueryAttention(_AttentionBase):
    def __init__(
        self,
        embed_dim: int,
        sequence_length: int,
        n_queries: int,
        n_groups: int,
        kdim: int = None,
        vdim: int = None,
        is_causal: bool = False,
        attn_mask: Tensor = None,
        sequence_attention: bool = False,
        scale_factor: float = None,
        flash_attention: bool = False,
    ) -> None:
        if scale_factor is None:
            if sequence_attention:
                scale_factor: float = 1 / sqrt(sequence_length)
            else:
                scale_factor: float = 1 / sqrt(embed_dim)
        super(GroupedQueryAttention, self).__init__(
            sequence_length,
            sequence_attention,
            scale_factor,
            flash_attention,
            is_causal,
            attn_mask,
        )
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
        self.group_query_amount = n_queries // n_groups
        self.n_groups = n_groups

        # Projections
        ## (B, S, embed_dim) -> (B, S, embed_dim)
        self.Wq = nn.Linear(embed_dim, n_queries * self.query_dim)
        ## (B, S, embed_dim) -> (B, S, embed_dim // n_queries) for each head, we have a single key and value
        self.Wk = nn.Linear(kdim, n_groups * self.query_dim)
        self.Wv = nn.Linear(vdim, n_groups * self.query_dim)

        self.fc = nn.Linear(vdim, vdim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Reshaping for broadcasting
        q = rearrange(
            self.Wq(q),
            "b s (g q d) -> b g q s d",
            q=self.group_query_amount,
            g=self.n_groups,
        )  # -> (B, S, vdim) -> (B, n_groups, n_queries, S, query_dim)
        k = rearrange(self.Wk(k), "b s (g d) -> b g s d", g=self.n_groups).unsqueeze(
            2
        )  # -> (B, n_groups, S, query_dim)
        v = rearrange(self.Wv(v), "b s (g d) -> b g s d", g=self.n_groups).unsqueeze(
            2
        )  # -> (B, n_groups, S, query_dim)

        print(q.shape, k.shape, v.shape)

        out = rearrange(self.attention(q, k, v), "b g q s d -> b s (g q d)")

        return self.fc(out)


class MultiHeadAttention(_AttentionBase):
    def __init__(
        self,
        embed_dim: int,
        sequence_length: int,
        n_heads: int,
        kdim: int = None,
        vdim: int = None,
        is_causal: bool = False,
        attn_mask: bool = None,
        sequence_attention: bool = False,
        scale_factor: float = None,
        flash_attention: bool = False,
    ) -> None:
        if scale_factor is None:
            if sequence_attention:
                scale_factor: float = 1 / sqrt(sequence_length)
            else:
                scale_factor: float = 1 / sqrt(embed_dim)
        super(MultiHeadAttention, self).__init__(
            sequence_length,
            sequence_attention,
            scale_factor,
            flash_attention,
            is_causal,
            attn_mask,
        )
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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Defining the shapes
        input_shape = v.shape
        q_shape = (*q.shape[:2], self.n_heads, self.head_dim)
        k_shape = (*k.shape[:2], self.n_heads, self.head_dim)
        v_shape = (*v.shape[:2], self.n_heads, self.head_dim)

        # Projecting and reshaping for hidden attention -> (B,HEADS, S, HEAD_DIM)
        q = self.Wq(q).view(*q_shape).transpose(1, 2)
        k = self.Wk(k).view(*k_shape).transpose(1, 2)
        v = self.Wv(v).view(*v_shape).transpose(1, 2)

        out = self.attention(q, k, v).transpose(1, 2).reshape(input_shape)

        return self.fc(out)


class MultiQueryAttention(_AttentionBase):
    def __init__(
        self,
        embed_dim: int,
        sequence_length: int,
        n_queries: int,
        kdim: int = None,
        vdim: int = None,
        is_causal: bool = False,
        attn_mask: Tensor = None,
        sequence_attention: bool = False,
        scale_factor: float = None,
        flash_attention: bool = False,
    ) -> None:
        if scale_factor is None:
            if sequence_attention:
                scale_factor: float = 1 / sqrt(sequence_length)
            else:
                scale_factor: float = 1 / sqrt(embed_dim)
        super(MultiQueryAttention, self).__init__(
            sequence_length,
            sequence_attention,
            scale_factor,
            flash_attention,
            is_causal,
            attn_mask,
        )
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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
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
        out = self.attention(q, k, v).transpose(1, 2).reshape(input_shape)
        # Last linear projection
        return self.fc(out)


__all__ = [
    "MultiQueryAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "SelfAttention",
    "CrossAttention",
]
