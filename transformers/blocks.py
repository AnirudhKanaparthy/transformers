import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self,
                 head_size: int,
                 embedding_dim: int,
                 context_length: int,
                 dropout: float = 0.5,
                 ):
        super().__init__()

        self._head_size = head_size
        self._key = nn.Linear(embedding_dim, head_size, bias=False)
        self._query = nn.Linear(embedding_dim, head_size, bias=False)
        self._value = nn.Linear(embedding_dim, head_size, bias=False)

        # `_tril` won't be trained but needs to be in the `state_dict`
        self.register_buffer('_tril', torch.tril(
            torch.ones(context_length, context_length)))

        self._dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # This is the core of the Transformer architecture
        B, T, C = x.shape

        k = self._key(x)
        q = self._query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self._tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self._dropout(wei)

        v = self._value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 number_of_heads: int,
                 head_size: int,
                 embedding_dim: int,
                 context_length: int,
                 dropout: float = 0.5,
                 ):
        super().__init__()
        self._heads = nn.ModuleList(
            [Head(head_size, embedding_dim, context_length, dropout) for _ in range(number_of_heads)])
        self._proj = nn.Linear(embedding_dim, embedding_dim)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.concat([h(x) for h in self._heads], dim=-1)
        out = self._proj(out)
        return self._dropout(out)


class FeedForward(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 scale: int = 4,
                 dropout: float = 0.5
                 ):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(embedding_dim, scale * embedding_dim),
            nn.ReLU(),
            nn.Linear(scale * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 number_of_heads: int,
                 context_length: int,
                 linear_layer_scale: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        # The embedding dimension should be divisible by the number of heads
        head_size = torch.math.ceil(embedding_dim / number_of_heads)
        self._embedding_dim = head_size * number_of_heads

        self._sa = MultiHeadAttention(
            number_of_heads, head_size, self._embedding_dim, context_length, dropout)
        self._ffwd = FeedForward(
            self._embedding_dim, linear_layer_scale, dropout)

        self._ln1 = nn.LayerNorm(self._embedding_dim)
        self._ln2 = nn.LayerNorm(self._embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Added Residual Connections
        # We also do pre tranformation normalization
        x = x + self._sa(self._ln1(x))
        x = x + self._ffwd(self._ln2(x))
        return x
