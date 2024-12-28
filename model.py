import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Union


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
                 dropout: float = 0.5
                 ):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)


class Block(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 number_of_heads: int,
                 context_length: int,
                 dropout: float = 0.5):
        super().__init__()
        head_size = embedding_dim // number_of_heads
        self._sa = MultiHeadAttention(
            number_of_heads, head_size, embedding_dim, context_length, dropout)
        self._ffwd = FeedForward(embedding_dim, dropout)

        self._ln1 = nn.LayerNorm(embedding_dim)
        self._ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Added Residual Connections
        # We also do pre tranformation normalization
        x = x + self._sa(self._ln1(x))
        x = x + self._ffwd(self._ln2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dim: int,
                 context_length: int,
                 number_of_layers: int,
                 number_of_heads: int = 1,
                 dropout: float = 0.5,
                 device: str = 'cpu'):
        super().__init__()
        self._device = device
        self._context_length = context_length
        self._token_embeddings_table = nn.Embedding(
            vocabulary_size, embedding_dim)
        self._position_embeddings_table = nn.Embedding(
            context_length, embedding_dim)

        # We essentially repeat the above multiple times.
        self._blocks = nn.Sequential(
            *[Block(embedding_dim, number_of_heads, self._context_length, dropout) for _ in range(number_of_layers)])
        self._layernorm = nn.LayerNorm(embedding_dim)
        self._lm_head = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, idx: Tensor, targets: Union[Tensor, None] = None) -> tuple[Tensor, Union[Tensor, None]]:
        # idx and targets are tensors of shape (Batch Size, Context Length)
        # (B, T)
        B, T = idx.shape

        token_embd = self._token_embeddings_table(idx)  # (B, T, C)
        pos_embd = self._position_embeddings_table(
            torch.arange(T, device=self._device))
        x = token_embd + pos_embd

        x = self._blocks(x)
        x = self._layernorm(x)

        logits = self._lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_next_tokens: int) -> Tensor:
        for _ in range(max_next_tokens):
            idx_cond = idx[:, -self._context_length:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
