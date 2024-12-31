import torch
from torch import nn, Tensor
from torch.nn import functional as F

from transformers.blocks import DecoderBlock


class SimpleDecoderTransformer(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dim: int,
                 context_length: int,
                 number_of_layers: int,
                 number_of_heads: int = 1,
                 linear_layer_scale: int = 4,
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
            *[DecoderBlock(embedding_dim, number_of_heads, self._context_length, linear_layer_scale, dropout) for _ in range(number_of_layers)])
        self._layernorm = nn.LayerNorm(embedding_dim)
        self._lm_head = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, idx: Tensor) -> Tensor:
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
        return logits

    def calculate_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)
        return loss

    def generate(self, idx: Tensor, max_next_tokens: int) -> Tensor:
        for _ in range(max_next_tokens):
            # The input contains batches. We take in all the batches
            # and within those branches we take in the last `context_length` number of token
            idx_cond = idx[:, -self._context_length:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx