import tiktoken

from torch import Tensor
from typing import Union


class TikTokenizer:
    def __init__(self, tokenizer_name: str):
        self._encoder = tiktoken.get_encoding(tokenizer_name)
    
    def encode(self, x: Tensor, allowed_special: Union[set[str], None] = None) -> Tensor:
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self._encoder.encode(x, allowed_special=allowed_special)

    def decode(self, x: Tensor) -> str:
        return self._encoder.decode(x)
    
    def vocabulary_size(self) -> int:
        return self._encoder.max_token_value + 1
    

class CharTokenizer:
    def __init__(self, vocabulary: list[str]):
        self._stoi = {ch: i for i, ch in enumerate(vocabulary)}
        self._itos = {i: ch for i, ch in enumerate(vocabulary)}

    def encode(self, x: Tensor, _: Union[set[str], None] = None) -> Tensor:
        return [self._stoi[ch] for ch in x]
    
    def decode(self, x: Tensor) -> str:
        return ''.join([self._itos[i] for i in x])
    
    def vocabulary_size(self) -> int:
        return len(self._stoi)