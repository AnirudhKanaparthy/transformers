import tiktoken

from torch import Tensor
from typing import Union


class TikTokenizer:
    def __init__(self, tokenizer_name: str):
        self._encoder = tiktoken.get_encoding(tokenizer_name)
    
    def encode(self, x: Tensor, allowed_special: Union[set[str], None] = None):
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self._encoder.encode(x, allowed_special=allowed_special)

    def decode(self, x: Tensor):
        return self._encoder.decode(x)
    

class CharTokenizer:
    def __init__(self, vocabulary: list[str]):
        self._stoi = {ch: i for i, ch in enumerate(vocabulary)}
        self._itos = {i: ch for i, ch in enumerate(vocabulary)}

    def encode(self, x: Tensor, _: Union[set[str], None] = None): 
        return [self._stoi[ch] for ch in x]
    
    def decode(self, x: Tensor): 
        return ''.join([self._itos[i] for i in x])