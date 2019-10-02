"""Token indexers.

Convert array of strings to torch.Long
"""
from typing import Iterable, Dict

from collections import defaultdict
import torch


class TokenIndexer:

    def __init__(self, word2id: Dict[str, int], unk_idx: int):
        self.word2id = defaultdict(lambda: unk_idx, word2id)
        print("Tokenize", len(self.word2id))

    def __call__(self, tokens: Iterable[str]) -> torch.LongTensor:
        return torch.tensor([self.word2id[token] for token in tokens],
                            dtype=torch.long)
