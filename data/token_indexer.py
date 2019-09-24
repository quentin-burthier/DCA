"""Token indexers.

Convert array of strings to torch.Long
"""
from typing import List, Dict

from collections import defaultdict
import torch


class TokenIndexer:

    def __init__(self, word2id: Dict[str, int], unk_idx: int):
        self.word2id = defaultdict(lambda: unk_idx, word2id)

    def __call__(self, tokens: List[str]):
        return torch.tensor([self.word2id[token] for token in tokens],
                            dtype=torch.long)
