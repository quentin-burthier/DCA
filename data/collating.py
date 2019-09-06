"""Collate functions."""

from typing import List, Tuple
from torch import Tensor

from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence


def collate_by_packing(
        batch: List[Tuple],
        padding_value: float = 0
    ) -> Tuple[PackedSequence, PackedSequence, Tensor]:
    """Collates a batch of sequences.

    Args:
        batch (List[Tuple]): [description]
        padding_value ([type]): [description]

    Returns:
        articles
        prev_input
        gold_summaries
    """

    articles, prev_inputs, gold_summaries = zip(*batch)

    articles = pack_sequence(articles, enforce_sorted=False)
    prev_inputs = pack_sequence(prev_inputs, enforce_sorted=False)
    gold_summaries = pad_sequence(gold_summaries, padding_value=padding_value)

    return articles, prev_inputs, gold_summaries
