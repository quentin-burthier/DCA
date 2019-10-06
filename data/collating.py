"""Collate functions."""

from typing import List, Tuple

from cytoolz import concatv

import torch
from torch import Tensor

from torch.nn.utils.rnn import pad_sequence


def collate_by_padding(
        batch: List[Tuple[Tensor, Tensor]],
        padding_value: int = 0
    ):
    """Collates a batch of sequences.

    Args:
        batch (List[Tuple[LongTensor, LongTensor]])
        padding_value (int)

    Returns:
        articles, articles_len
        prev_input, prev_inputs_len
        gold_summaries
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)  # sort by decreasing article
                                                       # length

    articles, prev_inputs, gold_summaries = zip(*batch)
    n_agents = articles[0].shape[1]
    articles_len = torch.tensor(list(concatv(*(n_agents*[len(article)]
                                               for article in articles))),
                                dtype=torch.long)
    articles = pad_sequence(articles, padding_value=padding_value)

    prev_inputs_len = torch.tensor([len(prev_input)
                                    for prev_input in prev_inputs],
                                   dtype=torch.long)
    prev_inputs = pad_sequence(prev_inputs, padding_value=padding_value)

    gold_summaries = pad_sequence(gold_summaries, padding_value=padding_value)

    return (articles, articles_len), (prev_inputs, prev_inputs_len), gold_summaries
