"""Encoders layers."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class BiLSTMLayer(nn.Module):
    """Single layer bi-LSTM.

    Args:
        input_size (int): dimension of input sequence elements
        hidden_size (int): dimension of the LSTM hidden size

    Input:
        input_seq (Tensor): batch of articles, split by n_agents paragraphs
        seq_lenghts (LongTensor): lenghts of the paragraphs

    Output:
        encoded_seq (Tensor)
        message (Tensor[bsz, 2*hsz])
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bidirectional=True)

    def forward(self, input_seq: PackedSequence) -> Tuple[Tuple[torch.Tensor,
                                                                torch.LongTensor],
                                                          torch.Tensor]:

        encoded_seq, _ = self.bi_lstm(input_seq)

        encoded_seq, seq_lenghts = pad_packed_sequence(encoded_seq)
        seq_len, bsz, bi_hsz = encoded_seq.shape

        separated_directions = encoded_seq.view(seq_len, bsz, 2, bi_hsz // 2)
        message = torch.cat((separated_directions[-1, :, 0, :],
                             separated_directions[0, :, 1, :]),
                            dim=-1)  # [bsz, 2*hsz]
        return (encoded_seq, seq_lenghts), message
