"""Decoder layers."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class LSTMLayer(nn.Module):
    """Single layer LSTM.

    Args:
        input_size (int): dimension of input sequence elements
        hidden_size (int): dimension of the LSTM hidden size

    Input:
        input_seq (PackedSequence): shifted gold summaries

    Output:
        dec_out (Tensor[tgt_len, bsz, hsz])
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, bidirectional=False)

    def forward(self, input_seq: PackedSequence, states: tuple) -> torch.Tensor:
        dec_out, _ = self.lstm(input_seq, states)
        dec_out, _ = pad_packed_sequence(dec_out)
        return dec_out
