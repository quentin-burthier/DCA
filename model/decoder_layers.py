"""Decoder."""

import torch.nn as nn


class LSTMLayer(nn.Module):
    """TODO Single layer LSTM.

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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, bidirectional=False)

    def forward(self, input_seq, states: tuple):
        dec_out, _ = self.lstm(input_seq, states)
        return dec_out
