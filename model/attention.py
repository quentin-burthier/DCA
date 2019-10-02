""" Attention functions."""

import torch
import torch.nn as nn

def dot_product(query, key):
    """Luong dot-product attention.

    Args:
        query (Tensor[bsz, tgt_len, hsz])
        key (Tensor[bsz, src_len, hsz])

    Returns:
        Tensor[bsz, tgt_len, src_len]
    """
    return torch.softmax(query @ key.transpose(1, 2), dim=-1)


class GeneralAttention(nn.Module):
    """Luong general attention.

    Args:
        enc_hsz (int):
        dec_hsz (int):

    Inputs:
        query (Tensor[bsz, tgt_len, hsz])
        key (Tensor[bsz, src_len, hsz])

    Returns:
        Tensor[bsz, tgt_len, src_len]
    """

    def __init__(self, query_hsz: int, key_hsz: int):
        super().__init__()
        self.weights = nn.Parameter(
            nn.init.xavier_uniform(torch.empty(query_hsz, key_hsz))
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key):
        return self.softmax(query @ self.weights @ key.transpose(1, 2))


class AdditiveAttention(nn.Module):
    """Bahdanau attention.

    Args:
        hidden_size (int)

    Inputs:
        query (Tensor[bsz, tgt_len, hsz])
        key (Tensor[bsz, src_len, hsz])

    Returns:
        Tensor[bsz, tgt_len, src_len]
    """

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.vT = nn.Linear(hidden_size, 1)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key):
        query = self.query_layer(query).unsqueeze(2)  # [bsz, tgt_len, 1, hsz]
        key = self.key_layer(key).unsqueeze(1)  # [bsz, 1, src_len, hsz]
        attn_energies = torch.tanh(query + key) # [bsz, tgt_len, src_len, hsz]
        attn_energies = self.vT(attn_energies).squeeze(-1)
        # [bsz, tgt_len, src_len]
        return self.softmax(attn_energies)  # [bsz, tgt_len, src_len]
