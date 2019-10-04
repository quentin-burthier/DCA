"""Test the decoder."""
from typing import Tuple

import random
import pytest

import torch
from torch.nn.utils.rnn import pack_sequence

from settings.model import build_decoder


@pytest.mark.parametrize(
    "n_agents, embedding_dim, hsz, voc_sz, bsz, src_len, prev_len_range",
    [(3, 64, 32, 500, 7, 111, (20, 40))])
def test_decoder(n_agents: int, embedding_dim: int,
                 hsz: int, voc_sz: int, bsz: int, src_len: int,
                 prev_len_range: Tuple[int, int]):
    decoder = build_decoder(hidden_size=hsz, embedding_dim=embedding_dim,
                            vocab_size=voc_sz)

    encoded_seq = torch.rand(src_len, bsz, n_agents, hsz)
    init_state = torch.rand(bsz, hsz)

    tgt_len = prev_len_range[1]
    prev_input = [torch.rand(random.randint(*prev_len_range), embedding_dim)
                  for _ in range(bsz-1)]
    prev_input.append(torch.rand(tgt_len, embedding_dim))
    prev_input = pack_sequence(prev_input, enforce_sorted=False)

    decoder_out = decoder(prev_input, encoded_seq, init_state)
    vocab_probs, generation_probs, agentwise_attn, agent_attn = decoder_out

    assert list(vocab_probs.shape) == [bsz, tgt_len, voc_sz], (
        f"{vocab_probs.shape} != {[bsz, tgt_len, voc_sz]}"
    )
    assert list(generation_probs.shape) == [bsz, tgt_len, n_agents], (
        f"{generation_probs.shape} != {[bsz, tgt_len]}"
    )
    assert list(agentwise_attn.shape) == [src_len, bsz, n_agents], (
        f"{agentwise_attn.shape} != {[src_len, bsz, n_agents]}"
    )
    assert list(agent_attn.shape) == [src_len, bsz], (
        f"{agent_attn.shape} != {[src_len, bsz]}"
    )
