"""Decoder of the multi-agents summarizer."""
from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class Decoder(nn.Module):
    """Multi-agents decoder.

        Args:
            n_agents (int):
            decoder (nn.Module):
            mlp (nn.Module):
            word_attention ([type]):
            agent_attention ([type]):

        Input:
            prev_input (PackedSequence[tgt_len, bsz, emb_dim]):
                Previously decoded words.
                tgt_len = 1 in recursive prediction mode.
            encoded_seq (Tensor[src_len, bsz, n_agents, hsz])
            init_state Tuple(Tensor[bsz, hsz], Tensor[bsz, hsz])

        Output:
            vocab_probs (Tensor[bsz, tgt_len, voc_sz])
            generation_probs (Tensor[bsz, tgt_len, n_agents])
            agentwise_attn (Tensor[bsz*n_agents, tgt_len, hsz])
            agent_attn (Tensor[bsz*tgt_len, n_agents, 1])
    """

    def __init__(self, decoder_layer: nn.Module, vocab_predictor: VocabPredictor,
                 generator: Generator, word_attention, agent_attention):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.word_attn = word_attention
        self.agent_attn = agent_attention
        self.vocab_predictor = vocab_predictor
        self.generator = generator

    def forward(self, prev_input: PackedSequence,
                encoded_seq: Tensor, init_state: Tuple[Tensor, Tensor]):

        src_len, bsz, n_agents, hsz = encoded_seq.shape

        encoded_seq = encoded_seq.view(src_len, bsz*n_agents, hsz)
        encoded_seq = encoded_seq.transpose(0, 1)  # [bsz*n_agents, src_len, hsz]

        state = self.decoder_layer(prev_input, init_state)  # [tgt_len, bsz, hsz]
        state = state.transpose(0, 1)  # [bsz, tgt_len, hsz]

        tgt_len = state.shape[1]

        ## Agent context
        agentwise_attn = self.word_attn(
            state.unsqueeze(1) \
                 .expand(-1, n_agents, -1, -1) \
                 .reshape(bsz*n_agents, tgt_len, hsz),
            encoded_seq
        )
        # [bsz*n_agents, tgt_len, src_len]

        agentwise_context = (agentwise_attn.unsqueeze(-1)
                             * encoded_seq.unsqueeze(1).expand(-1, tgt_len, -1, -1)
                            ).sum(dim=2)
        # (
        #  [bsz*n_agents, tgt_len, src_len, 1] * [bsz*n_agents, tgt_len, src_len, hsz]
        # ).sum(dim=2)
        # -> [bsz*n_agents, tgt_len, src_len, hsz].sum(dim=2)
        # -> [bsz*n_agents, tgt_len, hsz]
        agentwise_context = agentwise_context.view(bsz, n_agents, tgt_len, hsz) \
                                             .transpose(1, 2) \
                                             .reshape(bsz*tgt_len, n_agents, hsz)

        ## Global context
        agent_attn = self.agent_attn(state.reshape(bsz*tgt_len, 1, hsz),
                                     agentwise_context)
        # [bsz*tgt_len, 1, n_agents]
        agent_attn = agent_attn.transpose(1, 2)  # [bsz*tgt_len, n_agents, 1]

        global_context = (agent_attn * agentwise_context).sum(dim=1)
        # ([bsz*tgt_len, n_agents, 1]*[bsz*tgt_len, n_agents, hsz]).sum(dim=1)
        # -> [bsz*tgt_len, n_agents, hsz].sum(dim=1)
        # -> [bsz*tgt_len, hsz]
        global_context = global_context.view(bsz, tgt_len, hsz)

        vocab_probs = self.vocab_predictor(state, global_context)
        # [bsz, tgt_len, voc_sz]

        pred_output, _ = pad_packed_sequence(prev_input, batch_first=True)
        # Pointer-Generator
        generation_probs = self.generator(
            agentwise_context.view(bsz, tgt_len, n_agents, hsz),
            state,
            pred_output
        )
        # [bsz, tgt_len, n_agents]

        return vocab_probs, generation_probs, agentwise_attn, agent_attn


class Generator(nn.Module):
    """Generator network.

    Computes the probability of generating a word from the model vocabulary.

    Args:
        hidden_size: int
        embedding_dim

    Input:
        agentwise_context (Tensor[bsz, tgt_len, n_agents, hsz]):
        state (Tensor[bsz, tgt_len, hsz]):
        predicted (Tensor[bsz, tgt_len, embedding_dim]): Predicted output (or
            ground truth output in teacher forcing mode)

    Output:
        generation_probs (Tensor[[bsz, tgt_len, n_agents]])
    """

    def __init__(self, hidden_size: int, embedding_dim: int):
        super().__init__()
        self.context_nn = nn.Linear(hidden_size, 1, bias=False)
        self.state_pred_nn = nn.Linear(hidden_size + embedding_dim, 1,
                                       bias=True)

    def forward(self, agentwise_context, state, predicted) -> torch.Tensor:
        context_importance = self.context_nn(agentwise_context).squeeze(-1)
        # [bsz, tgt_len, n_agents]
        state_pred_importance = self.state_pred_nn(torch.cat((state, predicted),
                                                             dim=-1))
        # [bsz, tgt_len, 1]
        return torch.sigmoid(context_importance + state_pred_importance)


class VocabPredictor(nn.Module):
    """Computes the probabilities over the vocabulary.

    Args:
        hidden_size (int)
        vocab_size (int)

    Input:
        state (Tensor[bsz, tgt_len, hsz])
        global_context (Tensor[bsz, tgt_len, hsz])

    Ouput:
        Tensor[bsz, tgt_len, vocab_size]
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()

        self.mlp = nn.Linear(2*hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: Tensor, global_context: Tensor) -> Tensor:
        logits = self.mlp(torch.cat((state, global_context), dim=-1))
        return self.softmax(logits)
