"""Encoder of the multi-agents summarizer."""
from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class MultiAgentsEncoder(nn.Module):
    """Multi-agents encoder.

    Args:
        n_agents (int):
        embedding_layer (nn.Embedding):
        local_layer (nn.Module): expected to do the concatenation of all directions
        contextual_layer (nn.ModuleList):
        msg_projector (nn.Module):

    Input:
        article (PackedSequence)

    Outputs:
        encoded_article (Tensor([src_len, bsz, n_agents, hsz])
        last_h_of_1st_agt (Tensor[bsz, hsz]): hidden state of first agent last
                                              layer
    """

    def __init__(self, n_agents: int,
                 local_layer: nn.Module, contextual_layers: nn.ModuleList,
                 bidir_hs_proj: nn.Module, msg_projector: MessageProjector):

        super().__init__()
        self.n_agents = n_agents

        # All agents weights are shared.
        self.local_layer = local_layer
        self.contextual_layers = contextual_layers
        self.bidir_hs_proj = bidir_hs_proj
        self.msg_projector = msg_projector

        self.mask_matrix = torch.ones(self.n_agents, self.n_agents, dtype=torch.uint8)
        self.mask_matrix -= torch.eye(self.n_agents, dtype=torch.uint8)

    def forward(self,
                article: PackedSequence
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Local encoding of each article
        (prev_layer_enc, seq_lenghts), prev_hs, _ = self.local_layer(article)
        # ([src_len, bsz*n_agents, 2*hsz],
        #  [bsz*n_agents, 2*hsz])
        prev_layer_enc = self.bidir_hs_proj(prev_layer_enc) # [src_len, bsz*n_agents, hsz]
        prev_hs = self.bidir_hs_proj(prev_hs) # [bsz*n_agents, hsz]
        bsz = prev_hs.shape[0] // self.n_agents

        # Contextual encoding
        for contextual_layer in self.contextual_layers:
            prev_hs = prev_hs.view(bsz, self.n_agents, -1)  # [bsz, n_agents, hsz]

            # Get the messages
            message = [prev_hs[:, self.mask_matrix[agt_id], :].mean(dim=1)
                       for agt_id in range(self.n_agents)]  # n_agents*[bsz, hsz]
            message = torch.stack([
                torch.stack([message[agt_id][batch]  # hsz
                             for agt_id in range(self.n_agents)])  # [n_agents, hsz]
                for batch in range(bsz)
            ])  # [bsz, n_agents, hsz]

            # Project the messages
            next_layer_in = self.msg_projector(prev_layer_enc,
                                               message.view(bsz*self.n_agents, -1))
            # [src_len, bsz*n_agents, hsz]

            # Feed the projection to the agents
            next_layer_in = pack_padded_sequence(next_layer_in, seq_lenghts,
                                                 enforce_sorted=False)
            contextual_layer_out = contextual_layer(next_layer_in)
            (prev_layer_enc, _), prev_hs, cell_state = contextual_layer_out

            # Projection of the LSTM directions
            prev_layer_enc = self.bidir_hs_proj(prev_layer_enc) # [src_len, bsz*n_agents, hsz]
            prev_hs = self.bidir_hs_proj(prev_hs) # [bsz*n_agents, hsz]

        src_len = prev_layer_enc.shape[0]
        prev_layer_enc = prev_layer_enc.view(src_len, bsz, self.n_agents, -1)
        # [src_len, bsz, n_agents, hsz]

        last_h_of_1st_agt = prev_hs.view(bsz, self.n_agents, -1)
        last_h_of_1st_agt = last_h_of_1st_agt[:, 0, :].unsqueeze(0)
        # [1, bsz, hsz]
        cell_state = cell_state.view(bsz, self.n_agents, -1)
        cell_state = cell_state[:, 0, :].unsqueeze(0)
        # [1, bsz, hsz]

        return prev_layer_enc, (last_h_of_1st_agt, cell_state)


class MessageProjector(nn.Module):
    """Attention-like message projector.

    Args:
        hidden_size (int)

    Inputs:
        encoded_seq (Tensor[src_len, bsz*n_agents, hsz])
        message (Tensor[bsz*n_agents, hsz])

    Ouput:
        Tensor[src_len, bsz*n_agents, hsz]
    """

    def __init__(self, hidden_size: int) -> torch.Tensor:
        super().__init__()
        self.msg_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.seq_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, encoded_seq, message):
        message = self.msg_layer(message).unsqueeze(0) # [1, bsz*n_agents, hsz]
        encoded_seq = self.seq_layer(encoded_seq) # [src_len, bsz*n_agents, hsz]
        return torch.tanh(message + encoded_seq) # [src_len, bsz*n_agents, hsz]
