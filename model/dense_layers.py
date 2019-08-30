"""Dense layers for probability predictions."""

import torch
import torch.nn as nn


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
        self.state_pred_nn = nn.Linear(hidden_size + embedding_dim, 1, bias=True) 

    def forward(self, agentwise_context, state, predicted):
        context_importance = self.context_nn(agentwise_context).squeeze(-1)
        # [bsz, tgt_len, n_agents]
        state_pred_importance = self.state_pred_nn(torch.cat((state, predicted), dim=-1))
        # [bsz, tgt_len, 1]
        return torch.sigmoid(context_importance + state_pred_importance)
