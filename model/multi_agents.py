"""Multi-agents summarizer."""

import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .multi_agents_encoder import MultiAgentsEncoder
from .decoder import Decoder


class MultiAgentsSummarizer(nn.Module):
    """Multi-Agents Summarizer.

    Args:
        embedding_layer (nn.Embedding):
        multi_agt_encoder (MultiAgentsEncoder):
        decoder (Decoder):

    Inputs:
        article (Tensor): [description]
        article_length (Tensor): [description]
        prev_input (LongTensor): [description]
        prev_input_length (LongTensor): [description]

    Output:
        extended_voc_probs (Tensor[tgt_len, bsz, extended_voc_sz])
    """

    def __init__(self, embedding_layer: nn.Embedding,
                 multi_agt_encoder: MultiAgentsEncoder, decoder: Decoder):
        super().__init__()
        self.embedding = embedding_layer
        self.multi_agt_encoder = multi_agt_encoder
        self.decoder = decoder

    def forward(self, article: Tensor, article_length: Tensor,
                prev_input: LongTensor, prev_input_length: LongTensor):

        src_len, bsz, n_agents = article.shape
        tgt_len = prev_input.shape[0]

        # Encoding
        embedded_article = self.embedding(article)
        embedded_article = embedded_article.view(src_len, bsz*n_agents, -1)
        # [src_len, bsz*n_agents, emb_dim]
        embedded_article = pack_padded_sequence(embedded_article,
                                                article_length,
                                                enforce_sorted=False)

        encoded_seq, state = self.multi_agt_encoder(embedded_article)

        # Decoding
        embedded_prev_input = self.embedding(prev_input)
        embedded_prev_input = pack_padded_sequence(embedded_prev_input,
                                                   prev_input_length,
                                                   enforce_sorted=False)

        decoder_out = self.decoder(prev_input=embedded_prev_input,
                                   encoded_seq=encoded_seq,
                                   init_state=state)
        vocab_probs, generation_probs, agentwise_attn, agent_attn = decoder_out

        generation_probs = generation_probs.unsqueeze(-1)

        voc_gen_probs = generation_probs * vocab_probs.unsqueeze(-2)
        # [bsz, tgt_len, n_agents, 1] * [bsz, tgt_len, 1, voc_sz]
        # -> [bsz, tgt_len, n_agents, voc_sz]

        copy_prob_weighted_attn = (1 - generation_probs) * agentwise_attn
        # [bsz, tgt_len, n_agents, 1]*[bsz*n_agents, tgt_len, src_len]
        # -> [bsz, tgt_len, n_agents, src_len]

        agt_extended_voc_probs = torch.cat((
            voc_gen_probs,
            torch.zeros(extended_voc_sz, device=self.device)))
        # [bsz, tgt_len, n_agents, extended_voc_sz]

        agt_extended_voc_probs.scatter_add_(
            dim=-1,
            index=article.unsqueeze(1).expand(-1, tgt_len, -1, -1),
            other=copy_prob_weighted_attn  # [bsz, tgt_len, n_agents, src_len]
        )
        # [bsz, tgt_len, n_agents, extended_voc_sz]

        extended_voc_probs = (agent_attn.view(bsz, tgt_len, n_agents, 1)
                              * agt_extended_voc_probs
                             ).sum(dim=-2)
        # ([bsz, tgt_len, n_agents, 1]*[bsz, tgt_len, n_agents, extended_voc_sz]).sum(-2)
        # -> [bsz, tgt_len, n_agents, extended_voc_sz].sum(dim=-2)
        # -> [bsz, tgt_len, extended_voc_sz]

        return extended_voc_probs
