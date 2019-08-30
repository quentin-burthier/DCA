"""Multi-agents summarizer"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pad_packed_sequence,
    pack_padded_sequence
)


class MultiAgentsSummarizer(nn.Module):
    """Multi-Agents Summarizer.

    Args:
        embedding_layer (nn.Embedding):
        multi_agt_encoder (MultiAgentsEncoder):
        decoder (Decoder):
    """

    def __init__(self, embedding_layer: nn.Embedding, multi_agt_encoder, decoder):
        super().__init__()
        self.embedding = embedding_layer
        self.multi_agt_encoder = multi_agt_encoder
        self.decoder = decoder

    def forward(self, article, seq_lengths, prev_input):

        _, bsz, n_agents, _ = article.shape
        tgt_len = prev_input.shape[0]

        embedded_article = self.embedding(article)

        encoded_seq, state = self.multi_agt_encoder(embedded_article, seq_lengths)

        vocab_probs, generation_probs, agentwise_attn, agent_attn = self.decoder(
            prev_input=prev_input,
            encoded_seq=encoded_seq,
            init_state=state
        )
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


class MultiAgentsEncoder(nn.Module):
    """Multi-agents encoder.

    Args:
        n_agents (int):
        embedding_layer (nn.Embedding):
        local_layer (nn.Module): expected to do the concatenation of all directions
        contextual_layer (nn.ModuleList):
        msg_projection (nn.Module):
    """

    def __init__(self, n_agents: int,
                 local_layer: nn.Module, contextual_layers: nn.ModuleList,
                 bidir_hs_proj: nn.Module, msg_projection: nn.Module):

        super().__init__()
        self.n_agents = n_agents

        # All agents weights are shared.
        self.local_layer = local_layer
        self.contextual_layers = contextual_layers
        self.bidir_hs_proj = bidir_hs_proj
        self.msg_projection = msg_projection

        self.mask_matrix = torch.ones(self.n_agents, self.n_agents, dtype=torch.uint8)
        self.mask_matrix -= torch.eye(self.n_agents, dtype=torch.uint8)

    def forward(self, article: PackedSequence, seq_lenghts):

        # Local encoding of each article
        prev_layer_enc, prev_hs = self.local_layer(article, seq_lenghts)
        # [src_len, bsz*n_agents, 2*hsz]
        # [bsz*n_agents, 2*hsz]
        prev_layer_enc = self.bidir_hs_proj(prev_layer_enc) # [src_len, bsz*n_agents, hsz]
        prev_hs = self.bidir_hs_proj(prev_hs) # [bsz*n_agents, hsz]

        bsz = prev_hs.shape[0] // self.n_agents

        # Contextual encoding
        for contextual_layer in self.contextual_layers:

            hsz = prev_hs.shape[1]
            prev_hs = prev_hs.view(bsz, self.n_agents, hsz)

            # Get the messages
            message = [prev_hs[:, self.mask_matrix[agt_id], :].mean(dim=1)
                       for agt_id in range(self.n_agents)]  # n_agents*[bsz, hsz]
            message = torch.stack([
                torch.stack([message[agt_id][batch]  # hsz
                             for agt_id in range(self.n_agents)])  # [n_agents, hsz]
                for batch in range(bsz)
            ])  # [bsz, n_agents, hsz]
            message = message.view(bsz*self.n_agents, hsz)

            # Project the messages
            next_layer_in = self.msg_projection(prev_layer_enc, message)
            # [src_len, bsz*n_agents, hsz]

            prev_layer_enc, prev_hs = contextual_layer(next_layer_in, seq_lenghts)

        last_h_of_1st_agt = prev_hs.view(bsz, self.n_agents, hsz)[:, 0, :]  # [bsz, hsz]
        prev_layer_enc = pad_packed_sequence(prev_layer_enc)  # [bsz*n_agents, hsz]
        return prev_layer_enc, last_h_of_1st_agt


class Decoder(nn.Module):
    """Multi-agents decoder.

        Args:
            n_agents (int):
            decoder (nn.Module):
            mlp (nn.Module):
            word_attention ([type]):
            agent_attention ([type]):

        Input:
            prev_input (Tensor[tgt_len, bsz, hsz]): previously decoded words.
                                                    tgt_len = 1 in recursive
                                                    prediction mode
            encoded_seq (Tensor[src_len, bsz, n_agents, hsz]):
            init_state Tuple(Tensor[bsz, hsz], Tensor[bsz, hsz]):
    """

    def __init__(self, decoder_layer: nn.Module, mlp: nn.Module,
                 generator: nn.Module, word_attention, agent_attention):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.word_attn = word_attention
        self.agent_attn = agent_attention
        self.mlp = mlp
        self.generator = generator

    def forward(self, prev_input, encoded_seq, init_state):

        src_len, bsz, n_agents, hsz = encoded_seq.shape
        tgt_len = prev_input.shape[0]

        encoded_seq = encoded_seq.view(src_len, bsz*n_agents, hsz)
        encoded_seq = encoded_seq.transpose(0, 1)  # [bsz*n_agents, src_len, hsz]

        state = self.decoder_layer(prev_input, init_state)  # [tgt_len, bsz, hsz]
        state = state.transpose(0, 1)  # [bsz, tgt_len, hsz]

        ## Agent context
        agentwise_attn = self.word_attn(
            state.unsqueeze(1) \
                 .expand(-1, n_agents, -1, -1) \
                 .view(bsz*n_agents, tgt_len, hsz),
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
                                             .view(bsz*tgt_len, n_agents, hsz)

        ## Global context
        agent_attn = self.agent_attn(state.view(bsz*tgt_len, 1, hsz),
                                     agentwise_context)
        # [bsz*tgt_len, 1, n_agents]
        agent_attn = agent_attn.transpose(1, 2)  # [bsz*tgt_len, n_agents, 1]

        global_context = (agent_attn * agentwise_context).sum(dim=1)
        # ([bsz*tgt_len, n_agents, 1]*[bsz*tgt_len, n_agents, hsz]).sum(dim=1)
        # -> [bsz*tgt_len, n_agents, hsz].sum(dim=1)
        # -> [bsz*tgt_len, hsz]

        vocab_probs = self.mlp(global_context, state)  # [bsz, tgt_len, voc_sz]

        # Pointer-Generator
        generation_probs = self.generator(agentwise_context, state, vocab_probs)
        # [bsz, tgt_len, n_agents]

        return vocab_probs, generation_probs, agentwise_attn, agent_attn
