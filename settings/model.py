"""Functions to build the model."""

import torch
import torch.nn as nn

from model.multi_agents import MultiAgentsSummarizer

from model.multi_agents_encoder import MultiAgentsEncoder, MessageProjector
from model.encoder_layers import BiLSTMLayer

from model.decoder import Decoder, Generator, VocabPredictor
from model.decoder_layers import LSTMLayer

from model.attention import AdditiveAttention


def build_multi_agt_summarizer(n_agents: int, embedding_dim: int, vocab_size: int,
                               hidden_size: int, n_contextual_layers: int,
                               embeddings: torch.Tensor):
    """Builds the Multi-Agents Summarizer."""

    embedding_layer = nn.Embedding.from_pretrained(embeddings)

    multi_agt_encoder = build_multi_agt_encoder(
        n_agents=n_agents, embedding_dim=embedding_dim,
        hidden_size=hidden_size, n_contextual_layers=n_contextual_layers)

    decoder = build_decoder(hidden_size, embedding_dim, vocab_size)

    return MultiAgentsSummarizer(embedding_layer, multi_agt_encoder, decoder)


def build_multi_agt_encoder(n_agents: int, embedding_dim: int, hidden_size: int,
                            n_contextual_layers: int):
    """Builds the multi-agents encoder."""
    local_layer = BiLSTMLayer(input_size=embedding_dim, hidden_size=hidden_size)
    contextual_layers = nn.ModuleList(
        [BiLSTMLayer(input_size=hidden_size, hidden_size=hidden_size)
         for _ in range(n_contextual_layers)]
    )

    bidir_hs_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
    msg_projector = MessageProjector(hidden_size)

    multi_agt_encoder = MultiAgentsEncoder(
        n_agents=n_agents, local_layer=local_layer,
        contextual_layers=contextual_layers,
        bidir_hs_proj=bidir_hs_proj, msg_projector=msg_projector
    )

    return multi_agt_encoder


def build_decoder(hidden_size: int, embedding_dim: int, vocab_size: int):
    decoder_layer = LSTMLayer(input_size=hidden_size, hidden_size=hidden_size)

    generator = Generator(hidden_size, embedding_dim)

    vocab_predictor = VocabPredictor(hidden_size, vocab_size)

    word_attention = AdditiveAttention(hidden_size, bias=True)
    agent_attention = AdditiveAttention(hidden_size, bias=True)

    return Decoder(decoder_layer=decoder_layer, vocab_predictor=vocab_predictor,
                   generator=generator, word_attention=word_attention,
                   agent_attention=agent_attention)
