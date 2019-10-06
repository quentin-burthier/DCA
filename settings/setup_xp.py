"""Routine loading every block needed to train a model."""

from typing import Tuple, Dict

import os
from os.path import join
import pickle
import yaml

import torch

from data.token_indexer import TokenIndexer
from utils.vocab import make_vocab, load_embedding


def load_params():
    """Loads and reads the yaml config file.
    FIXME: more general params path"""
    with open("/home/quentinb/DCA/training_params.yml", 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    xp_name = params["xp_name"]
    debug = params["debug"]
    hparams = params["hparams"]
    optim_params = params["optimizer"]
    training_params = params["training"]

    return (xp_name, debug), (hparams, optim_params, training_params)


def set_device(use_gpu: bool) -> Tuple[torch.device, bool]:
    """Sets device and pin_memory parameters.

    Args:
        use_gpu (bool)

    Returns:
        device (torch.device): cpu or cuda
        pin_memory (bool): whether using pin_memory.
                           True if the model is trained on GPU.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False
    return device, pin_memory


def get_embedders(vocab_size: int, embedding_dim: int,
                  special_tokens: Dict[str, int]):

    with open(join(os.environ["CNNDM_PATH"], 'vocab_cnt.pkl'), 'rb') as f:
        word_count = pickle.load(f)

    word2id, id2word = make_vocab(word_count, vocab_size, special_tokens)

    w2v_path = join(os.environ["XP_PATH"],
                    "word2vec", f"word2vec.{embedding_dim}d.226k.bin")

    embedding, _ = load_embedding(id2word, word2id, w2v_path)

    return TokenIndexer(word2id, special_tokens["<unk>"]), embedding
