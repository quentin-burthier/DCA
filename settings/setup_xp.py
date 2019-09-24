"""Routine loading every block needed to train a model."""

from typing import Tuple, Dict

import os
from os.path import join
import pickle
import yaml

import torch
from torch.optim import SGD, Adam

from data.token_indexer import TokenIndexer
from utils.vocab import make_vocab, load_embedding

# from trainer import Trainer

# from settings.model import build_multi_agt_summarizer

# from metrics.losses import sequence_nll
# from metrics.scores import compute_rouge_n


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


# def setup_xp(xp_name: str, hparams: dict, train_params: dict) -> Tuple[Trainer,
#                                                                        int,
#                                                                        torch.device,
#                                                                        bool]:
#     """Sets up the training experiment.

#     Args:
#         xp_name (str): [description]
#         hparams (dict): [description]
#         train_params (dict): [description]

#     Returns:
#         trainer (trainer.Trainer)
#         n_max_epochs (int)
#         device (torch.device)
#         pin_memory (bool)
#     """
#     optimizer_name = train_params["optimizer"]["name"]
#     optimizer_params = train_params["optimizer"]["params"]

#     n_max_epochs = train_params["n_max_epochs"]

#     device, pin_memory = set_device(train_params["use_gpu"])

#     model = build_multi_agt_summarizer(**hparams)
#     model.to(device)

#     optimizer = get_optimizer(model, optimizer_name, **optimizer_params)

#     criterion = sequence_nll

#     trainer = Trainer(xp_name=xp_name, model=model,
#                       optimizer=optimizer, criterion=criterion,
#                       validation_fn=compute_rouge_n)

#     return trainer, n_max_epochs, device, pin_memory


def set_device(use_gpu: bool) -> Tuple[torch.device, bool]:
    """Sets device and pin_memory parameters.

    Args:
        use_gpu (bool): [description]

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


# def get_optimizer(model, optimizer_name: str, optim_params: dict) -> torch.optim.Optimizer:
#     """Gets the optimizer.

#     Args:
#         model (nn.Module): PyTorch model
#         optimizer_name (str): Optimizer (SGD, Adam...)
#         optim_params (dict): Optimizer paramters. At least learning rate
#                              is required.

#     Returns:
#         torch.optim.Optimizer
#     """
#     optimizers = {"SGD": SGD, "Adam": Adam}
#     return optimizers[optimizer_name](model.parameters(), **optim_params)


def get_embedders(vocab_size: int, embedding_dim: int,
                  special_tokens: Dict[str, int]):

    with open(join(os.environ["CNNDM_PATH"], 'vocab_cnt.pkl'), 'rb') as f:
        word_count = pickle.load(f)

    word2id, id2word = make_vocab(word_count, vocab_size, special_tokens)

    w2v_path = join(os.environ["XP_PATH"],
                    "word2vec", f"word2vec.{embedding_dim}d.226k.bin")

    embedding, _ = load_embedding(id2word, word2id, w2v_path)

    return TokenIndexer(word2id, special_tokens["<unk>"]), embedding
