"""PyTorch Lightning test."""

import os
from os.path import join

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from test_tube import Experiment

from data.dataset import DCADataset
from data.collating import collate_by_padding
from metrics.losses import sequence_nll
# from metrics.scores import compute_rouge_n

from settings import load_params, build_multi_agt_summarizer, get_embedders

class SummarizerModule(pl.LightningModule):
    """PyTorch Lightning system.
    """

    def __init__(self, summarizer, token_indexer,
                 loaders_params: dict, optimizer_params: dict):
        super().__init__()
        self.summarizer = summarizer
        self._optimizer_params = optimizer_params
        self._loaders_params = loaders_params
        self._token_indexer = token_indexer

    def forward(self, article, article_length,
                prev_input, prev_input_length):
        return self.summarizer(article, article_length,
                               prev_input, prev_input_length)

    def training_step(self, batch, batch_nb):
        article_data, prev_input_data, gold_summary = batch
        article, article_length = article_data
        prev_input, prev_input_length = prev_input_data

        infered_summary = self.forward(article, article_length,
                                       prev_input, prev_input_length)
        return {'loss': sequence_nll(infered_summary, gold_summary)}

    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb)

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.summarizer.parameters(),
                                **self._optimizer_params)

    @pl.data_loader
    def tng_dataloader(self):
        train_set = DCADataset(split="train", n_agents=3,
                               token_indexer=self._token_indexer)
        return DataLoader(train_set, **self._loaders_params, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        val_set = DCADataset(split="val", n_agents=3,
                             token_indexer=self._token_indexer)
        return DataLoader(val_set, **self._loaders_params, shuffle=False)


def teacher_forcing_training():
    """Builds everything needed for the training, and launch the training."""
    xp_info, xp_params = load_params()
    xp_name, debug = xp_info
    xp_path = join(os.environ["XP_PATH"], xp_name)
    hparams, optimizer_params, training_params = xp_params

    trainer_params = training_params["trainer"]
    # device, pin_memory = set_device["trainer"]

    loaders_params = training_params["loaders"]
    loaders_params["collate_fn"] = collate_by_padding
    loaders_params["pin_memory"] = trainer_params["gpus"] is not None

    special_tokens = {
        "<pad>": 0,
        "<unk>": 1,
        "<start>": 2,
        "<end>": 3
    }
    token_indexer, embeddings = get_embedders(
        vocab_size=hparams["vocab_size"],
        embedding_dim=hparams["embedding_dim"],
        special_tokens=special_tokens
    )
    dca_summarizer = build_multi_agt_summarizer(**hparams,
                                                embeddings=embeddings)

    summarizer_module = SummarizerModule(
        summarizer=dca_summarizer,
        token_indexer=token_indexer,
        loaders_params=loaders_params,
        optimizer_params=optimizer_params
    )

    trainer_params["experiment"] = Experiment(save_dir=xp_path, name=xp_name,
                                              debug=debug)

    trainer_params["checkpoint_callback"] = ModelCheckpoint(
        filepath=join(xp_path, "checkpoint"),
        save_best_only=True,
        verbose=True,
        monitor='avg_val_loss',
        mode='min'
    )

    trainer = Trainer(**trainer_params)
    print(f"tensorboard --logdir {xp_path}", end="\n\n")
    trainer.fit(summarizer_module)


if __name__ == "__main__":
    teacher_forcing_training()
