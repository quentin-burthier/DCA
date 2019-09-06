"""Teacher forcing training of the model."""

from torch.utils.data import DataLoader

from data.dataset import CnnDmDataset
from data.collating import collate_by_packing
from settings import load_params, setup_xp


if __name__ == '__main__':
    # https://pytorch.org/docs/stable/data.html#platform-specific-behaviors
    # 

    xp_name, hparams, train_params = load_params()

    trainer, n_max_epochs, device, pin_memory = setup_xp(
        xp_name, hparams, train_params)

    train_set = CnnDmDataset(split="train")
    val_set = CnnDmDataset(split="validation")

    loaders_params = {"batch_size": train_params["batch_size"],
                      "collate_fn": collate_by_packing,
                      "pin_memory": pin_memory}
    train_loader = DataLoader(train_set, **loaders_params, shuffle=True)
    val_loader = DataLoader(val_set, **loaders_params, shuffle=False)

    for epoch in range(n_max_epochs):
        trainer.train_mode()
        for articles, gold_summaries in train_loader:
            articles = articles.to(device, non_blocking=True)
            gold_summaries = gold_summaries.to(device, non_blocking=True)

            trainer.train_step(articles, gold_summaries)

        trainer.validation_mode()
        for articles, gold_summaries in val_loader:
            articles = articles.to(device, non_blocking=True)
            gold_summaries = gold_summaries.to(device, non_blocking=True)

            trainer.validation_step(articles, gold_summaries)

        trainer.save_checkpoint()
