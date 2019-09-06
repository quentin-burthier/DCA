
import os
from os.path import join

from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, xp_path: str):
        self.writer = SummaryWriter(join(xp_path, "log"))

    def __call__(self, loss, step: int, mode="train"):
        if mode == "train":
            self.log_train_step(loss, step)
        else:
            self.log_val_step(loss, step, mode)

    def log_train_step(self, loss, step: int):
        self.writer.add_scalar("loss/train", loss, step)

    def log_val_step(self, loss, step: int, mode: str):
        self.writer.add_scalar(f"{mode}", loss, step)
