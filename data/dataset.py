""" CNN / Daily Mail dataset

From the implementation of
Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
(Yen-Chun Chen and Mohit Bansal, 2018)
https://github.com/ChenRocks/fast_abs_rl

Copyright (c) 2018 Yen-Chun Chen
"""

import json
import re
import os
from os.path import join

from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    """CNN - Daily Mail Dataset

    Args:
        split (str): [description]

    __getitem__:
        article
        abstract
    """

    def __init__(self, split: str, token_indexer) -> None:
        """[summary]

        Args:
            split (str): [description]

        Returns:
            None: [description]
        """
        assert split in ['train', 'val', 'test']
        self._data_path = join(os.environ["CNNDM_PATH"], split)
        self._n_data = count_data(self._data_path)
        self.token_indexer = token_indexer

    def __getitem__(self, index: int):
        with open(join(self._data_path, f"{index}.json")) as f:
            sample = json.loads(f.read())
        article: list = self.token_indexer(sample["article"])
        abstract: list = self.token_indexer(sample["abstract"])

        return article, abstract

    def __len__(self) -> int:
        return self._n_data


class DCADataset(CnnDmDataset):

    def __init__(self, split: str, n_agents: int, token_indexer):
        super().__init__(split, token_indexer)
        self.n_agents = n_agents

    def __getitem__(self, i: int):
        article, abstract = super().__getitem__(i)

        splitted_article = self.split_article(article)

        return article, abstract[1:], abstract

    def split_article(self, article: list):
        n = len(article)
        return [article[:n//3], article[n//3:2*n//3], article[2*n//3:]]


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data
