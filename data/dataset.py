""" PyTorch datasets.
"""
from typing import List

import json
import re
import os
from os.path import join
from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    """CNN - Daily Mail Dataset

    Args:
        split (str): train, val or test

    __getitem__:
        article (LongTensor)
        abstract (LongTensor)
    """

    def __init__(self, split: str, token_indexer) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(os.environ["CNNDM_PATH"], split)
        self._n_data = count_data(self._data_path)
        self.token_indexer = token_indexer

    def __getitem__(self, index: int):
        with open(join(self._data_path, f"{index}.json")) as f:
            sample = json.loads(f.read())
        article: List[str] = sample["article"]
        abstract: List[str] = sample["abstract"]

        article = torch.tensor(self.preprocess(article), dtype=torch.long)
        abstract = torch.tensor(self.preprocess(abstract), dtype=torch.long)

        return article, abstract

    def preprocess(self, text: List[str]) -> torch.LongTensor:
        text = chain(*(sentence.split(' ') for sentence in text))
        return self.token_indexer(text)

    def __len__(self) -> int:
        return self._n_data


class DCADataset(CnnDmDataset):

    def __init__(self, split: str, n_agents: int, token_indexer):
        super().__init__(split, token_indexer)
        self.n_agents = n_agents

    def __getitem__(self, i: int):
        article, abstract = super().__getitem__(i)

        splitted_article = self.split_article(article)

        splitted_article = pad_sequence(splitted_article)
        # [src_len, n_agents]

        return splitted_article, abstract[:-1], abstract

    def split_article(self, article: list) -> List[torch.LongTensor]:
        """Split the article in n_agents paragraphs.

        Current implementation simply returns paragraphs of equal lengths
        (+/- 1)
        """
        article_len = len(article)
        split_len = article_len / self.n_agents
        return [article[round(i*split_len): round((i+1)*split_len)]
                for i in range(self.n_agents)]



def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data
