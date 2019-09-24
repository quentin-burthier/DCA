""" pretrain a word2vec on the corpus

From the implementation of
Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
(Yen-Chun Chen and Mohit Bansal, 2018)
https://github.com/ChenRocks/fast_abs_rl

Copyright (c) 2018 Yen-Chun Chen
"""
import argparse
import json
import logging
import os
from os.path import join, exists
from time import time
from datetime import timedelta

from cytoolz import concatv
import gensim

from data.dataset import count_data



class Sentences(object):
    """ needed for gensim word2vec training"""
    def __init__(self):
        self._path = join(os.environ["CNNDM_PATH"], 'train')
        self._n_data = count_data(self._path)

    def __iter__(self):
        for i in range(self._n_data):
            with open(join(self._path, f'{i}.json')) as f:
                data = json.loads(f.read())
            for s in concatv(data['article'], data['abstract']):
                yield ['<s>'] + s.lower().split() + [r'<\s>']


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time()
    save_dir = os.environ["XP_PATH"]
    if not exists(save_dir):
        os.makedirs(save_dir)

    sentences = Sentences()
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=5, workers=16, sg=1)
    model.build_vocab(sentences)
    print(f'vocab built in {timedelta(seconds=time()-start)}')
    model.train(sentences,
                total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(save_dir,
                    f"word2vec.{args.dim}d.{len(model.wv.vocab)//1000}k.bin"))
    model.wv.save_word2vec_format(join(
        save_dir,
        f"word2vec.{args.dim}d.{len(model.wv.vocab)//1000}k.w2v'"
    ))

    print(f"word2vec trained in {timedelta(seconds=time()-start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    # parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=128)
    args = parser.parse_args()

    main(args)
