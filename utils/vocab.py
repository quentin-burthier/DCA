from typing import Dict

from os.path import basename

import torch
import torch.nn as nn
import gensim

def make_vocab(word_count, vocab_size, special_tokens: Dict[str, int]):
    word2id = special_tokens.copy()
    id2word = {idx: token for token, idx in special_tokens.items()}

    for i, (token, _) in enumerate(word_count.most_common(vocab_size),
                                   len(special_tokens)):
        word2id[token] = i
        id2word[i] = token
    return word2id, id2word


def load_embedding(id2word, word2id, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == word2id['<start>']:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == word2id['<end>']:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)

    return embedding, oovs
