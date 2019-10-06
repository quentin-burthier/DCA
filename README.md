# Deep Communicating Agents for Abstractive Summarization

PyTorch implementation of
[Deep Communicating Agents for Abstractive Summarization](https://www.aclweb.org/anthology/N18-1150)
(Celikyilmaz et al., NAACL 2018)

_Ongoing work_

## Setup

The following environment variables should be defined:

- XP_PATH: directory where the experiments checkpoints and results are saved
- CNNDM_PATH: path of the CNN / Daily Mail dataset, preprocessed as described [here](https://github.com/ChenRocks/cnn-dailymail)

```
$CNNDM_PATH/
│   vocab_cnt.pkl
│
└───train
│      1.json
│      2.json
|      ...
|
└───val
|
└───test

```
