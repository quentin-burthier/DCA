"""Differentiable loss functions."""

import torch
import torch.nn.functional as F

def sequence_nll(predicted_probs, gold_summaries, padding_idx: int = 0):
    """Negative Log-Likelihood

    Args:
        predicted_probs (Tensor[bsz, tgt_len, voc_hsz])
        gold_summaries ([LongTensor[bsz, tgt_len])
        padding_idx ([type])

    Returns:
        [type]
    """
    log_probs = torch.log(predicted_probs) #  [bsz, tgt_len, voc_hsz]

    return F.nll_loss(log_probs.transpose(1, 2), gold_summaries,
                      reduction='mean', ignore_index=padding_idx)
