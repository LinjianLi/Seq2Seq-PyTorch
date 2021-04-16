#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import logging
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


logger = logging.getLogger(__name__)


class NLLLoss(_Loss):
    """
    NLLLoss
    The torch.nn.NLLLoss need input to be of shape (N,C) where C = number of classes,
    or (N, C, d_1, d_2, ..., d_K)with K≥1 in the case of K-dimensional loss.
    We want an NLLLoss takes input of shape (batch_size, max_len, vocab_size).
    """
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(NLLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, reduction=True):
        """
        input: (batch_size, max_len, vocab_size)
        target: (batch_size, max_len)

        return: The mean loss of elements in target if applying mean reduction.
        """
        # batch_size = input.size(0)

        # The torch.nn.NLLLoss need input to be of shape (N,C) where C = number of classes,
        # or (N, C, d_1, d_2, ..., d_K)with K≥1 in the case of K-dimensional loss.
        # We need to reshape the input and the target therefore.
        logger.debug(str(input.topk(1)))
        nll = F.nll_loss(input=input.view(-1, input.size(-1)),
                         target=target.contiguous().view(-1),
                         weight=self.weight,
                         ignore_index=self.ignore_index,
                         reduction='none')

        # The original implementation of Baidu will group the loss for each batch
        # instead of each element. The disadvantage of this is that the loss will
        # have different scales for different number of elements in a batch.

        # nll = nll.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                nll = nll.mean()
            elif self.reduction == 'sum':
                nll = nll.sum()

        return nll
