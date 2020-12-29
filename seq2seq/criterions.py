#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

# import torch
import torch.nn.functional as F
# from torch import distributions
from torch.nn.modules.loss import _Loss

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

# class NormalKLLoss(_Loss):
#     """
#     NormalKLLoss
#     """
#     def __init__(self, reduction='mean'):
#         super(NormalKLLoss, self).__init__()
#         assert reduction in ['none', 'sum', 'mean']
#         self.reduction = reduction

#     def forward(self, q_mu, q_logvar, p_mu=None, p_logvar=None):
#         """
#         q_mu: (batch_size, latent_size)
#         q_logvar: (batch_size, latent_size)
#         """
#         if p_mu is None:
#             p_mu = torch.zeros_like(q_mu)
#         if p_logvar is None:
#             p_logvar = torch.zeros_like(q_logvar)

#         q_norm = distributions.Normal(q_mu, q_logvar.exp().sqrt())
#         p_norm = distributions.Normal(p_mu, p_logvar.exp().sqrt())
#         kl = distributions.kl_divergence(q_norm, p_norm).sum(dim=1)

#         if self.reduction == 'mean':
#             kl = kl.mean()
#         elif self.reduction == 'sum':
#             kl = kl.sum()
#         return kl


# class CatKLLoss(_Loss):
#     """
#     CatKLLoss
#     """
#     def __init__(self, reduction='none'):
#         super(CatKLLoss, self).__init__()
#         assert reduction in ['none', 'sum', 'mean']
#         self.reduction = reduction

#     def forward(self, log_qy, log_py):
#         """
#         KL(qy|py) = Eq[qy * log(q(y) / p(y))]

#         log_qy: (batch_size, latent_size)
#         log_py: (batch_size, latent_size)
#         """
#         qy = torch.exp(log_qy)
#         kl = torch.sum(qy * (log_qy - log_py), dim=1)

#         if self.reduction == 'mean':
#             kl = kl.mean()
#         elif self.reduction == 'sum':
#             kl = kl.sum()
#         return kl


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


# class MaskBCELoss(_Loss):
#     """
#     MaskBCELoss
#     """
#     def __init__(self, reduction='mean'):
#         super(MaskBCELoss, self).__init__()
#         assert reduction in ['none', 'sum', 'mean']
#         self.reduction = reduction

#     def forward(self, input, target, mask=None):
#         """
#         input: (batch_size, max_len)
#         target: (batch_size, max_len)
#         mask: (batch_size, max_len)
#         """
#         bce = F.binary_cross_entropy(input=input,
#                                      target=target,
#                                      reduction='none')
#         if mask is not None:
#             bce *= mask.float()

#         bce = bce.sum(dim=1)

#         if self.reduction == 'mean':
#             bce = bce.mean()
#         elif self.reduction == 'sum':
#             bce = bce.sum()
#         return bce


# class RedundancyLoss(_Loss):
#     """
#     RedundancyLoss
#     """
#     def __init__(self):
#         super(RedundancyLoss, self).__init__()

#     def forward(self, A):
#         """
#         forward
#         """
#         I = torch.eye(A.size(1))
#         if A.is_cuda:
#             I = I.cuda()
#         norm = torch.bmm(A, A.transpose(1, 2)) - I
#         norm = torch.sum(
#             torch.sum(norm.pow(2), dim=2), dim=1)  # ** 0.5
#         loss = norm.mean()
#         return loss
