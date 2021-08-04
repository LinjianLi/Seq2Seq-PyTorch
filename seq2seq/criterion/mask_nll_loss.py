#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import torch
from torch.nn.modules.loss import _Loss


class MaskNLLLoss(_Loss):

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, inp, target, mask=None, reduction=True):
        """
        inp: (batch_size, seq_length, out_size)
        target: (batch_size, seq_length)
        mask: (batch_size, seq_length)
        """
        n_total = mask.sum().item() if mask is not None else None
        try:
            probs = torch.gather(input=inp, dim=-1, index=target.unsqueeze(-1))
            probs = probs.squeeze(-1)
        except:
            print(inp.shape, target.shape)
            print(inp, target)
            raise
        cross_entropy = -torch.log(probs)
        if mask is not None:
            try:
                cross_entropy = cross_entropy.masked_select(mask)
            except:
                print(cross_entropy.shape, mask.shape)
                raise
        if reduction:
            if self.reduction == 'mean':
                loss = cross_entropy.mean()
            elif self.reduction == 'sum':
                loss = cross_entropy.sum()
        return loss, n_total
