#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MaskBCELoss(_Loss):
    """
    MaskBCELoss
    """
    def __init__(self, reduction='mean'):
        super(MaskBCELoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, input, target, mask=None):
        """
        input: (batch_size, max_len)
        target: (batch_size, max_len)
        mask: (batch_size, max_len)
        """
        bce = F.binary_cross_entropy(input=input,
                                     target=target,
                                     reduction='none')
        if mask is not None:
            bce *= mask.float()

        bce = bce.sum(dim=1)

        if self.reduction == 'mean':
            bce = bce.mean()
        elif self.reduction == 'sum':
            bce = bce.sum()
        return bce
