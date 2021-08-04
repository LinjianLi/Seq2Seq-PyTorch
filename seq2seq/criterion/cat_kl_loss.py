#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch
from torch.nn.modules.loss import _Loss


class CatKLLoss(_Loss):
    """
    CatKLLoss
    """
    def __init__(self, reduction='none'):
        super(CatKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, log_qy, log_py):
        """
        KL(qy|py) = Eq[qy * log(q(y) / p(y))]

        log_qy: (batch_size, latent_size)
        log_py: (batch_size, latent_size)
        """
        qy = torch.exp(log_qy)
        kl = torch.sum(qy * (log_qy - log_py), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl
