#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch
from torch.nn.modules.loss import _Loss


class RedundancyLoss(_Loss):
    """
    RedundancyLoss
    """
    def __init__(self):
        super(RedundancyLoss, self).__init__()

    def forward(self, A):
        """
        forward
        """
        I = torch.eye(A.size(1))
        if A.is_cuda:
            I = I.cuda()
        norm = torch.bmm(A, A.transpose(1, 2)) - I
        norm = torch.sum(
            torch.sum(norm.pow(2), dim=2), dim=1)  # ** 0.5
        loss = norm.mean()
        return loss
