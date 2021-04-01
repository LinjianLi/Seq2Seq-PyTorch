#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import torch

def maskNLLLoss(inp, target, mask=None):
    """
    inp: (batch_size, seq_length, out_size)
    target: (batch_size, seq_length)
    mask: (batch_size, seq_length)
    """
    nTotal = mask.sum().item() if mask is not None else None
    try:
        probs = torch.gather(input=inp, dim=-1, index=target.unsqueeze(-1))
        probs = probs.squeeze(-1)
    except:
        print(inp.shape, target.shape)
        print(inp, target)
        raise
    crossEntropy = -torch.log(probs)
    if mask is not None:
        try:
            crossEntropy = crossEntropy.masked_select(mask)
        except:
            print(crossEntropy.shape, mask.shape)
            raise
    loss = crossEntropy.mean()
    return loss, nTotal
