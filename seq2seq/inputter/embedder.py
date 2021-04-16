#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/embedder.py
"""

import logging
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Embedder(nn.Embedding):
    """
    Embedder
    """
    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            # If no pretrained embedding for this token, randomly generate one.
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        logger.info("{} words have pretrained embeddings"
                    " (coverage: {:.3f})".format(
                        num_known, num_known / self.num_embeddings))
