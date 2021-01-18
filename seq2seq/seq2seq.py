#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import torch
import torch.nn as nn

from .criterions import NLLLoss
from .base_model import BaseModel
from .embedder import Embedder
from .simple_rnn import SimpleRNN
from .decoder_rnn import DecoderRNN

import logging

logger = logging.getLogger(__name__)

class Seq2Seq(BaseModel):
    """
    Seq2Seq
    """
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_size,
                 hidden_size,
                 start_token=1,
                 end_token=2,
                 padding_idx=None,
                 batch_first=True,
                 num_layers=1,
                 bidirectional=True,
                 attn_mode=None,
                 attn_hidden_size=None,
                 with_bridge=False,
                 tie_embedding=False,
                 dropout=0.0,
                 rnn_cell='gru',
                 teacher_forcing_ratio=0,
                 use_gpu=False):
        super(Seq2Seq, self).__init__()

        if not batch_first:
            raise Exception('Sorry, this module supports batch first mode only.')

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.start_token = start_token
        self.end_token = end_token
        self.padding_idx = padding_idx
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.rnn_cell = rnn_cell
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_gpu = use_gpu

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)


        self.encoder = SimpleRNN(input_size=self.embed_size,
                                 hidden_size=self.hidden_size,
                                 embedder=enc_embedder,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 dropout=self.dropout,
                                 batch_first=self.batch_first,
                                 rnn_cell=self.rnn_cell,
                                 use_gpu=self.use_gpu)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)

        self.decoder = DecoderRNN(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size,
                                  embedder=dec_embedder,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  attn_hidden_size=self.attn_hidden_size,
                                  rnn_cell=self.rnn_cell,
                                  dropout=self.dropout,
                                  batch_first=self.batch_first,
                                  use_gpu=self.use_gpu)

        # TODO: The loss definition block maybe no longer needed. Delete it if so.
        # Loss Definition
        if self.padding_idx is not None:
            weight = torch.ones(self.tgt_vocab_size)
            weight[self.padding_idx] = 0
        else:
            weight = None
        self.nll_loss = NLLLoss(weight=weight,
                                ignore_index=self.padding_idx,
                                reduction='mean')

        if self.use_gpu:
            logger.info("Using GPU")
            self.cuda()
        else:
            logger.info("Using CPU")

        logger.debug(self)

    def encode(self, inputs, hidden=None):
        """
        encode
        """
        enc_inputs, enc_input_lengths = inputs #inputs["inputs"], inputs["lengths"]
        enc_outputs, enc_hidden = self.encoder(enc_inputs, enc_input_lengths, hidden)
        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)
        return enc_outputs, enc_hidden

    def forward(self, inputs, hidden=None):
        """
        forward
        """
        enc_inputs, dec_inputs = inputs["input"], inputs["target"]
        enc_outputs, enc_hidden = self.encode(enc_inputs, hidden)
        dec_inputs, dec_input_lengths = dec_inputs
        # Target does not include SOS token. Need to add SOS.
        # And need to remove the last token to maintain the length.
        dec_inputs = torch.cat(
            (torch.ones(
                (len(dec_inputs), 1),
                dtype=torch.long,
                device=dec_inputs.device
            ) * self.start_token,
            dec_inputs),
            dim=-1
        )[:, :-1]
        decoder_output_tokens, decoder_outputs, hidden\
            = self.decoder(inputs=dec_inputs,
                           hidden=enc_hidden,
                           lengths=dec_input_lengths,
                           encoder_outputs=enc_outputs,
                           teacher_forcing_ratio=self.teacher_forcing_ratio)
        return decoder_outputs

    def infer(self, input, max_length=20):
        """
        infer
        input: 1-D list of integers representing tokens.
        return: 1-D list of integers representing tokens
        """

        assert isinstance(input, (list, tuple))
        assert isinstance(input[0], int)

        self.eval()
        with torch.no_grad():
            enc_inputs = torch.tensor(input, dtype=torch.long) # shape: (seq_len)
            enc_inputs = enc_inputs.unsqueeze(0) # shape: (1, seq_len)
            if self.use_gpu:
                enc_inputs = enc_inputs.cuda()
            enc_inputs = (enc_inputs, None)

            # shape: (batch_size, seq_len)=(1, 1)
            dec_inputs = torch.tensor([self.start_token], dtype=torch.long).unsqueeze(0)
            if self.use_gpu:
                dec_inputs = dec_inputs.cuda()

            enc_outputs, enc_hidden = self.encode(inputs=enc_inputs)
            decoder_output_tokens, decoder_outputs, hidden\
                = self.decoder(inputs=dec_inputs,
                            hidden=enc_hidden,
                            lengths=None,
                            encoder_outputs=enc_outputs,
                            max_length=max_length,
                            teacher_forcing_ratio=0)
            decoder_output_tokens = decoder_output_tokens.squeeze(0).tolist() # shape: (seq_len)

            # Discard the content after the first end_token.
            for i in range(len(decoder_output_tokens)):
                if decoder_output_tokens[i] == self.end_token:
                    decoder_output_tokens = decoder_output_tokens[:i+1]
                    break
            return decoder_output_tokens
