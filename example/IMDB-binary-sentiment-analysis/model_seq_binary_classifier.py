#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from seq2seq.model.base_model import BaseModel
from seq2seq.inputter.embedder import Embedder

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SeqBinaryClassifier(BaseModel):
    """
    SeqBinaryClassifier
    """

    def __init__(self,
                 src_vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 pretrained_embedding: list = None,
                 batch_first: bool = True,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 with_bridge: bool = False,
                 dropout: float = 0.0,
                 embedding_dropout: float = 0.0,
                 rnn_cell: str = 'gru',
                 use_gpu: bool = False):
        super(SeqBinaryClassifier, self).__init__()

        if not batch_first:
            raise Exception('Sorry, this module supports batch first mode only.')

        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pretrained_embedding = pretrained_embedding
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.with_bridge = with_bridge
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.rnn_cell = rnn_cell

        self.use_gpu = use_gpu
        if self.use_gpu and (not torch.cuda.is_available()):
            logger.error("Passing argument use_gpu=True but torch.cuda.is_available()==False")
            logger.error("Switch use_gpu=False")
            self.use_gpu = False
        if self.use_gpu:
            logger.info("Using GPU")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")
        self.to(self.device)

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size
                                )
        if self.pretrained_embedding is not None:
            enc_embedder.load_embeddings(self.pretrained_embedding)

        self.encoder = SimpleRNN(input_size=self.embed_size,
                                 hidden_size=self.hidden_size,
                                 embedder=enc_embedder,
                                 num_layers=self.num_layers,
                                 bidirectional=self.bidirectional,
                                 dropout=self.dropout,
                                 embedding_dropout=self.embedding_dropout,
                                 batch_first=self.batch_first,
                                 rnn_cell=self.rnn_cell,
                                 use_gpu=self.use_gpu)

        self.output_logit_layer = nn.Linear(self.hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
        self.to(self.device)
        logger.debug(self)

    def encode(self, inputs, hidden=None):
        """
        encode
        """
        enc_inputs, enc_input_lengths = inputs  # inputs["inputs"], inputs["lengths"]
        enc_output_dict = self.encoder(enc_inputs, enc_input_lengths, hidden)
        enc_outputs, enc_hidden = enc_output_dict["outputs"], enc_output_dict["last_hidden_state"]
        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)
        return enc_outputs, enc_hidden

    def forward(self,
                inputs,
                hidden=None,
                is_training: bool = True):
        """
        forward
        return: log_probs
        """
        enc_inputs, targets = inputs["input"], inputs["target"]
        enc_outputs, enc_hidden = self.encode(enc_inputs, hidden)
        # print(enc_hidden.size()) # TODO: for debugging
        if isinstance(enc_hidden, tuple):
            # LSTM hidden: (hidden, cell_state)
            enc_hidden = enc_hidden[0]
        logits = self.output_logit_layer(enc_hidden[-1])
        log_probs = self.log_softmax(logits)
        return log_probs

    def infer(self, input):
        log_probs = self.forward(input)
        _, preds = log_probs.topk(1)
        preds = preds.squeeze(-1)
        return preds
