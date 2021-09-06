#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from dataclasses import dataclass

from seq2seq.utility.utilities import list2tensor
from seq2seq.model.base_model import BaseModel
from seq2seq.inputter.embedder import Embedder
from seq2seq.model.beam_search_utils import BeamSearchScorer
from seq2seq.model.generation_utils import GenerationMixin

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # These attributes are needed in `GenerationMixin.beam_search()`.
    is_encoder_decoder = True
    max_length = -1
    pad_token_id = -1
    eos_token_id = -1


class Seq2SeqTransformer(BaseModel, GenerationMixin):
    """
    Seq2SeqTransformer
    """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            embed_size: int,
            hidden_size: int,
            start_token: int = 1,
            end_token: int = 2,
            padding_idx: int = 0,
            batch_first: bool = True,
            num_layers: int = 1,
            attn_head_num=None,
            tie_embedding: bool = False,
            dropout: float = 0.0,
            teacher_forcing_ratio: float = 0.0,
            use_gpu: bool = False
    ):
        super(Seq2SeqTransformer, self).__init__()

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
        self.attn_head_num = attn_head_num
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio

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

        self.enc_embedder = Embedder(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.padding_idx
        )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.dec_embedder = self.enc_embedder
            logger.info("Tie embedding.")
        else:
            self.dec_embedder = Embedder(
                num_embeddings=self.tgt_vocab_size,
                embedding_dim=self.embed_size,
                padding_idx=self.padding_idx
            )

        self.pos_encoder = PositionalEncoding(
            self.embed_size,
            self.dropout,
            batch_first=self.batch_first
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.embed_size,
                nhead=self.attn_head_num,
                dim_feedforward=self.hidden_size,
                dropout=self.dropout,
                batch_first=self.batch_first
            ),
            self.num_layers
        )

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=self.embed_size,
                nhead=self.attn_head_num,
                dim_feedforward=self.hidden_size,
                dropout=self.dropout,
                batch_first=self.batch_first
            ),
            self.num_layers
        )

        self.output_layer = nn.Linear(self.embed_size, self.tgt_vocab_size)

        self.to(self.device)

        logger.debug(self)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def encode(self, inputs):
        """
        encode
        """
        enc_inputs, enc_input_lengths = inputs  # inputs["inputs"], inputs["lengths"]
        enc_input_pad_masks = (enc_inputs == self.padding_idx) if self.padding_idx is not None else None
        enc_inputs = self.pos_encoder(self.enc_embedder(enc_inputs))
        memory = self.encoder(enc_inputs, None, enc_input_pad_masks)
        return memory

    def forward(
            self,
            inputs,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            is_training: bool = True
    ):
        """
        forward
        """
        enc_inputs, dec_inputs = inputs["input"], inputs["target"]
        memory = self.encode(enc_inputs)
        dec_inputs, dec_input_lengths = dec_inputs
        # Target does not include SOS token. Need to add SOS.
        # And need to remove the last token to maintain the length.
        dec_inputs = torch.cat(
            (
                torch.ones(
                    (len(dec_inputs), 1),
                    dtype=torch.long,
                    device=dec_inputs.device
                ) * self.start_token,
                dec_inputs
            ),
            dim=-1
        )[:, :-1]

        dec_inputs = self.pos_encoder(self.dec_embedder(dec_inputs))

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(dec_inputs.size(1))

        dec_output = self.decoder(
            dec_inputs, memory, tgt_mask, None, tgt_key_padding_mask
        )
        dec_output = self.output_layer(dec_output)
        log_probs = functional.log_softmax(dec_output, dim=-1)
        return log_probs

    def infer(
            self,
            input,
            max_length: int = 20,
            mode: str = "greedy",
            beam_width: int = -1
    ):
        assert isinstance(mode, str)
        mode = mode.lower()
        assert mode in ("greedy", "beam")
        if mode == "greedy":
            return self.infer_greedy(src=input, max_length=max_length)
        elif mode == "beam":
            return self.infer_beam(input=input, max_length=max_length, beam_width=beam_width)

    def infer_greedy(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
            tgt=None,
            max_length: int = 20,
            batch_first: bool = True
    ):
        """
        infer_greedy
        """
        need_transpose = (batch_first != self.batch_first)
        self_batch_dim = 0 if self.batch_first else 1
        self_seq_dim = 1 - self_batch_dim

        self.eval()
        with torch.no_grad():
            if isinstance(src, (list, tuple)):
                src, lengths = list2tensor(src)
            if len(src.size()) == 1:
                src = src.unsqueeze(0)
                lengths = lengths.unsqueeze(0)
            src = src.to(self.device)

            # shape: (batch_size, seq_len)
            tgt = torch.ones(
                size=[src.size(self_batch_dim), 1],
                dtype=torch.long,
                device=self.device
            ) * self.start_token

            if need_transpose:
                src = src.transpose(0, 1)
                src_mask = src_mask.transpose(0, 1)
                src_key_padding_mask = src_key_padding_mask.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

            src_emb = self.pos_encoder(self.enc_embedder(src))
            memory = self.encoder(src_emb, src_mask, src_key_padding_mask).detach()

            for i in range(max_length):
                tgt_input = self.pos_encoder(self.dec_embedder(tgt))
                output = self.decoder(tgt_input, memory, None, None, None)
                output = self.output_layer(output)
                log_probs = functional.log_softmax(output, dim=-1)
                o_score, max_index = torch.max(log_probs, dim=2)
                last_index = max_index[-1] if not self.batch_first else max_index[:, -1]
                tgt = torch.cat((tgt, last_index.unsqueeze(self_seq_dim)), dim=self_seq_dim).detach()
                end_flag = (
                    ((tgt == self.end_token).int().sum(dim=self_seq_dim) > 0).int().sum()
                    ==
                    tgt.size(self_batch_dim)
                )
                if end_flag:
                    break

            if need_transpose:
                tgt = tgt.transpose(0, 1)
            tgt = tgt[:, 1:] if batch_first else tgt[1:, :]  # Remove start token.
            tgt = tgt.squeeze(0).cpu().numpy().tolist()
            return tgt

    def infer_beam(
            self,
            input,
            max_length=20,
            beam_width=5
    ):
        """
        infer_beam
        """
        raise NotImplementedError


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#
# [Reference](https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py)
#

class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000,
            batch_first: bool = False
    ):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if self.batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
                or shape [batch_size, seq_len, embedding_dim] if batch_first=True
        """
        x += self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0)]
        return self.dropout(x)
