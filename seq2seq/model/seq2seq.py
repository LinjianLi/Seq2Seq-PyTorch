#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# Modified by Linjian Li

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass

from seq2seq.model.base_model import BaseModel
from seq2seq.inputter.embedder import Embedder
from seq2seq.module.simple_rnn import SimpleRNN
from seq2seq.module.decoder_rnn import DecoderRNN
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


class Seq2Seq(BaseModel, GenerationMixin):
    """
    Seq2Seq
    """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            embed_size: int,
            hidden_size: int,
            start_token: int = 1,
            end_token: int = 2,
            padding_idx=None,
            batch_first: bool = True,
            num_layers: int = 1,
            bidirectional: bool = False,
            attn_mode=None,
            attn_hidden_size=None,
            with_bridge: bool = False,
            tie_embedding: bool = False,
            dropout: float = 0.0,
            rnn_cell: str = 'gru',
            teacher_forcing_ratio: float = 0.0,
            use_gpu: bool = False
    ):
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

        enc_embedder = Embedder(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=self.padding_idx
        )

        self.encoder = SimpleRNN(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            embedder=enc_embedder,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=self.batch_first,
            rnn_cell=self.rnn_cell,
            use_gpu=self.use_gpu
        )

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
        else:
            dec_embedder = Embedder(
                num_embeddings=self.tgt_vocab_size,
                embedding_dim=self.embed_size,
                padding_idx=self.padding_idx
            )

        self.decoder = DecoderRNN(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            output_size=self.tgt_vocab_size,
            embedder=dec_embedder,
            num_layers=self.num_layers,
            attn_mode=self.attn_mode,
            attn_hidden_size=self.attn_hidden_size,
            rnn_cell=self.rnn_cell,
            dropout=self.dropout,
            batch_first=self.batch_first,
            use_gpu=self.use_gpu
        )

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

    def forward(
            self,
            inputs,
            hidden=None,
            is_training: bool = True
    ):
        """
        forward
        """
        enc_inputs, dec_inputs = inputs["input"], inputs["target"]
        enc_outputs, enc_hidden = self.encode(enc_inputs, hidden)
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
        dec_output_dict = self.decoder(
            inputs=dec_inputs,
            hidden=enc_hidden,
            lengths=dec_input_lengths,
            encoder_outputs=enc_outputs,
            teacher_forcing_ratio=self.teacher_forcing_ratio,
            return_tokens=(not is_training)
        )
        # decoder_output_tokens = dec_output_dict["decoded_tokens"]
        decoder_outputs = dec_output_dict["outputs"]
        # decoder_last_hidden = dec_output_dict["last_hidden_state"]
        # decoder_attention_weights = dec_output_dict["attention_weights"]
        return decoder_outputs

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
            return self.infer_greedy(input=input, max_length=max_length)
        elif mode == "beam":
            return self.infer_beam(input=input, max_length=max_length, beam_width=beam_width)

    def infer_greedy(self, input, max_length=20):
        """
        input: 1-D list of integers representing tokens.
        return: 1-D list of integers representing tokens
        """

        assert isinstance(input, (list, tuple))
        assert isinstance(input[0], int)

        self.eval()
        with torch.no_grad():
            enc_inputs = torch.tensor(input, dtype=torch.long)  # shape: (seq_len)
            enc_inputs = enc_inputs.unsqueeze(0)  # shape: (1, seq_len)
            enc_inputs = enc_inputs.to(self.device)
            enc_inputs = (enc_inputs, None)

            # shape: (batch_size, seq_len)=(1, 1)
            dec_inputs = torch.tensor([self.start_token], dtype=torch.long).unsqueeze(0)
            dec_inputs = dec_inputs.to(self.device)

            enc_outputs, enc_hidden = self.encode(inputs=enc_inputs)
            dec_output_dict = self.decoder(
                inputs=dec_inputs,
                hidden=enc_hidden,
                lengths=None,
                encoder_outputs=enc_outputs,
                max_length=max_length,
                teacher_forcing_ratio=0,
                return_tokens=True
            )
            decoder_output_tokens = dec_output_dict["decoded_tokens"].squeeze(0).tolist()  # shape: (seq_len)

            if decoder_output_tokens[0] == self.end_token:
                logger.warning("The first token of the output tokens is the <EOS> token!")
                logger.warning(
                    "The related variables:\n\tEncoder inputs:{}\n\tDecoder inputs:{}".format(str(enc_inputs),
                                                                                              str(dec_inputs)))

            # Discard the content after the first end_token.
            for i in range(len(decoder_output_tokens)):
                if decoder_output_tokens[i] == self.end_token:
                    decoder_output_tokens = decoder_output_tokens[:i + 1]
                    break
            return decoder_output_tokens

    def infer_beam(self, input, max_length=20, beam_width=5):
        """
        input: 1-D list of integers representing sequence of tokens.
        return: 1-D list of integers representing tokens.

        This function is only intended for batch size of 1, inorder to be consistent
        with the `infer_greedy()`. May will support batch size greater than 1 in the future.

        Use the beam search implementation from Transformers by HuggingFace.
        The implementation code is in the class `GenerationMixin`.
        Thus, we need to adapt the input and output to fit the API.

        TODO: Test the correctness.
        """

        assert isinstance(input, (list, tuple))
        assert isinstance(input[0], int)

        batch_size = 1  # Intended for batch size of 1.

        self.eval()
        with torch.no_grad():
            enc_inputs = torch.tensor(input, dtype=torch.long)  # shape: (seq_len)
            enc_inputs = enc_inputs.unsqueeze(0)  # shape: (1, seq_len)
            enc_inputs = enc_inputs.to(self.device)
            enc_inputs = (enc_inputs, None)

            dec_inputs = torch.ones((batch_size * beam_width, 1),
                                    device=self.device,
                                    dtype=torch.long)
            dec_inputs = dec_inputs * self.start_token
            dec_inputs = dec_inputs.to(self.device)

            enc_outputs, enc_hidden = self.encode(inputs=enc_inputs)

            # add encoder_outputs to model keyword arguments
            # This format is needed in `GenerationMixin.beam_search()`.
            model_kwargs = {
                "encoder_outputs": {
                    "hidden_states":
                        enc_outputs.repeat_interleave(beam_width, dim=0),
                    "last_hidden_state":
                    # Last hidden state is not batch first.
                        enc_hidden.repeat_interleave(beam_width, dim=1)
                }
            }

            # instantiate beam scorer
            # Intended for batch size of 1.
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                max_length=max_length,
                num_beams=beam_width,
                device=self.device,
            )

            # These attributes are needed in `GenerationMixin.beam_search()`.
            self.config = Config()
            self.config.is_encoder_decoder = True
            self.config.max_length = max_length
            self.config.pad_token_id = self.padding_idx
            self.config.eos_token_id = self.end_token

            outputs = self.beam_search(
                input_ids=dec_inputs,
                beam_scorer=beam_scorer,
                max_length=max_length,
                pad_token_id=self.padding_idx,
                eos_token_id=self.end_token,
                output_attentions=False,
                **model_kwargs
            )

            sequences, sequences_scores = outputs.sequences, outputs.sequences_scores
            # Remove the <SOS> token. `GenerationMixin.beam_search()` does not exclude <SOS> token.
            sequences = sequences[:, 1:]
            sequences, sequences_scores = sequences.tolist(), sequences_scores.tolist()
            # Keep the result with the top score.
            sequences, sequences_scores = sequences[0], sequences_scores[0]

            if sequences[0] == self.end_token:
                logger.warning("The first token of the output tokens is the <EOS> token!")
                logger.warning(
                    "The related variables:\n\tEncoder inputs:{}\n\tDecoder inputs:{}".format(str(enc_inputs),
                                                                                              str(dec_inputs)))

            # Discard the content after the first end_token.
            # `max_length - 1` because of the removal of the start token.
            for i in range(max_length - 1):
                if sequences[i] == self.end_token:
                    sequences = sequences[:i + 1]
                    break
            return sequences  # , sequences_scores
