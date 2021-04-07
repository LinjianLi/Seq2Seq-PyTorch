# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Linjian Li

from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple, Union
from collections import OrderedDict
import logging

import torch
from torch.nn import functional as F

# from .generation_beam_search import BeamScorer, BeamSearchScorer
from .beam_search_utils import BeamScorer, BeamSearchScorer


logger = logging.getLogger(__name__)

def is_tensor(x):
    """ Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`. """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return True
    except:
        import numpy
        return isinstance(x, numpy.ndarray)

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, num_heads,
            generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]


class GenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.PreTrainedModel`.
    """

    """
    I modify this source code from the
    [Transformers](https://github.com/huggingface/transformers).
    I need to use it in a simple RNN sequence-to-sequence model to implement beam
    search algorithm. Since my model is not transformer-based, but RNN-based, some
    input and output specifications need to be modified to fit the RNN model.

    For instance, for the transformer, the output of is always one step further
    than the input. So, in each time step of beam search, we can input the whole
    partial candidate and the last one element of the output is the predicted token
    for next time setp. Then, we can concatenate the last element with the input
    partial candidate to form a new candidate and repeat the process. However, the
    `forward()` function of my RNN seq2seq model is in fact greedy search process,
    which will keep moving forward until reaching the max length. What's more, my
    RNN seq2seq model does not have attention mechanism at encoding stage.

        -- Linjian Li
    """

    @staticmethod
    def _update_seq_length_for_generation(
        sequence_lengths: torch.LongTensor,
        unfinished_sequences: torch.LongTensor,
        cur_len: int,
        is_eos_in_next_token: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        # check if sentence is not finished yet
        is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

        # update sentence length
        sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
        unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
        return sequence_lengths, unfinished_sequences

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        """
        About Parallel:

            To utilize the parallel computation power, we can transform the candidates of
            shape (batch_size, beam_width, sequence_length) to the shape of
            (batch_size * beam_width, sequence_length), as well as the scores. If we omit
            the dimension of sequence length, the transformation is similar to flatting a
            2-D matrix to a 1-D array. The n-th element of the flatted array is the
            (n / num_rows, n % num_columns) element of the original 2-D matrix. Note that
            the division operation is integer division. That is how we can recover the top
            beams of each batch from the flatted tensor.

                -- Linjian Li
        """

        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # init attention / hidden states / scores tuples
        scores = ()
        decoder_attentions = () if output_attentions else None
        decoder_hidden_states = ()

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.config.is_encoder_decoder:
            """
            My RNN-based seq2seq model does not have attention mechanism at encoding stage.
                -- Linjian Li
            """
            encoder_attentions = None  # model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states")
            encoder_last_hidden_state = model_kwargs["encoder_outputs"].get("last_hidden_state")

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Initialize scores of all other beams, except the first one, to be very small. 
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # One step at a time, until the max_length steps have done.
        while cur_len < max_length:

            """
            At each time step, the whole candidate sequence and the initial hidden state are
            input to the model. The last step of the output is the new decoded state.

            The disadvantage of this is that the whole partial candidate sequence is input
            to the model each time, which makes the computational cost larger than inputting
            single time step with the hidden state each time.

                -- Linjian Li
            """
            # output_tokens, outputs, hidden = self.decoder.forward(
            output_dict = self.decoder.forward(
                inputs=input_ids,
                hidden=encoder_last_hidden_state,
                lengths=None,
                encoder_outputs=encoder_hidden_states,
                max_length=-1,
                teacher_forcing_ratio=0,
                return_tokens=False,
                return_attention=output_attentions
            )
            outputs = output_dict["outputs"]
            # next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)
            # next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            next_token_scores = outputs[:, -1, :]
            """ 
            Add the previous beam scores to the next token scores,
            inorder to pick the top-k candidates and keep track of the scores of candidates.
                -- Linjian Li
            """
            # next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = next_token_scores + beam_scores.view(-1, 1).expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (output_dict["attention_weights"], )

            decoder_hidden_states += (output_dict["last_hidden_state"], )

            # reshape for beam search
            """
            Reshape the scores and group them by the batch.
            Then we can use `topk()` and set `dim=-1` to get the top-k candidates in each batch.
                -- Linjian Li
            """
            vocab_size = next_token_scores.shape[-1]  # next_token_scores: (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)

            """
            I do not understand the purpose of setting `k = 2 * num_beams`
            instead of `k = num_beams`.
                -- Linjian Li
            """
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            """
            If we omit the dimension of batch size, the previous operation is like
            transform a (num_beams, vocab_size) 2-D matrix to a (num_beams * vocab_size)
            1-D array. The n-th element of the flatted array is the (n // num_rows,
            n % num_columns) element of the original 2-D matrix.
                -- Linjian Li
            """
            next_indices = next_tokens // vocab_size  # The original beam ID of the top-k tokens. (batch_size, num_beams)
            next_tokens = next_tokens % vocab_size  # The actual token ID in vocab of the top-k tokes. (batch_size, num_beams)

            """
            Setting `k = 2 * num_beams` in the above `topk()` function makes the number of
            obtained tokens and scores exceed the beam width. Thus, we need to process them
            to acutally get the tokens and scores within the number of beam width.
                -- Linjian Li
            """
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        """
        If some candidates are still not finished after max_len time steps, we need to
        finish them and see if the scores are better than those who have finished.
            -- Linjian Li
        """
        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
