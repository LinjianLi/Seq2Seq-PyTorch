import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import Attention


logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

class DecoderRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 dropout=0,
                 embedding_dropout=0,
                 rnn_cell='gru',
                 batch_first=True,
                 attn_mode=None,
                 attn_hidden_size=-1,
                 use_gpu=False):

        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        if self.embedder is None:
            logger.warning("No embedder has been provided!")
        self.num_layers = num_layers
        self.dropout = dropout

        if not batch_first:
            raise NotImplementedError(
                'Sorry, this module supports batch first mode only.')
        self.batch_first = batch_first

        self.attn_mode = attn_mode
        if isinstance(self.attn_mode, str) and self.attn_mode.lower() == "none":
            self.attn_mode = None
        self.attn_hidden_size = attn_hidden_size

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))
        self.rnn = self.rnn_cell(input_size=self.input_size,
                                hidden_size =self.hidden_size,
                                num_layers =self.num_layers,
                                dropout=(self.dropout if self.num_layers > 1 else 0),
                                bidirectional=False,
                                batch_first=self.batch_first)

        if self.attn_mode is not None:
            self.attn = Attention(query_size=self.hidden_size,
                                            key_size=self.hidden_size,
                                            value_size=self.hidden_size,
                                            hidden_size=self.attn_hidden_size,
                                            mode=attn_mode)
            self.linear_concat = nn.Linear(self.hidden_size * 2,
                                            self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.use_gpu = use_gpu
        if self.use_gpu:
            logger.info("Using GPU")
            self.cuda()
        else:
            logger.info("Using CPU")

    def forward_step(self,
                     input_step,
                     last_hidden=None,
                     encoder_outputs=None):
        """

        Input:
            input_step: token tensor of shape (batch, seq_len)

        Output:
            log_probs: tensor of shape (batch_size, seq_len, output_size)
                        representing log of token probability distribution
            hidden: shape (num_layers * num_directions, batch, hidden_size)
        """
        if self.attn_mode is not None and encoder_outputs is None:
            raise ValueError("Expect encoder outputs for the attention but get none!")

        # batch_dim = 0 if self.batch_first == True else 1

        if self.embedder is not None:
            # Embedding.
            inputs = self.embedder(input_step)
            inputs = self.embedding_dropout(inputs)
        else:
            inputs = input_step

        # Forward through unidirectional RNN
        # rnn_output.shape: (batch_size, seq_len, hidden_size)
        rnn_output, hidden = self.rnn(inputs, last_hidden)

        if self.attn_mode is not None:
            # Calculate the attention.
            attn_weighted_sum, attn_weights\
                = self.attn(query=rnn_output, keys=encoder_outputs)
            # Use attention output according to (Luong, 2015).
            cat = torch.cat((rnn_output, attn_weighted_sum), dim=-1)
            cat = torch.tanh(self.linear_concat(cat))
            h_tilde = self.out(cat)   # (batch_size, seq_len, output_size)
        else:
            h_tilde = self.out(rnn_output)   # (batch_size, seq_len, output_size)

        # Predict probability distribution of next word.
        # Some implementation will use log-softmax instead of simple softmax.
        # The reason is explained in this discuss (https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/2).
        # "Using the log-softmax will punish bigger mistakes in likelihood space higher."
        # Also, see (https://ai.stackexchange.com/questions/12068/whats-the-advantage-of-log-softmax-over-softmax).
        # Also, see (https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
        # "Use LogSoftmax instead (it’s faster and has better numerical properties)" said by the PyTorch website.
        # Maybe I should use log-softmax too.
        # output = F.softmax(h_tilde, dim=-1)
        log_probs = F.log_softmax(h_tilde, dim=-1)
        # Return output and final hidden state
        return log_probs, hidden

    def forward(self,
                inputs,
                hidden=None,
                lengths=None,
                encoder_outputs=None,
                max_length=None,
                teacher_forcing_ratio=0):
        """
        Input:
            inputs: token tensor of shape (batch, seq_len) or (batch, seq_len, hidden_size)
            hidden: tensor of shape (num_layers * num_directions, batch, hidden_size)

        Output:
            decoder_output_tokens: tensor of shape (batch_size, seq_len)
                                   representing decoded tokens
            decoder_outputs: tensor of shape (batch_size, seq_len, output_size)
                             representing log of token distributions
            hidden: tensor of shape (num_layers * num_directions, batch, hidden_size)
                            containing the last hidden state of the decoder
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Tensor to store the output.
        # decoder_output_tokens = torch.tensor([], dtype=torch.long, device=("cuda" if self.use_gpu else "cpu"))
        # decoder_output_tokens = inputs[:, :1] if inputs.dim() == 2 else torch.tensor([], dtype=torch.long, device=("cuda" if self.use_gpu else "cpu"))
        decoder_output_tokens = torch.tensor([], dtype=torch.long, device=("cuda" if self.use_gpu else "cpu"))
        decoder_outputs = torch.tensor([], device=("cuda" if self.use_gpu else "cpu"))

        batch_dim = 0 if self.batch_first else 1
        seq_len_dim = 1 - batch_dim

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph, despite the function name "forward_step".
        if use_teacher_forcing:
            decoder_input = inputs#[:, :-1]
            decoder_output, hidden\
                = self.forward_step(decoder_input, hidden, encoder_outputs)
            # If unrolling is done in graph,
            # decoder_output will be of shape (batch_size, seq_len, output_size)
            top_vals, top_ids = decoder_output.topk(1)
            decoder_output_tokens\
                = torch.cat((decoder_output_tokens, top_ids.squeeze(-1)),
                            dim=seq_len_dim) # shape: (batch_size, seq_len)
            decoder_outputs = decoder_output

        else:
            if max_length is None:
                max_length = inputs.size(seq_len_dim)
            decoder_input = inputs[:, :1]#[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, hidden\
                    = self.forward_step(decoder_input, hidden, encoder_outputs)
                # decoder_output.shape: (batch, 1, output_size)
                # decoder_outputs.shape: (batch, seq_len, output_size)
                decoder_outputs = torch.cat((decoder_outputs, decoder_output), dim=seq_len_dim)
                # Without teacher forcing, the next input is decoder's own current output
                step_output = decoder_output.squeeze(1) # shape: (batch, output_size)
                top_vals, top_ids = step_output.topk(1) # shape: (batch, 1)
                decoder_output_tokens\
                    = torch.cat((decoder_output_tokens, top_ids),
                                dim=seq_len_dim) # shape: (batch_size, seq_len)
                decoder_input = top_ids
        logger.debug("\nDecoder input tokens: {}\nDecoder output tokens: {}"\
                        .format(inputs, decoder_output_tokens))
        return decoder_output_tokens, decoder_outputs, hidden
