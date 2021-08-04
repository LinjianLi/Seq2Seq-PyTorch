import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.module.attention import Attention

logger = logging.getLogger(__name__)


class DecoderRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            embedder=None,
            num_layers: int = 1,
            dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            rnn_cell: str = 'gru',
            batch_first: bool = True,
            attn_mode: str = None,
            attn_hidden_size: int = -1,
            use_gpu: bool = False
    ):

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
        self.attn_hidden_size = attn_hidden_size
        if isinstance(self.attn_mode, str) and self.attn_mode.lower() == "none":
            self.attn_mode = None
            self.attn_hidden_size = -1

        self.embedding_dropout_rate = embedding_dropout
        if self.embedding_dropout_rate != 0:
            self.embedding_dropout = nn.Dropout(self.embedding_dropout_rate)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))
        self.rnn = self.rnn_cell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers > 1 else 0),
            bidirectional=False,
            batch_first=self.batch_first
        )

        if self.attn_mode is not None:
            self.attn = Attention(
                query_size=self.hidden_size,
                key_size=self.hidden_size,
                value_size=self.hidden_size,
                hidden_size=self.attn_hidden_size,
                mode=self.attn_mode
            )
            self.linear_concat = nn.Linear(
                self.hidden_size * 2,
                self.hidden_size
            )

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.use_gpu = use_gpu
        if self.use_gpu:
            logger.info("Using GPU")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward_step(
            self,
            input_step,
            last_hidden=None,
            encoder_outputs=None,
            return_attention=False
    ):
        """
        Input:
            input_step: token tensor of shape (batch, seq_len)

        Output:
            Dict {
                log_probs:
                    tensor of shape (batch_size, seq_len, output_size)
                    representing log of token probability distribution
                last_hidden_state:
                    shape (num_layers * num_directions, batch, hidden_size)
                attention_weights:
                    None if `return_attention=False`
            }
        """
        if self.attn_mode is not None and encoder_outputs is None:
            raise ValueError("Expect encoder outputs for the attention but get none!")

        # batch_dim = 0 if self.batch_first == True else 1

        if self.embedder is not None:
            # Embedding.
            inputs = self.embedder(input_step)
            if self.embedding_dropout_rate != 0:
                inputs = self.embedding_dropout(inputs)
        else:
            inputs = input_step

        # Forward through unidirectional RNN
        # rnn_output.shape: (batch_size, seq_len, hidden_size)
        rnn_output, hidden = self.rnn(inputs, last_hidden)

        # The type of attention mechanism used here is Luong attention (Luong, 2015).
        # The Luong attention at time step `t` is computed using only the hidden state
        # of the time step `t`, instead of using the hidden state of the time step `t-1`
        # which is the Bahdanau attention (Bahdanau, 2015).
        if self.attn_mode is not None:
            # Calculate the attention.
            attn_weighted_sum, attn_weights \
                = self.attn(query=rnn_output, keys=encoder_outputs)
            cat = torch.cat((rnn_output, attn_weighted_sum), dim=-1)
            cat = torch.tanh(self.linear_concat(cat))
            h_tilde = self.out(cat)  # (batch_size, seq_len, output_size)
        else:
            attn_weights = None
            h_tilde = self.out(rnn_output)  # (batch_size, seq_len, output_size)

        # Predict probability distribution of next word.
        # Some implementation will use log-softmax instead of simple softmax.
        # The reason is explained in this discuss (https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/2).
        # "Using the log-softmax will punish bigger mistakes in likelihood space higher."
        # Also, see (https://ai.stackexchange.com/questions/12068/whats-the-advantage-of-log-softmax-over-softmax).
        # Also, see (https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
        # "Use LogSoftmax instead (it's faster and has better numerical properties)" said by the PyTorch website.
        # Maybe I should use log-softmax too.
        # output = F.softmax(h_tilde, dim=-1)
        log_probs = F.log_softmax(h_tilde, dim=-1)
        # Return output and final hidden state
        output_dict = {"log_probs": log_probs, "last_hidden_state": hidden}
        if return_attention:
            output_dict["attention_weights"] = attn_weights
        else:
            output_dict["attention_weights"] = None
        return output_dict

    def forward(
            self,
            inputs,
            hidden=None,
            lengths=None,
            encoder_outputs=None,
            max_length: int = -1,
            teacher_forcing_ratio: float = 0.0,
            return_tokens: bool = False,
            return_attention: bool = False
    ):
        """
        The forward process is greedy, that is, only consider the token with 
        the highest probability at each time step.

        Input:
            inputs: token tensor of shape (batch, seq_len) or (batch, seq_len, hidden_size)
            hidden: tensor of shape (num_layers * num_directions, batch, hidden_size)
            return_tokens:
                A boolean indicating whether or not decoder needs to decode the output
                tokens using `torch.topk()`. By setting to `False` at the training
                stage, we can reduce the memory usage and the computational cost.

        Output:
            Dict {
                decoded_tokens:
                    tensor of shape (batch_size, seq_len) representing decoded tokens
                outputs:
                    tensor of shape (batch_size, seq_len, output_size)
                    representing token log probability distributions
                last_hidden_state:
                    tensor of shape (num_layers * num_directions, batch, hidden_size)
                    containing the last hidden state of the decoder
                attention_weights:
                    tensor of shape (batch_size, seq_len, encoder_output_len)
                    representing the attention weights
            }
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        batch_dim = 0 if self.batch_first else 1
        seq_len_dim = 1 - batch_dim

        output_dict = {
            "outputs": None,
            "last_hidden_state": None,
            "decoded_tokens": None,
            "attention_weights": None
        }

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph, despite the function name "forward_step".
        # [Reference](https://github.com/IBM/pytorch-seq2seq)
        if use_teacher_forcing:
            # NOTE: The process about the first token and the length of the input
            #       should be done in the upper level Seq2Seq model
            #       instead of in the lower level decoder module.
            dec_step_output_dict = self.forward_step(inputs, hidden, encoder_outputs, return_attention=return_attention)
            decoder_outputs = dec_step_output_dict["log_probs"]
            output_dict["outputs"] = decoder_outputs
            output_dict["last_hidden_state"] = dec_step_output_dict["last_hidden_state"]
            # If unrolling is done in graph,
            # decoder_output will be of shape (batch_size, seq_len, output_size)
            if return_tokens:
                # Decode the tokens by selecting those with the highest probabilities.
                top_vals, top_ids = decoder_outputs.topk(1)
                output_dict["decoded_tokens"] = top_ids.squeeze(-1)  # shape: (batch_size, seq_len)
            else:
                output_dict["decoded_tokens"] = None
            if return_attention:
                output_dict["attention_weights"] = dec_step_output_dict["attention_weights"]
            else:
                output_dict["attention_weights"] = None

        else:
            # Tensor to store the output.
            decoder_outputs = torch.tensor([], device=("cuda" if self.use_gpu else "cpu"))
            if return_tokens:
                decoded_tokens = torch.tensor([], dtype=torch.long, device=("cuda" if self.use_gpu else "cpu"))
            else:
                decoded_tokens = None
            if return_attention:
                attention_weights = torch.tensor([], dtype=torch.float, device=("cuda" if self.use_gpu else "cpu"))
            else:
                attention_weights = None

            if max_length <= 0:
                # NOTE: The process about the first token and the length of the input
                #       should be done in the upper level Seq2Seq model
                #       instead of in the lower level decoder module.
                max_length = inputs.size(seq_len_dim)
            inputs = inputs[:, :1]  # Only take the first SOS token.
            for di in range(max_length):
                dec_step_output_dict = self.forward_step(
                    inputs,
                    hidden,
                    encoder_outputs,
                    return_attention=return_attention
                )
                step_output, hidden = dec_step_output_dict["log_probs"], dec_step_output_dict["last_hidden_state"]
                # step_output.shape: (batch, 1, output_size)
                # decoder_outputs.shape: (batch, seq_len, output_size)
                decoder_outputs = torch.cat((decoder_outputs, step_output), dim=seq_len_dim)
                # Without teacher forcing, the next input is decoder's own current output
                # Decode the tokens by selecting those with the highest probabilities.
                top_vals, top_ids = step_output.squeeze(1).topk(1)  # shape: (batch, 1)
                if return_tokens:
                    decoded_tokens = torch.cat((decoded_tokens, top_ids),
                                               dim=seq_len_dim)  # shape: (batch_size, seq_len)
                if return_attention:
                    attention_weights = torch.cat((attention_weights, dec_step_output_dict["attention_weights"]),
                                                  dim=seq_len_dim)
                inputs = top_ids
            output_dict["outputs"] = decoder_outputs
            output_dict["last_hidden_state"] = hidden
            if return_tokens:
                output_dict["decoded_tokens"] = decoded_tokens
            else:
                output_dict["decoded_tokens"] = None
            if return_attention:
                output_dict["attention_weights"] = attention_weights
            else:
                output_dict["attention_weights"] = None
        return output_dict
