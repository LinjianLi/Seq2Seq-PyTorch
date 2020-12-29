import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

class SimpleRNN(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                num_layers=1,
                dropout=0,
                embedder=None,
                rnn_cell='gru',
                bidirectional=False,
                batch_first=True,
                use_gpu=False):
        super(SimpleRNN, self).__init__()

        if not batch_first:
            raise NotImplementedError(
                'Sorry, this module supports batch first mode only.')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.embedder = embedder
        if self.embedder is None:
            logger.warning("No embedder has been provided!")

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))
        self.rnn = self.rnn_cell(input_size, hidden_size, num_layers,
                                dropout=(dropout if num_layers > 1 else 0),
                                bidirectional=bidirectional,
                                batch_first=batch_first)

        self.use_gpu = use_gpu
        if self.use_gpu:
            logger.info("Using GPU")
            self.cuda()
        else:
            logger.info("Using CPU")

    def forward(self, token_seqs, input_lengths=None, hidden=None):
        """
        Input:
            token_seqs: Tensor(batch, seq_lens)
            input_lengths: Tensor(batch)

        Output:
            output: Tensor(batch, seq_len, num_directions * hidden_size)
            hidden: Tensor(num_layers * num_directions, batch_size, hidden_size)
        """
        batch_dim = 0 if self.batch_first == True else 1

        if self.embedder is not None:
            # Embedding.
            inputs = self.embedder(token_seqs)
        else:
            inputs = token_seqs

        if input_lengths is not None:
            # Sort according to the lengths.
            sorted_lengths, indices = input_lengths.sort(descending=True)
            inputs = inputs.index_select(batch_dim, indices)
            # Pack padded batch of sequences for RNN module
            inputs = pack_padded_sequence(
                        inputs,
                        sorted_lengths,
                        batch_first=self.batch_first)

        # Forward pass through RNN.
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        outputs, hidden = self.rnn(inputs, hidden)

        # if self.bidirectional:
        #     # Now hidden is of shape (num_layers, batch_size, num_directions * hidden_size)
        #     hidden = self._bridge_bidirectional_hidden(hidden)

        if input_lengths is not None:
            # Unpack padding
            outputs, _ = pad_packed_sequence(outputs, batch_first=self.batch_first)
            # Restore the outputs according to the original order of the sorted inputs.
            _, inv_indices = indices.sort()
            outputs = outputs.index_select(batch_dim, inv_indices)
            # hidden is not batch first, so dim=1.
            hidden = hidden.index_select(dim=1, index=inv_indices)

        # Return output and final hidden state.
        return outputs, hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        (Code from Baidu Inc)
        The bidirectional hidden is of shape (num_layers * num_directions, batch_size, hidden_size).
        This function is to convert it to shape (num_layers, batch_size, num_directions * hidden_size).
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)
