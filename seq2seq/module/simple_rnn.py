import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class SimpleRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            dropout: float = 0,
            embedder=None,
            rnn_cell: str = 'gru',
            bidirectional: bool = False,
            batch_first: bool = True,
            use_gpu: bool = False
    ):
        super(SimpleRNN, self).__init__()

        if not batch_first:
            raise NotImplementedError('Sorry, this module supports batch first mode only.')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # The GRU and LSTM in PyTorch have the following shape of the output and the last hidden state:
        #     output of shape (seq_len, batch, num_directions * hidden_size)
        #     h_n of shape (num_layers * num_directions, batch, hidden_size)
        #
        # When using bidirectional RNN, the RNN cell hidden size is set to the half of the network hidden size.
        # Then the outputs and the processed last hidden state will be of the network hidden size.
        self.num_directions = 2 if self.bidirectional else 1
        assert self.hidden_size % self.num_directions == 0
        self.rnn_cell_hidden_size = self.hidden_size // self.num_directions

        self.embedder = embedder
        if self.embedder is None:
            logger.warning("No embedder has been provided!")

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))
        self.rnn = self.rnn_cell(
            input_size=self.input_size,
            hidden_size=self.rnn_cell_hidden_size,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers > 1 else 0),
            bidirectional=self.bidirectional,
            batch_first=self.batch_first
        )

        self.use_gpu = use_gpu
        if self.use_gpu:
            logger.info("Using GPU")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self,
                inputs,
                input_lengths=None,
                hidden=None):
        """
        Input:
            inputs: Tensor(batch, seq_lens)
            input_lengths: Tensor(batch)

        Output: dict {
            "outputs": Tensor(batch, seq_len, num_directions * hidden_size)
            "last_hidden_state": Tensor(num_layers * num_directions, batch_size, hidden_size)
        }
        """
        batch_dim = 0 if self.batch_first else 1

        if self.embedder is not None:
            # Embedding.
            inputs = self.embedder(inputs)
        else:
            inputs = inputs

        if input_lengths is not None:
            # Sort according to the lengths.
            sorted_lengths, indices = input_lengths.sort(descending=True)
            inputs = inputs.index_select(batch_dim, indices)
            # Pack padded batch of sequences for RNN module
            inputs = pack_padded_sequence(
                inputs,
                sorted_lengths.to("cpu"),
                batch_first=self.batch_first)
            if hidden is not None:
                # Hidden states are always not batch-first.
                hidden = hidden.index_select(dim=1, index=indices)

        # Forward pass through RNN.
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        outputs, hidden = self.rnn(inputs, hidden)

        if self.bidirectional:
            # Now hidden is of shape (num_layers, batch_size, num_directions * hidden_size)
            hidden = self._bridge_bidirectional_hidden(hidden)

        if input_lengths is not None:
            # Unpack padding
            outputs, _ = pad_packed_sequence(outputs, batch_first=self.batch_first)
            # Restore the outputs according to the original order of the sorted inputs.
            _, inv_indices = indices.sort()
            outputs = outputs.index_select(batch_dim, inv_indices)
            # hidden is not batch first, so dim=1.
            hidden = hidden.index_select(dim=1, index=inv_indices)

        # Return output and final hidden state.
        output_dict = {"outputs": outputs, "last_hidden_state": hidden}
        return output_dict

    def _bridge_bidirectional_hidden(self, hidden):
        """
        (Code from Baidu Inc)
        The bidirectional hidden is of shape (num_layers * num_directions, batch_size, hidden_size).
        This function is to convert it to shape (num_layers, batch_size, num_directions * hidden_size).
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)
