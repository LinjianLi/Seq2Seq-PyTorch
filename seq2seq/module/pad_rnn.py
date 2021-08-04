import logging
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


def pad_rnn(
        inputs,
        hidden_size,
        lengths=None,
        rnn=None,
        embedder=None,
        hidden=None,
        num_layers: int = 1,
        batch_first: bool = True
):
    """
    forward
    input.shape: (batch, max_len, xxx) where xxx can be
                    any number (including 0) of dimensions
    lengths.shape: (batch)
    """
    assert rnn is not None
    if embedder is not None:
        rnn_inputs = embedder(inputs)
    else:
        rnn_inputs = inputs

    batch_dim = 0 if batch_first else 1
    batch_size = inputs.size(batch_dim)

    # Sort the inputs and hidden by descending input lengths.
    if lengths is not None:
        num_valid = lengths.gt(0).int().sum().item()
        sorted_lengths, indices = lengths.sort(descending=True)
        rnn_inputs = rnn_inputs.index_select(batch_dim, indices)
        rnn_inputs = pack_padded_sequence(
            rnn_inputs[:num_valid],
            sorted_lengths[:num_valid].tolist(),
            batch_first=batch_first)

        if hidden is not None:
            # The `hidden` is not batch first.
            hidden = hidden.index_select(1, indices)[:, :num_valid]

    outputs, last_hidden = rnn(rnn_inputs, hidden)

    if lengths is not None:
        outputs, _ = pad_packed_sequence(outputs, batch_first=batch_first)

        if num_valid < batch_size:
            zeros = outputs.new_zeros(
                batch_size - num_valid, outputs.size(1), hidden_size)
            # Complete the batch by padding zero.
            outputs = torch.cat([outputs, zeros], dim=batch_dim)

            zeros = last_hidden.new_zeros(
                num_layers, batch_size - num_valid, hidden_size)
            # Complete the batch by padding zero.
            last_hidden = torch.cat([last_hidden, zeros], dim=1)

        # Restore the sorted inputs to original order.
        _, inv_indices = indices.sort()
        outputs = outputs.index_select(batch_dim, inv_indices)
        last_hidden = last_hidden.index_select(1, inv_indices)

    return outputs, last_hidden, lengths
