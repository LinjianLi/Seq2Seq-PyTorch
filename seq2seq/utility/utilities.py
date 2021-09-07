import torch


def max_lens(X):
    """
    max_lens
    Return: a tuple of which each element is the max length of the elements in
            the corresponding dimension of the input X.
    Example: max_lens([[1,2,3], [4,5]]) --> [2,3]
    """
    if not isinstance(X[0], list):
        return tuple([len(X)])
    elif not isinstance(X[0][0], list):
        return tuple([len(X), max(len(x) for x in X)])
    elif not isinstance(X[0][0][0], list):
        return tuple([len(X), max(len(x) for x in X),
                      max(len(x) for xs in X for x in xs)])
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def shape_fit(X):
    """
    Return: the shape of the minimum size tensor that can fit the list X into.
    Example: shape_fit([[1,2,3], [4,5]]) --> [2,3]
    """
    return max_lens(X)


def seq_mask_from_lens(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    If the `lengths` is of shape (...), the `mask` is of shape (..., max_len).
    The last dimension is of shape (max_len) and consisting of consecutive
    `True`s and `False`s. The number of `True`s is decided by the number in
    the `lengths`. `True` means that the corresponding position is not
    padding token, and `False` otherwise.

    lengths: tensor containing the lengths of sequences
    max_len: the max length of all the sequences
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(*lengths.size(), 1)
    mask = mask.lt(lengths.unsqueeze(-1))
    return mask


def seq_pad_mask_from_lens(lengths, max_len=None):
    non_pad_mask = seq_mask_from_lens(lengths, max_len)
    pad_mask = ~non_pad_mask
    return pad_mask


def list2tensor(X):
    """
    Convert an irregular shape list to a regular shape tensor padded with zero.
    Return:
        tensor: regular shape tensor padded with zero
        lengths: the lengths of the original un-padded list
    Support: 1/2/3-dimensional list.
    """
    try:
        size = shape_fit(X)
    except Exception as e:
        print(X)
        raise e

    if len(size) == 1:  # 1 dimensional list
        tensor = torch.tensor(X)
        length = torch.tensor(size[0])
        return tensor, length

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:  # 2 dimensional list
        for i, x in enumerate(X):
            length = len(x)
            tensor[i, :length] = torch.tensor(x)
            lengths[i] = length
    else:  # 3 dimensional list
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                length = len(x)
                tensor[i, j, :length] = torch.tensor(x)
                lengths[i, j] = length

    return tensor, lengths
