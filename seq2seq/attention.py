import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

class Attention(nn.Module):
    """
    Attention
    Based on [Attention Is All You Need]
    The inputs are always batch first!

        Input:
            query: Tensor(batch_size, query_length, query_size)
            keys: Tensor(batch_size, key_length, key_size)
            values: Tensor(batch_size, key_length, value_size)
            mask: Tensor(batch_size, key_length)

        Return:
            weighted_sum_values: Tensor(batch_size, query_length, value_size)
            weights: Tensor(batch_size, query_length, key_length)
    """

    def __init__(self,
                 query_size,
                 key_size=None,
                 value_size=None,
                 hidden_size=None,
                 mode="scaled-dot",
                 return_attn_only=False,
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ("dot", "scaled-dot", "general", "mlp", "default", None)), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.key_size = key_size or query_size
        self.value_size = value_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = "scaled-dot" if mode in ("default", None) else mode
        self.return_attn_only = return_attn_only
        self.project = project
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.value_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_key = nn.Linear(
                self.key_size, self.hidden_size, bias=False)
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.value_size,
                          out_features=self.hidden_size),
                self.tanh())

    def __repr__(self):
        main_string = "Attention({}, {}, {}"\
            .format(self.query_size, self.key_size, self.value_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, keys, values=None, mask=None, debug=False):
        if self.return_attn_only == False and values == None:
            # I am not sure if I should use `.clone()` or not.
            values = keys#.clone()

        logger.debug(self)
        logger.debug('query.shape: {}\n'
                    'keys.shape: {}\n'
                    'values.shape: {}'
                    'mask.shape: {}'\
                        .format(query.shape,
                                keys.shape,
                                values.shape if values is not None else None,
                                mask.shape if mask is not None else None))

        if self.mode == "dot":
            if query.size(-1) != keys.size(-1):
                raise Exception("Attention of dot mode expects the query and"
                                "the key to have the same size of the last"
                                "dimension! But receives {} and {}."\
                                    .format(query.size(-1), keys.size(-1)))
            # (batch_size, query_length, key_length)
            attn = torch.bmm(query, keys.transpose(1, 2))

        elif self.mode == "scaled-dot":
            if query.size(-1) != keys.size(-1):
                raise Exception("Attention of dot mode expects the query and"
                                "the key to have the same size of the last"
                                "dimension! But receives {} and {}."\
                                    .format(query.size(-1), keys.size(-1)))
            # (batch_size, query_length, key_length)
            attn = torch.bmm(query, keys.transpose(1, 2))
            attn /= torch.sqrt(torch.tensor(keys.size(-1), dtype=torch.float))

        elif self.mode == "general":
            if self.key_size != keys.size(-1):
                raise Exception("Expected key size ({}) and "
                                "actural key size ({}) do not match!"\
                                    .format(self.key_size, keys.size(-1)))
            # (batch_size, query_length, key_size)
            query_linear_to_key_size = self.linear_query(query)
            # (batch_size, query_length, key_length)
            attn = torch.bmm(query_linear_to_key_size, keys.transpose(1, 2))

        elif self.mode == "mlp":
            # (batch_size, query_length, key_length, hidden_size)
            hidden = self.linear_query(query).unsqueeze(2)\
                    + self.linear_key(keys).unsqueeze(1)
            hidden = self.tanh(hidden)
            # (batch_size, query_length, key_length)
            attn = self.v(hidden).squeeze(-1)

        else:
            raise NotImplementedError("Attention mode not supported!")

        if mask is not None:
            # (batch_size, query_length, key_length)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        # (batch_size, query_length, key_length)
        weights = self.softmax(attn)

        if mask is not None:
            # If some rows (or columns) in mask are all True, then the
            # corresponding positions in attn will be all `-inf`. After the softmax
            # function, the corresponding positions in weights will be all `nan`.
            # This step is to fill the positions of `nan` with zeros.
            nan_mask = (weights != weights) # It will be true that (nan != nan).
            weights.masked_fill_(nan_mask, 0)

        if self.return_attn_only:
            return weights
        else:
            # (batch_size, query_length, value_size)
            weighted_sum_values = torch.bmm(weights, values)

            if self.project:
                project_output = self.linear_project(
                    torch.cat([weighted_sum_values, query], dim=-1))
                return project_output, weights
            else:
                return weighted_sum_values, weights
