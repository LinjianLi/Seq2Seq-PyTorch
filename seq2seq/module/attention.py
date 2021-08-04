import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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

    def __init__(
            self,
            query_size: int,
            key_size=None,
            value_size=None,
            hidden_size=None,
            mode: str = "scaled-dot",
            return_attn_only: bool = False,
            project: bool = False
    ):
        super(Attention, self).__init__()

        mode = mode.lower()
        assert (
                mode in (
            "dot",
            "scaled-dot",
            "general",
            "concat",
            "mlp",
            "additive",
            "default",
            None
        )
        ), (
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

        if self.mode in ("concat", "mlp", "additive"):
            assert (
                    self.hidden_size > 0
            ), (
                "Attention mode is \"{mode}\" but receive attention hidden size of {hidden_size}".format(
                    mode=self.mode, hidden_size=self.hidden_size
                )
            )

        if self.mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.key_size, bias=False)

        elif self.mode == "concat":
            self.W = nn.Linear(
                self.query_size + self.key_size,
                self.hidden_size,
                bias=False
            )
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        elif self.mode == "mlp" or self.mode == "additive":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=False)
            self.linear_key = nn.Linear(
                self.key_size, self.hidden_size, bias=False)
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.value_size,
                          out_features=self.hidden_size),
                self.tanh()
            )

    def __repr__(self):
        main_string = "Attention(query_size={}, key_size={}, value_size={}" \
            .format(self.query_size, self.key_size, self.value_size)
        if self.mode in ("concat", "mlp", "additive"):
            main_string += ", hidden_size={}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, keys, values=None, mask=None):
        if (not self.return_attn_only) and (values is None):
            # I am not sure if I should use `.clone()` or not.
            values = keys

        if self.mode == "dot" or self.mode == "scaled-dot":
            if query.size(-1) != keys.size(-1):
                raise Exception(
                    "Attention of dot mode expects the query and"
                    "the key to have the same size of the last"
                    "dimension! But receives {} and {}.".format(
                        query.size(-1), keys.size(-1))
                )
            # (batch_size, query_length, key_length)
            attn = torch.bmm(query, keys.transpose(1, 2))
            if self.mode == "scaled-dot":
                attn /= torch.sqrt(torch.tensor(keys.size(-1), dtype=torch.float))

        elif self.mode == "general":
            if self.key_size != keys.size(-1):
                raise Exception(
                    "Expected key size ({}) and "
                    "actual key size ({}) do not match!".format(
                        self.key_size, keys.size(-1))
                )
            # (batch_size, query_length, key_size)
            query_linear_to_key_size = self.linear_query(query)
            # (batch_size, query_length, key_length)
            attn = torch.bmm(query_linear_to_key_size, keys.transpose(1, 2))

        elif self.mode == "concat":
            if self.key_size != keys.size(-1):
                raise Exception(
                    "Expected key size ({}) and "
                    "actual key size ({}) do not match!".format(
                        self.key_size, keys.size(-1))
                )
            num_queries, num_keys = query.size(1), keys.size(1)
            query = query.unsqueeze(2).expand(-1, -1, num_keys, -1)
            keys = keys.unsqueeze(1).expand(-1, num_queries, -1, -1)
            attn = self.v(  # (batch_size, query_length, key_length, 1)
                self.tanh(
                    self.W(  # (batch_size, query_length, key_length, hidden_size)
                        # (batch_size, query_length, query_size + key_size)
                        torch.cat((query, keys), dim=-1)
                    )
                )
            ).squeeze(-1)  # (batch_size, query_length, key_length)

        elif self.mode == "mlp" or self.mode == "additive":
            # (batch_size, query_length, key_length)
            attn = self.v(  # (batch_size, query_length, key_length, 1)
                self.tanh(
                    # (batch_size, query_length, key_length, hidden_size)
                    self.linear_query(query).unsqueeze(2) + self.linear_key(keys).unsqueeze(1)
                )
            ).squeeze(-1)  # (batch_size, query_length, key_length)

        else:
            raise NotImplementedError("Attention mode not supported!")

        if mask is not None:
            # (batch_size, query_length, key_length)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        # (batch_size, query_length, key_length)
        weights = self.softmax(attn)

        if mask is not None:
            # If some rows (or columns) in mask are all True, then the corresponding
            # positions in `attn` will be all `-inf`. After the softmax function,
            # those positions in weights will be all `nan`s.
            # This step is to fill the positions of `nan` with zeros.
            nan_mask = (weights != weights)  # It will be true that (nan != nan).
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
