"""Target-conditioned GlobalPointer for Task2 argument span detection."""

import torch
import torch.nn as nn

from .model import Model as BaselineModel
from .target_utils import build_inserted_target_mask, mean_pool_inserted_target


class TargetConditionedModel(BaselineModel):
    """Inject an explicitly pooled target representation before q/k projection.

    Dataset target coordinates refer to BERT positions before the two target
    markers are inserted. After insertion, target characters occupy
    [target_start + 1, target_end + 1] (inclusive).
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.target_projection = nn.Linear(hidden_size, hidden_size)
        self.target_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.target_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.target_dropout = nn.Dropout(config.hidden_dropout_prob)

    @staticmethod
    def build_target_mask(hidden_token, target):
        return build_inserted_target_mask(hidden_token, target)

    def condition_hidden(self, hidden_token, target):
        target_vector = mean_pool_inserted_target(hidden_token, target)
        target_update = torch.tanh(self.target_projection(target_vector))
        expanded_target = target_update.unsqueeze(1).expand(-1, hidden_token.shape[1], -1)
        gate = torch.sigmoid(self.target_gate(torch.cat([hidden_token, expanded_target], dim=-1)))
        conditioned = hidden_token + self.target_dropout(gate * expanded_target)
        return self.target_layer_norm(conditioned)
