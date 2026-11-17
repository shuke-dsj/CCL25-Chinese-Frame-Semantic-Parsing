"""Additive content-aware span scorer for Task2 argument identification."""

import math

import torch
import torch.nn as nn

from .model import Model as BaselineModel
from .target_utils import mean_pool_inserted_target


class ContentAwareModel(BaselineModel):
    """Keep GlobalPointer endpoints and add low-dimensional span-content score.

    Span means are computed for every (start, end) pair using prefix sums. The
    projection dimension is deliberately small, keeping memory at O(B*L^2*d)
    with d << BERT hidden size while retaining nested-span support.
    """

    def __init__(self, config):
        super().__init__(config)
        self.content_dim = int(getattr(config, 'task2_content_dim', 32))
        if self.content_dim <= 0:
            raise ValueError(f'task2_content_dim must be positive, got {self.content_dim}')
        scale_init = float(getattr(config, 'task2_content_scale_init', 0.1))
        self.content_projection = nn.Linear(config.hidden_size, self.content_dim)
        self.content_target_projection = nn.Linear(config.hidden_size, self.content_dim)
        self.content_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.content_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float))

    def augment_logits(self, logits, hidden_token, target, attention_mask, target_cls=None):
        token_content = torch.tanh(self.content_projection(hidden_token))
        token_content = self.content_dropout(token_content)
        target_vector = mean_pool_inserted_target(hidden_token, target)
        target_content = torch.tanh(self.content_target_projection(target_vector))

        batch_size, sequence_length, content_dim = token_content.shape
        zero = token_content.new_zeros(batch_size, 1, content_dim)
        prefix = torch.cat([zero, token_content.cumsum(dim=1)], dim=1)
        # prefix[j+1] - prefix[i] gives the inclusive span sum for (i, j).
        span_sum = prefix[:, 1:, :].unsqueeze(1) - prefix[:, :-1, :].unsqueeze(2)
        positions = torch.arange(sequence_length, device=hidden_token.device)
        span_length = (positions.unsqueeze(0) - positions.unsqueeze(1) + 1).clamp_min(1)
        span_mean = span_sum / span_length.to(hidden_token.dtype).unsqueeze(0).unsqueeze(-1)
        content_logits = torch.einsum('bijd,bd->bij', span_mean, target_content)
        content_logits = content_logits / math.sqrt(self.content_dim)
        # Prefix subtraction is meaningful only for start <= end. Keeping the
        # lower triangle at zero also prevents masked exp(logit) from seeing
        # artificially large reversed-span scores.
        upper_triangle = torch.triu(torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=hidden_token.device
        ))
        content_logits = content_logits.masked_fill(~upper_triangle.unsqueeze(0), 0.0)
        return logits + self.content_scale * content_logits
