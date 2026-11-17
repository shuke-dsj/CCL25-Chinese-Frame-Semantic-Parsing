"""Boundary auxiliary supervision for Task2 GlobalPointer (experiment H1)."""

import torch
import torch.nn as nn

from .decoding import build_safe_span_mask
from .model import Model as BaselineModel


class BoundaryAuxiliaryModel(BaselineModel):
    """Keep span decoding unchanged while supervising start/end boundaries."""

    def __init__(self, config):
        super().__init__(config)
        self.boundary_loss_weight = float(
            getattr(config, 'task2_boundary_loss_weight', 0.1)
        )
        if self.boundary_loss_weight < 0:
            raise ValueError('task2_boundary_loss_weight must be non-negative')
        self.boundary_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.boundary_start = nn.Linear(config.hidden_size, 1)
        self.boundary_end = nn.Linear(config.hidden_size, 1)

    @staticmethod
    def _multilabel_logsumexp_loss(logits, labels, valid_mask):
        valid = valid_mask.to(logits.dtype)
        labels = labels.to(logits.dtype)
        positive = torch.sum(torch.exp(-logits) * labels * valid, dim=1)
        negative = torch.sum(torch.exp(logits) * (1.0 - labels) * valid, dim=1)
        return torch.mean(torch.log1p(positive) + torch.log1p(negative))

    def compute_auxiliary_loss(self, hidden_token, labels, attention_mask, target):
        hidden = self.boundary_dropout(hidden_token)
        start_logits = self.boundary_start(hidden).squeeze(-1)
        end_logits = self.boundary_end(hidden).squeeze(-1)
        outputs = {
            'boundary_start_logits': start_logits,
            'boundary_end_logits': end_logits,
            'boundary_loss_weight': self.boundary_loss_weight,
        }
        if labels is None:
            return None, outputs

        valid_span = build_safe_span_mask(attention_mask, target)
        start_valid = valid_span.any(dim=2)
        end_valid = valid_span.any(dim=1)
        start_labels = labels.bool().any(dim=2)
        end_labels = labels.bool().any(dim=1)
        start_loss = self._multilabel_logsumexp_loss(
            start_logits, start_labels, start_valid
        )
        end_loss = self._multilabel_logsumexp_loss(
            end_logits, end_labels, end_valid
        )
        boundary_loss = 0.5 * (start_loss + end_loss)
        outputs.update({
            'boundary_loss': boundary_loss,
            'boundary_start_loss': start_loss,
            'boundary_end_loss': end_loss,
            'boundary_start_labels': start_labels,
            'boundary_end_labels': end_labels,
            'boundary_start_valid': start_valid,
            'boundary_end_valid': end_valid,
        })
        return self.boundary_loss_weight * boundary_loss, outputs
