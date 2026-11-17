"""Efficient predicate-argument span classifier for Task3 Experiment C."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class EfficientSpanRoleClassifier(nn.Module):
    def __init__(self, config, span_dim=256, role_dim=256,
                 distance_dim=32, width_dim=16, max_distance=128, max_width=31):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.span_dim = span_dim
        self.role_dim = role_dim
        self.max_distance = max_distance
        self.max_width = max_width
        self.bert = BertModel(config)
        dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        hidden = config.hidden_size
        span_input = hidden * 3
        self.argument_projection = nn.Sequential(
            nn.Linear(span_input, span_dim), nn.GELU(), nn.LayerNorm(span_dim), nn.Dropout(dropout)
        )
        self.target_projection = nn.Sequential(
            nn.Linear(span_input, span_dim), nn.GELU(), nn.LayerNorm(span_dim), nn.Dropout(dropout)
        )
        self.distance_embedding = nn.Embedding(2 * max_distance + 1, distance_dim)
        self.width_embedding = nn.Embedding(max_width + 1, width_dim)
        interaction_size = span_dim * 4 + distance_dim + width_dim
        self.interaction = nn.Sequential(
            nn.Linear(interaction_size, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
            nn.Dropout(dropout),
            nn.Linear(role_dim, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
        )
        self.role_embeddings = nn.Parameter(torch.empty(self.num_labels, role_dim))
        self.role_bias = nn.Parameter(torch.zeros(self.num_labels))
        nn.init.normal_(self.role_embeddings, mean=0.0, std=0.02)

    @staticmethod
    def _span_components(hidden, spans):
        device = hidden.device
        span_tensor = torch.as_tensor(spans, dtype=torch.long, device=device)
        batch = torch.arange(hidden.shape[0], device=device)
        start = hidden[batch, span_tensor[:, 0]]
        end = hidden[batch, span_tensor[:, 1]]
        positions = torch.arange(hidden.shape[1], device=device)[None, :]
        mask = (
            (positions >= span_tensor[:, 0, None])
            & (positions <= span_tensor[:, 1, None])
        )
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.cat([start, end, mean], dim=-1), span_tensor

    def forward(self, input_ids, attention_mask, target, argument):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        states = bert_output.hidden_states
        hidden = (states[-4] + states[-3] + states[-2] + states[-1]) / 4.0
        target_raw, target_tensor = self._span_components(hidden, target)
        argument_raw, argument_tensor = self._span_components(hidden, argument)
        target_repr = self.target_projection(target_raw)
        argument_repr = self.argument_projection(argument_raw)

        target_center = (target_tensor[:, 0] + target_tensor[:, 1]) // 2
        argument_center = (argument_tensor[:, 0] + argument_tensor[:, 1]) // 2
        signed_distance = (argument_center - target_center).clamp(
            -self.max_distance, self.max_distance
        ) + self.max_distance
        width = (argument_tensor[:, 1] - argument_tensor[:, 0]).clamp(0, self.max_width)
        distance_feature = self.distance_embedding(signed_distance)
        width_feature = self.width_embedding(width)
        interaction = torch.cat([
            argument_repr,
            target_repr,
            argument_repr * target_repr,
            torch.abs(argument_repr - target_repr),
            distance_feature,
            width_feature,
        ], dim=-1)
        role_query = self.interaction(interaction)
        logits = F.linear(role_query, self.role_embeddings, self.role_bias)
        return {'logits': logits, 'role_query': role_query}

