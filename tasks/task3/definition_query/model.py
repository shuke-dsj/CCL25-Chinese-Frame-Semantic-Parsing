"""Cross-encoder used by Task3 Experiment D3."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class DefinitionQueryCrossEncoder(nn.Module):
    def __init__(self, config, projection_dim=512):
        super().__init__()
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden * 5, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(projection_dim, 1),
        )

    @staticmethod
    def _gather(hidden, positions):
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, positions]

    def forward(
        self, input_ids, attention_mask, token_type_ids, marker_positions,
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden = sum(output.hidden_states[-4:]) / 4.0
        cls = hidden[:, 0]
        target = 0.5 * (
            self._gather(hidden, marker_positions['target_start'])
            + self._gather(hidden, marker_positions['target_end'])
        )
        argument = 0.5 * (
            self._gather(hidden, marker_positions['argument_start'])
            + self._gather(hidden, marker_positions['argument_end'])
        )
        features = torch.cat([
            cls,
            target,
            argument,
            torch.abs(target - argument),
            target * argument,
        ], dim=-1)
        return self.projection(features).squeeze(-1)
