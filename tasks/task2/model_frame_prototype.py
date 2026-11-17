"""Cheap oracle-frame prototype conditioning for Task2 (local experiment G2)."""
import math

import torch
import torch.nn as nn

from .model import Model as BaselineModel


class FramePrototypeConditionedModel(BaselineModel):
    """Add fixed frame-definition prototype compatibility to span endpoints."""

    def __init__(self, config):
        super().__init__(config)
        payload = torch.load(config.task2_frame_prototype_file, map_location='cpu')
        prototypes = payload['prototypes'].float()
        if prototypes.shape != (config.max_cls, config.hidden_size):
            raise ValueError(
                f'prototype shape {tuple(prototypes.shape)} != '
                f'({config.max_cls}, {config.hidden_size})'
            )
        expected_names = getattr(config, 'task2_frame_names', None)
        if expected_names is not None and list(payload['idx2label']) != list(expected_names):
            raise ValueError('frame prototype order does not match frame_info order')
        self.register_buffer('frame_prototypes', prototypes)
        self.frame_query_dim = int(getattr(config, 'task2_frame_query_dim', 64))
        self.frame_scale = nn.Parameter(torch.tensor(
            float(getattr(config, 'task2_frame_scale_init', 0.1)), dtype=torch.float
        ))
        hidden = config.hidden_size
        self.frame_norm = nn.LayerNorm(hidden)
        self.frame_start_query = nn.Linear(hidden, self.frame_query_dim)
        self.frame_end_query = nn.Linear(hidden, self.frame_query_dim)
        self.frame_start_token = nn.Linear(hidden, self.frame_query_dim)
        self.frame_end_token = nn.Linear(hidden, self.frame_query_dim)
        self.frame_dropout = nn.Dropout(config.hidden_dropout_prob)

    def augment_logits(self, logits, hidden_token, target, attention_mask, target_cls=None):
        if target_cls is None:
            raise ValueError('frame_prototype_conditioned requires oracle target_cls')
        frame_vector = self.frame_prototypes.index_select(0, target_cls.long())
        frame_vector = self.frame_dropout(self.frame_norm(frame_vector))
        start_query = self.frame_start_query(frame_vector)
        end_query = self.frame_end_query(frame_vector)
        start_token = self.frame_start_token(hidden_token)
        end_token = self.frame_end_token(hidden_token)
        start_score = torch.einsum('bd,bld->bl', start_query, start_token)
        end_score = torch.einsum('bd,bld->bl', end_query, end_token)
        start_score = start_score / math.sqrt(self.frame_query_dim)
        end_score = end_score / math.sqrt(self.frame_query_dim)
        return logits + self.frame_scale * (
            start_score.unsqueeze(-1) + end_score.unsqueeze(-2)
        )
