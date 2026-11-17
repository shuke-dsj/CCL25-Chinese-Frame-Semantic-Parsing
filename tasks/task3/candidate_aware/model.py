"""Base-preserving candidate-aware Definition residual reranker (D4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.task3.definition_query.model import DefinitionQueryCrossEncoder


class CandidateAwareDefinitionReranker(nn.Module):
    def __init__(self, config, projection_dim=512, pair_frames_per_role=2):
        super().__init__()
        self.pair_encoder = DefinitionQueryCrossEncoder(
            config, projection_dim=projection_dim,
        )
        self.pair_frames_per_role = int(pair_frames_per_role)
        self.gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        # Zero makes the initial final score exactly equal to the frozen
        # generator.  Auxiliary definition CE still trains the new encoder.
        self.residual_scale_raw = nn.Parameter(torch.tensor(0.0))

    def score_pairs(
        self, input_ids, attention_mask, token_type_ids, marker_positions,
        pair_batch_size=0,
    ):
        total = len(input_ids)
        step = total if not pair_batch_size else int(pair_batch_size)
        scores = []
        for start in range(0, total, step):
            stop = min(start + step, total)
            scores.append(self.pair_encoder(
                input_ids=input_ids[start:stop],
                attention_mask=attention_mask[start:stop],
                token_type_ids=token_type_ids[start:stop],
                marker_positions={
                    key: value[start:stop]
                    for key, value in marker_positions.items()
                },
            ))
        return torch.cat(scores, dim=0)

    def combine(self, pair_scores, generator_scores, frame_log_weights, group_features):
        batch, candidates, frames = frame_log_weights.shape
        expected = batch * candidates * frames
        if pair_scores.numel() != expected:
            raise ValueError(
                f'D4 pair shape mismatch: {pair_scores.numel()} != {expected}'
            )
        pair_scores = pair_scores.view(batch, candidates, frames)
        definition_scores = torch.logsumexp(
            pair_scores + frame_log_weights, dim=-1,
        )
        centered = definition_scores - definition_scores.mean(dim=1, keepdim=True)
        definition_z = centered / definition_scores.std(
            dim=1, keepdim=True, unbiased=False,
        ).clamp_min(1e-6)
        gate = torch.sigmoid(self.gate(group_features)).squeeze(-1)
        residual_scale = 2.0 * torch.tanh(self.residual_scale_raw)
        final_scores = (
            generator_scores
            + residual_scale * gate[:, None] * definition_z
        )
        return {
            'final_scores': final_scores,
            'definition_scores': definition_scores,
            'definition_z': definition_z,
            'gate': gate,
            'residual_scale': residual_scale,
            'pair_scores': pair_scores,
        }

    def forward(
        self, input_ids, attention_mask, token_type_ids, marker_positions,
        generator_scores, frame_log_weights, group_features,
        pair_batch_size=0,
    ):
        pair_scores = self.score_pairs(
            input_ids, attention_mask, token_type_ids, marker_positions,
            pair_batch_size=pair_batch_size,
        )
        return self.combine(
            pair_scores, generator_scores, frame_log_weights, group_features,
        )

