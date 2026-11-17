"""Experiment C3: uncertainty-aware global/local role classifier.

The strong Experiment-C predicate--argument encoder remains the global branch.
C3 adds a frame-conditioned residual branch, but never hard-removes a role.
The residual is dynamically gated by the amount/confidence/frequency of the
available frame condition.  Dropping the frame condition therefore recovers
the original global C classifier exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_efficient import EfficientSpanRoleClassifier


class UncertaintyAwareRoleClassifier(EfficientSpanRoleClassifier):
    def __init__(
        self,
        config,
        num_frames,
        frame_role_legal,
        frame_log_frequency,
        span_dim=256,
        role_dim=256,
        distance_dim=32,
        width_dim=16,
        legal_prior_scale_init=0.5,
    ):
        super().__init__(
            config,
            span_dim=span_dim,
            role_dim=role_dim,
            distance_dim=distance_dim,
            width_dim=width_dim,
        )
        legal = torch.as_tensor(frame_role_legal, dtype=torch.float32)
        frequencies = torch.as_tensor(frame_log_frequency, dtype=torch.float32)
        if legal.shape != (num_frames, self.num_labels):
            raise ValueError(
                f'frame_role_legal must be {(num_frames, self.num_labels)}, '
                f'got {tuple(legal.shape)}'
            )
        if frequencies.shape != (num_frames,):
            raise ValueError(
                f'frame_log_frequency must be {(num_frames,)}, '
                f'got {tuple(frequencies.shape)}'
            )
        self.num_frames = int(num_frames)
        self.register_buffer('frame_role_legal', legal, persistent=True)
        self.register_buffer('frame_log_frequency', frequencies, persistent=True)

        dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.frame_embeddings = nn.Parameter(torch.empty(num_frames, role_dim))
        nn.init.normal_(self.frame_embeddings, mean=0.0, std=0.02)
        self.local_interaction = nn.Sequential(
            nn.Linear(role_dim * 4, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
            nn.Dropout(dropout),
            nn.Linear(role_dim, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
        )
        self.local_role_embeddings = nn.Parameter(
            torch.empty(self.num_labels, role_dim)
        )
        self.local_role_bias = nn.Parameter(torch.zeros(self.num_labels))
        nn.init.normal_(self.local_role_embeddings, mean=0.0, std=0.02)

        # role_query + frame_repr + four scalar routing diagnostics.
        self.gate_network = nn.Sequential(
            nn.Linear(role_dim * 2 + 4, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
            nn.Linear(role_dim, 1),
        )
        # Softplus keeps the legal prior non-negative without a hard mask.
        initial = max(float(legal_prior_scale_init), 1e-4)
        self.legal_prior_raw_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial)))
        )

    def _condition_features(self, probabilities):
        probabilities = torch.as_tensor(
            probabilities,
            dtype=self.frame_embeddings.dtype,
            device=self.frame_embeddings.device,
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_frames:
            raise ValueError(
                f'frame_probabilities must be [batch,{self.num_frames}], '
                f'got {tuple(probabilities.shape)}'
            )
        if (probabilities < 0).any():
            raise ValueError('frame_probabilities cannot be negative.')
        observed_mass = probabilities.sum(dim=1, keepdim=True)
        if (observed_mass > 1.0001).any():
            raise ValueError('frame_probabilities must sum to at most one.')
        safe_mass = observed_mass.clamp_min(1e-8)
        normalized = probabilities / safe_mass
        entropy = -(normalized.clamp_min(1e-8).log() * normalized).sum(
            dim=1, keepdim=True
        ) / max(math.log(self.num_frames), 1e-8)
        entropy = torch.where(observed_mass > 0, entropy, torch.ones_like(entropy))
        max_probability = probabilities.max(dim=1, keepdim=True).values
        expected_frequency = torch.matmul(
            probabilities, self.frame_log_frequency[:, None]
        )
        frame_repr = torch.matmul(probabilities, self.frame_embeddings)

        # Omitted mass is an explicit global fallback.  Hence no role is ever
        # assigned -inf even when the frame distribution is wrong.
        fallback_mass = (1.0 - observed_mass).clamp_min(0.0)
        legal_mass = torch.matmul(probabilities, self.frame_role_legal)
        soft_legal_mass = (legal_mass + fallback_mass).clamp_min(1e-6)
        legal_log_prior = soft_legal_mass.log()
        diagnostics = torch.cat([
            observed_mass,
            max_probability,
            entropy,
            expected_frequency,
        ], dim=-1)
        return frame_repr, diagnostics, legal_log_prior, observed_mass

    def forward(
        self, input_ids, attention_mask, target, argument, frame_probabilities,
    ):
        base = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target,
            argument=argument,
        )
        query = base['role_query']
        frame_repr, diagnostics, legal_log_prior, observed_mass = (
            self._condition_features(frame_probabilities)
        )
        local_query = self.local_interaction(torch.cat([
            query,
            frame_repr,
            query * frame_repr,
            torch.abs(query - frame_repr),
        ], dim=-1))
        local_logits = F.linear(
            local_query, self.local_role_embeddings, self.local_role_bias
        )
        prior_scale = F.softplus(self.legal_prior_raw_scale)
        local_residual = local_logits + prior_scale * legal_log_prior
        gate = torch.sigmoid(self.gate_network(torch.cat([
            query, frame_repr, diagnostics,
        ], dim=-1)))
        # Exact global fallback when frame conditioning is dropped.
        gate = gate * observed_mass
        logits = base['logits'] + gate * local_residual
        return {
            **base,
            'global_logits': base['logits'],
            'local_logits': local_logits,
            'local_residual': local_residual,
            'dynamic_gate': gate,
            'legal_prior_scale': prior_scale,
            'frame_observed_mass': observed_mass,
            'logits': logits,
        }
