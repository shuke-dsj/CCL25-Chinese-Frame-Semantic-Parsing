"""Frame--FE definition prototype classifier for Task3 Experiment D.

The learned classifier from Experiment C remains intact.  A second semantic
branch compares each predicate--argument representation with frozen textual
prototypes built from ``frame_def`` and ``fe_def``.  The final score is a
residual combination, so D can be warm-started from C without discarding the
strong supervised role embeddings.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_efficient import EfficientSpanRoleClassifier


class DefinitionPrototypeRoleClassifier(EfficientSpanRoleClassifier):
    def __init__(
        self,
        config,
        prototype_file,
        span_dim=256,
        role_dim=256,
        distance_dim=32,
        width_dim=16,
        definition_scale_init=0.1,
        definition_temperature_init=10.0,
        definition_aggregation='residual',
        freeze_definition_temperature=False,
    ):
        super().__init__(
            config,
            span_dim=span_dim,
            role_dim=role_dim,
            distance_dim=distance_dim,
            width_dim=width_dim,
        )
        bank = np.load(prototype_file, allow_pickle=False)
        global_prototypes = np.asarray(bank['global_role_prototypes'], dtype=np.float32)
        pair_prototypes = np.asarray(bank['pair_prototypes'], dtype=np.float32)
        frame_pair_indices = np.asarray(bank['frame_pair_indices'], dtype=np.int64)
        frame_role_ids = np.asarray(bank['frame_role_ids'], dtype=np.int64)
        if global_prototypes.shape[0] != self.num_labels:
            raise ValueError(
                f'Prototype role count {global_prototypes.shape[0]} does not match '
                f'num_labels={self.num_labels}'
            )
        if frame_pair_indices.shape != frame_role_ids.shape:
            raise ValueError('frame_pair_indices and frame_role_ids must have identical shapes.')
        if pair_prototypes.shape[1] != global_prototypes.shape[1]:
            raise ValueError('Pair/global definition prototype dimensions do not match.')

        definition_dim = int(global_prototypes.shape[1])
        self.definition_projection = nn.Sequential(
            nn.Linear(definition_dim, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
            nn.Linear(role_dim, role_dim),
        )
        self.definition_query_projection = nn.Sequential(
            nn.Linear(role_dim, role_dim),
            nn.GELU(),
            nn.LayerNorm(role_dim),
        )
        # Buffers are reconstructed from the NPZ and intentionally excluded
        # from checkpoints.  This keeps D checkpoints close to C in size.
        self.register_buffer(
            'global_definition_prototypes', torch.from_numpy(global_prototypes),
            persistent=False,
        )
        self.register_buffer(
            'pair_definition_prototypes', torch.from_numpy(pair_prototypes),
            persistent=False,
        )
        self.register_buffer(
            'frame_pair_indices', torch.from_numpy(frame_pair_indices),
            persistent=False,
        )
        self.register_buffer(
            'frame_role_ids', torch.from_numpy(frame_role_ids),
            persistent=False,
        )
        self.definition_gate = nn.Parameter(torch.tensor(float(definition_scale_init)))
        self.definition_log_temperature = nn.Parameter(
            torch.tensor(math.log(float(definition_temperature_init)))
        )
        if definition_aggregation not in ('residual', 'pair_logsumexp'):
            raise ValueError(
                'definition_aggregation must be residual or pair_logsumexp, '
                f'got {definition_aggregation!r}'
            )
        self.definition_aggregation = definition_aggregation
        if freeze_definition_temperature:
            self.definition_log_temperature.requires_grad_(False)

    def _definition_base(self, role_query):
        query = F.normalize(self.definition_query_projection(role_query), dim=-1)
        global_prototypes = F.normalize(
            self.definition_projection(self.global_definition_prototypes), dim=-1
        )
        temperature = self.definition_log_temperature.exp().clamp(1.0, 50.0)
        global_logits = temperature * torch.matmul(query, global_prototypes.transpose(0, 1))
        return query, global_logits, temperature

    def _local_scores(self, query, frame_ids, temperature):
        """Return frame-specific role ids, scores and valid positions."""
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=query.device)
        pair_indices = self.frame_pair_indices.index_select(0, frame_ids)
        role_ids = self.frame_role_ids.index_select(0, frame_ids)
        valid = (pair_indices >= 0) & (role_ids >= 0)
        safe_pair_indices = pair_indices.clamp_min(0)
        local_raw = self.pair_definition_prototypes[safe_pair_indices]
        local_prototypes = F.normalize(self.definition_projection(local_raw), dim=-1)
        local_logits = temperature * torch.einsum('bd,bkd->bk', query, local_prototypes)
        return role_ids, local_logits, valid

    def _definition_logits(self, role_query, frame_ids):
        query, global_logits, temperature = self._definition_base(role_query)

        role_ids, local_logits, valid = self._local_scores(
            query, frame_ids, temperature
        )

        # Start with global role definitions and replace legal entries with
        # frame-specific Frame--FE definitions.  Augmented train licenses that
        # have no frame-specific ontology entry safely retain the global score.
        definition_logits = global_logits.clone()
        for batch_index in range(definition_logits.shape[0]):
            current = valid[batch_index]
            if current.any():
                definition_logits[batch_index, role_ids[batch_index, current]] = (
                    local_logits[batch_index, current]
                )
        return definition_logits

    def _definition_logits_soft(self, role_query, frame_probabilities, topk=5):
        """Marginalise local Frame--FE residuals under Task1 probabilities.

        Frames outside top-k retain the global FE definition score.  Top-k
        probabilities are deliberately not renormalised: omitted probability
        mass therefore means "use the global prototype", which is safer when
        Task1 is uncertain.
        """
        query, global_logits, temperature = self._definition_base(role_query)
        probabilities = torch.as_tensor(
            frame_probabilities, dtype=query.dtype, device=query.device
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != self.frame_pair_indices.shape[0]:
            raise ValueError(
                f'frame_probabilities must be [batch,{self.frame_pair_indices.shape[0]}], '
                f'got {tuple(probabilities.shape)}'
            )
        k = min(max(int(topk), 1), probabilities.shape[1])
        top_probabilities, top_frames = torch.topk(probabilities, k=k, dim=1)
        definition_logits = global_logits.clone()
        for rank in range(k):
            role_ids, local_logits, valid = self._local_scores(
                query, top_frames[:, rank], temperature
            )
            weight = top_probabilities[:, rank]
            for batch_index in range(definition_logits.shape[0]):
                current = valid[batch_index]
                if current.any():
                    current_roles = role_ids[batch_index, current]
                    residual = (
                        local_logits[batch_index, current]
                        - global_logits[batch_index, current_roles]
                    )
                    definition_logits[batch_index, current_roles] += (
                        weight[batch_index] * residual
                    )
        return definition_logits

    def _definition_logits_pair_marginal(
        self, role_query, frame_probabilities, topk=5, probability_epsilon=1e-8,
    ):
        """Marginalise frame-specific ``(frame, FE)`` definition scores.

        For each role ``r`` this computes the stable log-mixture

        ``logsumexp(log p(f) + score(span, Definition(f,r)))``.

        Only the most probable frames are expanded.  Their omitted probability
        mass is assigned to the global FE definition, so every one of the 1009
        roles retains a finite score even when Task1 is wrong or uncertain.
        Repeated global FE names in different frames are merged with
        ``logaddexp`` rather than overwritten.
        """
        query, global_logits, temperature = self._definition_base(role_query)
        probabilities = torch.as_tensor(
            frame_probabilities, dtype=query.dtype, device=query.device
        )
        expected_frames = self.frame_pair_indices.shape[0]
        if probabilities.ndim != 2 or probabilities.shape[1] != expected_frames:
            raise ValueError(
                f'frame_probabilities must be [batch,{expected_frames}], '
                f'got {tuple(probabilities.shape)}'
            )
        if (probabilities < 0).any():
            raise ValueError('frame_probabilities cannot contain negative values.')
        row_mass = probabilities.sum(dim=1)
        if (row_mass > 1.0001).any():
            raise ValueError('frame_probabilities must sum to at most one per row.')

        k = min(max(int(topk), 1), probabilities.shape[1])
        top_probabilities, top_frames = torch.topk(probabilities, k=k, dim=1)
        selected_mass = top_probabilities.sum(dim=1)
        fallback_mass = (1.0 - selected_mass).clamp_min(float(probability_epsilon))
        marginal_logits = global_logits + fallback_mass.log().unsqueeze(1)

        for rank in range(k):
            role_ids, local_logits, valid = self._local_scores(
                query, top_frames[:, rank], temperature
            )
            log_probability = top_probabilities[:, rank].clamp_min(
                float(probability_epsilon)
            ).log()
            for batch_index in range(marginal_logits.shape[0]):
                current = valid[batch_index]
                # A zero-probability frame must not receive epsilon mixture
                # mass; it is only present because torch.topk always returns k.
                if current.any() and top_probabilities[batch_index, rank] > 0:
                    current_roles = role_ids[batch_index, current]
                    pair_terms = (
                        local_logits[batch_index, current]
                        + log_probability[batch_index]
                    )
                    marginal_logits[batch_index, current_roles] = torch.logaddexp(
                        marginal_logits[batch_index, current_roles], pair_terms
                    )
        return marginal_logits

    def forward(
        self, input_ids, attention_mask, target, argument,
        frame_ids=None, frame_probabilities=None, frame_topk=5,
    ):
        base = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target,
            argument=argument,
        )
        if (frame_ids is None) == (frame_probabilities is None):
            raise ValueError('Provide exactly one of frame_ids or frame_probabilities.')
        if self.definition_aggregation == 'pair_logsumexp':
            if frame_probabilities is None:
                probabilities = torch.zeros(
                    (len(frame_ids), self.frame_pair_indices.shape[0]),
                    dtype=base['role_query'].dtype,
                    device=base['role_query'].device,
                )
                probabilities.scatter_(1, torch.as_tensor(
                    frame_ids, dtype=torch.long, device=probabilities.device
                )[:, None], 1.0)
            else:
                probabilities = frame_probabilities
            definition_logits = self._definition_logits_pair_marginal(
                base['role_query'], probabilities, topk=frame_topk
            )
        elif frame_probabilities is not None:
            definition_logits = self._definition_logits_soft(
                base['role_query'], frame_probabilities, topk=frame_topk
            )
        else:
            definition_logits = self._definition_logits(base['role_query'], frame_ids)
        logits = base['logits'] + self.definition_gate * definition_logits
        return {
            **base,
            'learned_logits': base['logits'],
            'prototype_logits': definition_logits,
            'logits': logits,
            'definition_gate': self.definition_gate,
            'definition_temperature': self.definition_log_temperature.exp(),
        }
