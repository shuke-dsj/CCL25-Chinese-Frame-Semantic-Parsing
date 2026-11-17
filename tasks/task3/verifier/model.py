"""Frame-Conditioned Span–Role Verifier (FCSR)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class FrameConditionedSpanRoleVerifier(nn.Module):
    def __init__(
        self,
        config,
        num_roles,
        num_frames,
        variant="marker_verifier",
        projection_dim=512,
        frame_dim=128,
        frame_condition_dropout=0.2,
    ):
        super().__init__()
        if variant not in {"marker_verifier", "frame_conditioned_verifier"}:
            raise ValueError(f"Unknown verifier variant: {variant}")
        self.variant = variant
        self.num_roles = int(num_roles)
        self.num_frames = int(num_frames)
        self.frame_condition_dropout = float(frame_condition_dropout)
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # CLS, target, argument, |target-argument| and target*argument.
        self.span_projection = nn.Sequential(
            nn.Linear(hidden * 5, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(config.hidden_dropout_prob),
        )
        if variant == "frame_conditioned_verifier":
            self.frame_embeddings = nn.Embedding(num_frames, frame_dim)
            self.frame_projection = nn.Linear(frame_dim, projection_dim)
            self.frame_gate = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim), nn.Sigmoid()
            )
            self.condition_norm = nn.LayerNorm(projection_dim)
        self.validity_head = nn.Linear(projection_dim, 1)
        self.role_head = nn.Linear(projection_dim, num_roles)

    @staticmethod
    def _gather(hidden, positions):
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, positions]

    def _frame_vector(self, query, frame_ids, frame_probabilities):
        if frame_probabilities is None:
            frame = self.frame_embeddings(frame_ids)
        else:
            frame = frame_probabilities @ self.frame_embeddings.weight
        frame = self.frame_projection(frame)
        if self.training and self.frame_condition_dropout > 0:
            keep = (
                torch.rand((len(frame), 1), device=frame.device)
                >= self.frame_condition_dropout
            ).to(frame.dtype)
            frame = frame * keep
        gate = self.frame_gate(torch.cat([query, frame], dim=-1))
        return self.condition_norm(query + gate * frame), gate.mean()

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_positions,
        frame_ids,
        frame_probabilities=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = sum(outputs.hidden_states[-4:]) / 4.0
        cls = hidden[:, 0]
        target = 0.5 * (
            self._gather(hidden, marker_positions["target_start"])
            + self._gather(hidden, marker_positions["target_end"])
        )
        argument = 0.5 * (
            self._gather(hidden, marker_positions["argument_start"])
            + self._gather(hidden, marker_positions["argument_end"])
        )
        query = self.span_projection(
            torch.cat(
                [cls, target, argument, torch.abs(target - argument), target * argument],
                dim=-1,
            )
        )
        frame_gate = query.new_zeros(())
        if self.variant == "frame_conditioned_verifier":
            query, frame_gate = self._frame_vector(
                query, frame_ids, frame_probabilities
            )
        query = self.dropout(query)
        return {
            "validity_logits": self.validity_head(query).squeeze(-1),
            "role_logits": self.role_head(query),
            "frame_gate": frame_gate,
        }
