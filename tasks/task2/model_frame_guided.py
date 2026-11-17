"""Frame/FE-definition guided multi-role GlobalPointer MVP."""
import math

import torch
import torch.nn as nn

from .decoding import build_safe_span_mask
from .model import Model as BaselineModel


class FrameGuidedGlobalPointer(BaselineModel):
    """Use contextualized FE slots as queries over the sentence span matrix.

    The baseline GlobalPointer score is retained. Each FE query adds a start
    and end compatibility term, producing ``[B, R, L, L]`` role-specific span
    scores. Unlike the single-answer pointer in AGED, this multi-label matrix
    preserves duplicate roles, nested spans, and overlapping spans.
    """

    def __init__(self, config):
        super().__init__(config)
        self.role_dim = int(getattr(config, 'task2_role_query_dim', 64))
        self.role_scale = nn.Parameter(torch.tensor(
            float(getattr(config, 'task2_role_scale_init', 0.1)), dtype=torch.float
        ))
        hidden_size = config.hidden_size
        self.role_start_query = nn.Linear(hidden_size, self.role_dim)
        self.role_end_query = nn.Linear(hidden_size, self.role_dim)
        self.role_start_token = nn.Linear(hidden_size, self.role_dim)
        self.role_end_token = nn.Linear(hidden_size, self.role_dim)
        self.role_dropout = nn.Dropout(config.hidden_dropout_prob)

    @staticmethod
    def pool_role_queries(hidden, role_spans, role_mask):
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, 1, -1)
        starts = role_spans[..., 0].unsqueeze(-1)
        ends = role_spans[..., 1].unsqueeze(-1)
        mask = (positions >= starts) & (positions <= ends) & role_mask.unsqueeze(-1)
        weights = mask.to(hidden.dtype)
        pooled = torch.einsum('brt,bth->brh', weights, hidden)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return pooled

    def baseline_span_logits(self, hidden_token, device):
        outputs = self.dense(hidden_token)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        pos_emb = self.sinusoidal_position_embedding(
            hidden_token.shape[0], hidden_token.shape[1], self.inner_dim, device
        )
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos
        return torch.einsum('bmhd,bnhd->bhmn', qw, kw).squeeze(1) / math.sqrt(self.inner_dim)

    @staticmethod
    def sentence_attention_mask(sentence_end, sentence_length):
        positions = torch.arange(sentence_length, device=sentence_end.device).unsqueeze(0)
        return (positions <= sentence_end.unsqueeze(1)).long()

    def compute_role_loss(self, role_logits, labels, role_mask, attention_mask, target):
        valid_span = build_safe_span_mask(attention_mask, target)
        valid = role_mask.unsqueeze(-1).unsqueeze(-1) & valid_span.unsqueeze(1)
        valid_float = valid.to(role_logits.dtype)
        loss_positive = torch.sum(
            torch.exp(-role_logits) * valid_float * labels, dim=(1, 2, 3)
        )
        loss_negative = torch.sum(
            torch.exp(role_logits) * valid_float * (1.0 - labels), dim=(1, 2, 3)
        )
        return torch.mean(torch.log1p(loss_positive) + torch.log1p(loss_negative))

    def forward(self, input_ids, attention_mask, target, sentence_end, role_spans,
                role_mask, labels=None, device=None, for_test=False):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = bert_output.last_hidden_state
        sentence_length = labels.shape[-1] if labels is not None else int(sentence_end.max().item()) + 1
        sentence_hidden = hidden[:, :sentence_length]

        baseline_logits = self.baseline_span_logits(sentence_hidden, device or hidden.device)
        role_query = self.role_dropout(self.pool_role_queries(hidden, role_spans, role_mask))
        start_query = self.role_start_query(role_query)
        end_query = self.role_end_query(role_query)
        start_token = self.role_start_token(sentence_hidden)
        end_token = self.role_end_token(sentence_hidden)
        start_score = torch.einsum('brd,bld->brl', start_query, start_token) / math.sqrt(self.role_dim)
        end_score = torch.einsum('brd,bld->brl', end_query, end_token) / math.sqrt(self.role_dim)
        role_residual = start_score.unsqueeze(-1) + end_score.unsqueeze(-2)
        role_logits = baseline_logits.unsqueeze(1) + self.role_scale * role_residual

        masked_role_logits = role_logits.masked_fill(
            ~role_mask.unsqueeze(-1).unsqueeze(-1), torch.finfo(role_logits.dtype).min
        )
        task2_logits = masked_role_logits.max(dim=1).values
        sentence_attention = self.sentence_attention_mask(sentence_end, sentence_length)
        loss = None
        if not for_test:
            loss = self.compute_role_loss(
                role_logits, labels, role_mask, sentence_attention, target
            )
        return {
            'logits': task2_logits,
            'role_logits': role_logits,
            'loss': loss,
            'role_scale': self.role_scale,
        }
