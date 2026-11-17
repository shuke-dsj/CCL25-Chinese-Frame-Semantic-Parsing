import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class FramePrototypeModel(nn.Module):
    """Target-aware context encoder scored against fixed frame prototypes."""

    def __init__(self, config, prototype_file):
        super().__init__()
        payload = torch.load(prototype_file, map_location='cpu')
        prototypes = payload['prototypes'].float()
        if prototypes.shape[0] != config.num_labels:
            raise ValueError(f'prototype count {prototypes.shape[0]} != num_labels {config.num_labels}')
        if prototypes.shape[1] != config.hidden_size:
            raise ValueError(f'prototype dim {prototypes.shape[1]} != hidden_size {config.hidden_size}')
        self.idx2label = payload['idx2label']
        self.bert = BertModel(config)
        h = config.hidden_size
        dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.context_projection = nn.Sequential(
            nn.LayerNorm(h * 4),
            nn.Dropout(dropout),
            nn.Linear(h * 4, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.register_buffer('frame_prototypes', F.normalize(prototypes, dim=-1))

    def encode_context(self, input_ids, attention_mask, target):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        target_tensor = torch.as_tensor(target, dtype=torch.long, device=hidden.device)
        batch = torch.arange(hidden.size(0), device=hidden.device)
        start = target_tensor[:, 0].clamp(0, hidden.size(1) - 1)
        end = torch.maximum(target_tensor[:, 1].clamp(0, hidden.size(1) - 1), start)
        start_vec = hidden[batch, start]
        end_vec = hidden[batch, end]
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        span_mask = (positions >= start.unsqueeze(1)) & (positions <= end.unsqueeze(1))
        span_vec = (hidden * span_mask.unsqueeze(-1)).sum(1)
        span_vec = span_vec / span_mask.sum(1, keepdim=True).clamp_min(1).to(hidden.dtype)
        cls_vec = hidden[:, 0]
        return F.normalize(self.context_projection(torch.cat([cls_vec, start_vec, end_vec, span_vec], dim=-1)), dim=-1)

    def forward(self, input_ids, attention_mask, target, labels=None,
                seen_label_ids=None, global_to_seen=None, for_test=False):
        context = self.encode_context(input_ids, attention_mask, target)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * context @ self.frame_prototypes.t()
        loss = None
        if labels is not None and not for_test:
            if seen_label_ids is None or global_to_seen is None:
                loss = F.cross_entropy(logits, labels)
            else:
                seen_logits = logits.index_select(1, seen_label_ids)
                seen_labels = global_to_seen.index_select(0, labels)
                if (seen_labels < 0).any():
                    raise ValueError('Training label is not in seen_label_ids.')
                loss = F.cross_entropy(seen_logits, seen_labels)
        return {'logits': logits, 'loss': loss, 'context': context}

