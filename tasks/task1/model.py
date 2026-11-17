import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = 64
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(768, 768 // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )


        self.dropout = nn.Dropout(classifier_dropout)
        self.model_variant = getattr(config, 'model_variant', 'rope')
        self.target_aware_layers = getattr(config, 'target_aware_layers', 'last')
        if self.model_variant == 'target_aware':
            self.lstm = None
            # O(B*L*H) target-conditioned classifier. It avoids producing
            # per-label q/k vectors for every token and directly pools the
            # target span plus its sentence context.
            self.dense = None
            self.target_classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size * 4),
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size * 4, config.num_labels),
            )
        else:
            self.dense = nn.Linear(config.hidden_size, config.num_labels * self.inner_dim * 2)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, target=None, labels=None, device=None, for_test=False):

        if self.model_variant == 'target_aware':
            use_layer_fusion = self.target_aware_layers == 'last4'
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=use_layer_fusion)
            if use_layer_fusion:
                hidden_token = (bert_out['hidden_states'][-4] + bert_out['hidden_states'][-3] +
                                bert_out['hidden_states'][-2] + bert_out['hidden_states'][-1]) / 4
            else:
                hidden_token = bert_out.last_hidden_state
            target_tensor = torch.as_tensor(target, dtype=torch.long, device=hidden_token.device)
            batch_idx = torch.arange(hidden_token.size(0), device=hidden_token.device)
            start = target_tensor[:, 0].clamp(0, hidden_token.size(1) - 1)
            end = target_tensor[:, 1].clamp(0, hidden_token.size(1) - 1)
            end = torch.maximum(end, start)
            start_vec = hidden_token[batch_idx, start]
            end_vec = hidden_token[batch_idx, end]
            # Mean-pool the inclusive target span without a Python loop.
            positions = torch.arange(hidden_token.size(1), device=hidden_token.device).unsqueeze(0)
            span_mask = (positions >= start.unsqueeze(1)) & (positions <= end.unsqueeze(1))
            span_vec = (hidden_token * span_mask.unsqueeze(-1)).sum(dim=1)
            span_vec = span_vec / span_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_token.dtype)
            cls_vec = hidden_token[:, 0]
            token_logits = self.target_classifier(torch.cat([cls_vec, start_vec, end_vec, span_vec], dim=-1))
            loss = None if for_test else nn.CrossEntropyLoss()(token_logits, labels)
            return {'logits': token_logits, 'loss': loss}

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # hidden_token = bert_out.last_hidden_state
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        outputs = self.dense(hidden_token)
        # lstm_out, (hidden, _) = self.lstm(hidden_token)
        # outputs = self.dense(lstm_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(hidden_token.shape[0], hidden_token.shape[1], 64, device)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # 优化：target 的 span 已知，无需计算完整的 [B,C,L,L] 矩阵，
        # 直接取 target 首尾两个位置的 q/k 做点积（与原 einsum+切片在数学上完全等价）。
        target_tensor = torch.tensor(target, dtype=torch.long, device=qw.device)
        batch_idx = torch.arange(qw.shape[0], device=qw.device)
        qw_start = qw[batch_idx, target_tensor[:, 0]]    # [B, C, inner_dim]：target 起始位置的 q
        kw_end = kw[batch_idx, target_tensor[:, 1]]      # [B, C, inner_dim]：target 结束位置的 k
        token_logits = (qw_start * kw_end).sum(dim=-1) / self.inner_dim ** 0.5   # [B, C]

        if for_test:
            loss = None
        else:
            loss_fc = torch.nn.CrossEntropyLoss()
            loss = loss_fc(token_logits, labels)
        return {
            "logits": token_logits,
            "loss": loss
        }





