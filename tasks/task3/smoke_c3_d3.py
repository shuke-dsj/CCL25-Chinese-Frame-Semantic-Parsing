#!/usr/bin/env python3
"""Fast forward/backward checks for Task3 C3 and D3."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT, set_seed
from .dataset_experiments import (
    Task3RoleDataset,
    build_augmented_frame_roles,
    make_role_collate,
    train_frame_frequencies,
    train_role_frequencies,
)
from .definition_query.dataset import DefinitionQueryDataset, collate_groups
from .definition_query.model import DefinitionQueryCrossEncoder
from .definition_query.train import grouped_loss
from .model_c3 import UncertaintyAwareRoleClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    train_file = REPO_ROOT / 'dataset/cfn-train.json'
    frame_file = REPO_ROOT / 'dataset/frame_info.json'
    vocab_file = REPO_ROOT / 'chinese_bert_wwm_ext/vocab.txt'
    prototype_file = (
        REPO_ROOT / 'tasks/task3/definition_prototypes/frame_fe_definition_last4_mean.npz'
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    frame_info, role_names, label2idx, frame_to_roles, _ = (
        build_augmented_frame_roles(frame_file, train_file)
    )
    frame_to_id = {
        frame['frame_name']: index for index, frame in enumerate(frame_info)
    }
    role_frequency = train_role_frequencies(train_file)
    frame_frequency = train_frame_frequencies(train_file)

    tiny = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=192,
        hidden_dropout_prob=0.1,
        num_labels=len(role_names),
    )
    legal = np.zeros((len(frame_info), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_roles.items():
        legal[frame_to_id[frame], role_ids] = 1.0
    raw_freq = np.asarray([
        frame_frequency.get(frame['frame_name'], 0) for frame in frame_info
    ], dtype=np.float32)
    log_freq = np.log1p(raw_freq)
    log_freq /= max(float(log_freq.max()), 1.0)
    c3 = UncertaintyAwareRoleClassifier(
        tiny,
        num_frames=len(frame_info),
        frame_role_legal=legal,
        frame_log_frequency=log_freq,
        span_dim=32,
        role_dim=32,
        distance_dim=8,
        width_dim=4,
    ).to(device)
    c3_dataset = Task3RoleDataset(
        train_file, tokenizer, label2idx, role_frequency,
        train_frame_frequency=frame_frequency,
        max_examples=args.batch_size,
    )
    c3_collate = make_role_collate(
        frame_to_roles, len(role_names), tokenizer.pad_token_id, device,
        frame_to_id=frame_to_id,
    )
    c3_batch = next(iter(DataLoader(
        c3_dataset, batch_size=args.batch_size, collate_fn=c3_collate
    )))
    frame_probabilities = torch.zeros(
        (args.batch_size, len(frame_info)), device=device
    )
    frame_probabilities.scatter_(1, c3_batch['frame_ids'][:, None], 0.9)
    c3_output = c3(
        input_ids=c3_batch['input_ids'],
        attention_mask=c3_batch['attention_mask'],
        target=c3_batch['target'],
        argument=c3_batch['argument'],
        frame_probabilities=frame_probabilities,
    )
    c3_loss = torch.nn.functional.cross_entropy(
        c3_output['logits'], c3_batch['labels']
    )
    c3_loss.backward()
    c3.eval()
    with torch.no_grad():
        zero = torch.zeros_like(frame_probabilities)
        fallback = c3(
            input_ids=c3_batch['input_ids'],
            attention_mask=c3_batch['attention_mask'],
            target=c3_batch['target'],
            argument=c3_batch['argument'],
            frame_probabilities=zero,
        )
        fallback_difference = float(
            (fallback['logits'] - fallback['global_logits']).abs().max().cpu()
        )
    assert fallback_difference == 0.0, 'C3 frame dropout is not an exact global fallback.'

    d3_dataset = DefinitionQueryDataset(
        train_file,
        frame_info=frame_info,
        tokenizer=tokenizer,
        label2idx=label2idx,
        role_names=role_names,
        negative_count=2,
        hard_negative_pool=6,
        prototype_file=str(prototype_file),
        train_role_frequency=role_frequency,
        train_frame_frequency=frame_frequency,
        max_examples=args.batch_size,
    )
    d3_batch = next(iter(DataLoader(
        d3_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_groups, pad_token_id=tokenizer.pad_token_id),
    )))
    d3 = DefinitionQueryCrossEncoder(tiny, projection_dim=64).to(device)
    d3_scores = d3(
        input_ids=d3_batch['input_ids'].to(device),
        attention_mask=d3_batch['attention_mask'].to(device),
        token_type_ids=d3_batch['token_type_ids'].to(device),
        marker_positions={
            key: value.to(device)
            for key, value in d3_batch['marker_positions'].items()
        },
    )
    d3_loss, rank_loss, binary_loss = grouped_loss(
        d3_scores, d3_batch['group_sizes'], bce_weight=0.2
    )
    d3_loss.backward()
    print({
        'device': str(device),
        'C3': {
            'logits_shape': list(c3_output['logits'].shape),
            'loss': float(c3_loss.detach().cpu()),
            'gate_mean': float(c3_output['dynamic_gate'].mean().detach().cpu()),
            'global_fallback_max_difference': fallback_difference,
            'gate_gradient_norm': float(
                c3.gate_network[-1].weight.grad.norm().detach().cpu()
            ),
        },
        'D3': {
            'groups': d3_batch['group_sizes'],
            'flattened_pairs': len(d3_scores),
            'scores_shape': list(d3_scores.shape),
            'loss': float(d3_loss.detach().cpu()),
            'ranking_loss': float(rank_loss.detach().cpu()),
            'binary_loss': float(binary_loss.detach().cpu()),
            'score_head_gradient_norm': float(
                d3.projection[-1].weight.grad.norm().detach().cpu()
            ),
        },
    })
    print('[OK] Task3 C3/D3 forward, backward, grouping and global fallback passed.')


if __name__ == '__main__':
    main()
