#!/usr/bin/env python3
"""Small D4 dataset/forward/loss/backward compatibility test."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from .dataset import CandidateAwareDataset, collate_candidate_groups
from .model import CandidateAwareDefinitionReranker
from .train import objective, to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--pair_frames_per_role', default=2, type=int)
    parser.add_argument('--output_dir', default=str(
        REPO_ROOT / 'tasks/task3/analysis/task3-D4-smoke'
    ))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_file = REPO_ROOT / 'dataset/cfn-train.json'
    frame_file = REPO_ROOT / 'dataset/frame_info.json'
    model_dir = REPO_ROOT / 'chinese_bert_wwm_ext'
    rows = json.loads(train_file.read_text(encoding='utf8'))
    frame_info, role_names, role2idx, frame_to_roles, _ = (
        build_augmented_frame_roles(frame_file, train_file)
    )
    frame_names = [row['frame_name'] for row in frame_info]
    records = []
    for instance_index, row in enumerate(rows):
        for span in row.get('cfn_spans', []):
            records.append((instance_index, row, span))
            if len(records) == 2:
                break
        if len(records) == 2:
            break
    candidate_ids = []
    gold_masks = []
    all_roles = np.arange(len(role_names), dtype=np.int32)
    for group_index, (_, _, span) in enumerate(records):
        gold = int(role2idx[span['fe_name']])
        negatives = [int(value) for value in all_roles if int(value) != gold][:4]
        values = [gold] + negatives
        if group_index == 1:
            values[0], values[1] = values[1], values[0]
        candidate_ids.append(values)
        gold_masks.append([role == gold for role in values])
    frame_probs = np.full((2, len(frame_names)), 0.1 / (len(frame_names) - 1), np.float32)
    for index, (_, row, _) in enumerate(records):
        frame_probs[index, frame_names.index(row['frame'])] = 0.9
    cache_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'smoke_cache.npz'
    np.savez_compressed(
        cache_file,
        candidate_role_ids=np.asarray(candidate_ids, np.int32),
        candidate_generator_scores=np.asarray([[3, 2, 1, 0, -1], [3, 2, 1, 0, -1]], np.float32),
        frame_probs=frame_probs,
        instance_indices=np.asarray([row[0] for row in records], np.int32),
        sentence_ids=np.asarray([int(row[1]['sentence_id']) for row in records], np.int64),
        starts=np.asarray([int(row[2]['start']) for row in records], np.int32),
        ends=np.asarray([int(row[2]['end']) for row in records], np.int32),
        role_names=np.asarray(role_names),
        frame_names=np.asarray(frame_names),
        gold_candidate_mask=np.asarray(gold_masks, bool),
        boundary_match=np.ones(2, bool),
        prior_weights=np.ones(2, np.float32),
        gold_tuples=np.asarray([2], np.int64),
        gold_boundaries=np.asarray([2], np.int64),
    )
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    dataset = CandidateAwareDataset(
        train_file, cache_file, frame_info, frame_to_roles, tokenizer,
        role2idx, role_names, pair_frames_per_role=args.pair_frames_per_role,
        max_length=512, train_mode=True,
        generator_perturb_probability=1.0,
        generator_perturb_min_margin=0.1,
        generator_perturb_max_margin=0.2,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(
            collate_candidate_groups, pad_token_id=tokenizer.pad_token_id,
        ),
    )
    config = BertConfig.from_pretrained(model_dir)
    model = CandidateAwareDefinitionReranker(
        config, projection_dim=64,
        pair_frames_per_role=args.pair_frames_per_role,
    ).to(device)
    model.residual_scale_raw.data.fill_(0.1)
    batch = to_device(next(iter(loader)), device)
    outputs = model(
        input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
        token_type_ids=batch['token_type_ids'],
        marker_positions=batch['marker_positions'],
        generator_scores=batch['generator_scores'],
        frame_log_weights=batch['frame_log_weights'],
        group_features=batch['group_features'], pair_batch_size=8,
    )
    loss_args = argparse.Namespace(
        hard_example_extra_weight=2.0,
        definition_ce_weight=0.3,
        distill_weight=0.1,
        distill_temperature=1.0,
    )
    loss, final_ce, definition_ce, distill = objective(outputs, batch, loss_args)
    loss.backward()
    gradients = {
        'definition_projection': float(
            model.pair_encoder.projection[0].weight.grad.norm().cpu()
        ),
        # The final gate weight is intentionally zero-initialized, so on the
        # first backward pass its bias/weight learn before gradients can flow
        # into the first gate layer.
        'gate_output_bias': float(model.gate[-1].bias.grad.norm().cpu()),
        'residual_scale': float(model.residual_scale_raw.grad.abs().cpu()),
    }
    result = {
        'device': str(device),
        'groups': len(dataset),
        'pairs': int(len(batch['input_ids'])),
        'candidate_shape': list(outputs['final_scores'].shape),
        'pair_score_shape': list(outputs['pair_scores'].shape),
        'loss': float(loss.detach().cpu()),
        'final_ce': float(final_ce.detach().cpu()),
        'definition_ce': float(definition_ce.detach().cpu()),
        'distill': float(distill.detach().cpu()),
        'gate_mean': float(outputs['gate'].mean().detach().cpu()),
        'residual_scale': float(outputs['residual_scale'].detach().cpu()),
        'synthetic_hard_groups': int(batch['generator_perturbed'].sum().cpu()),
        'gradient_norms': gradients,
    }
    if not all(value > 0 for value in gradients.values()):
        raise AssertionError(f'D4 expected non-zero gradients, got {gradients}')
    if result['synthetic_hard_groups'] < 1:
        raise AssertionError('D4 synthetic generator corruption did not activate.')
    (cache_dir / 'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8',
    )
    print(json.dumps(result, ensure_ascii=False))
    print('[OK] D4 candidate cache/dataset/forward/loss/backward passed.')


if __name__ == '__main__':
    main()
