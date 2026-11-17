#!/usr/bin/env python3
"""Search frequency-aware soft Frame->Role routing on cached pipeline logits.

This is a decoding experiment: it never uses Dev gold frame/role information
to construct a prediction.  Frame frequency comes only from Train and routing
confidence comes only from Task1 probabilities.  Gold tuples are read after
prediction solely for model selection and reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.analyze_pipeline_ensemble import (
    build_gold_mask,
    evaluate,
    percentile_confidence,
    zscore_rows,
)
from tasks.task3.dataset_experiments import build_augmented_frame_roles


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'
BUCKETS = ('unseen_0', 'rare_1_5', 'few_6_15', 'medium_16_50', 'head_gt_50')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--c2_cache',
        default=str(TASK_DIR / 'analysis' / 'task3-full-pipeline-B-C2-frame' /
                    'pipeline_logits.npz'),
    )
    parser.add_argument(
        '--d_cache',
        default=str(TASK_DIR / 'analysis' / 'task3-full-pipeline-B-D-soft-top5' /
                    'pipeline_logits.npz'),
    )
    parser.add_argument('--b_weight', default=0.25, type=float)
    parser.add_argument('--c2_weight', default=0.10, type=float)
    parser.add_argument('--d_weight', default=0.65, type=float)
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset/cfn-dev.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument('--prior_lows', default='0.2,0.4,0.6,0.8,1.0')
    parser.add_argument('--prior_highs', default='0.8,1.0,1.2,1.4,1.6,1.8')
    parser.add_argument('--powers', default='0.5,1,2')
    parser.add_argument(
        '--output_dir',
        default=str(TASK_DIR / 'analysis' / 'task3-frequency-aware-routing'),
    )
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def frame_bucket(frequency):
    if frequency == 0:
        return 'unseen_0'
    if frequency <= 5:
        return 'rare_1_5'
    if frequency <= 15:
        return 'few_6_15'
    if frequency <= 50:
        return 'medium_16_50'
    return 'head_gt_50'


def parse_float_list(text):
    return [float(value) for value in text.split(',') if value.strip()]


def validate_cache_alignment(first, second):
    for field in ('instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names'):
        if not np.array_equal(first[field], second[field]):
            raise ValueError(f'C2/D pipeline caches differ in {field}.')
    if not np.allclose(first['frame_probs'], second['frame_probs'], atol=1e-7):
        raise ValueError('C2/D pipeline caches contain different Task1 probabilities.')


def selected_frame_probabilities(probabilities, policy, bucket_ids, confidence):
    """Return unnormalised selected probabilities; omitted mass is fallback."""
    batch, num_frames = probabilities.shape
    if policy['kind'] == 'all':
        return probabilities
    selected = np.zeros_like(probabilities, dtype=np.float32)
    if policy['kind'] == 'bucket_k':
        k_values = np.asarray(policy['k_values'], dtype=np.int32)[bucket_ids]
        # Low confidence adds up to two frames without changing the frequency
        # profile.  The maximum remains small enough for efficient routing.
        k_values = np.minimum(
            k_values + (confidence < 0.50).astype(np.int32)
            + (confidence < 0.20).astype(np.int32),
            num_frames,
        )
        max_k = int(k_values.max())
        top = np.argpartition(probabilities, -max_k, axis=1)[:, -max_k:]
        order = np.take_along_axis(probabilities, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for row in range(batch):
            indices = top[row, :k_values[row]]
            selected[row, indices] = probabilities[row, indices]
        return selected
    if policy['kind'] == 'cumulative':
        threshold = float(policy['threshold'])
        max_k = int(policy['max_k'])
        top = np.argpartition(probabilities, -max_k, axis=1)[:, -max_k:]
        values = np.take_along_axis(probabilities, top, axis=1)
        order = values.argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        cumulative = np.cumsum(values, axis=1)
        counts = np.maximum((cumulative < threshold).sum(axis=1) + 1, 1)
        for row in range(batch):
            indices = top[row, :counts[row]]
            selected[row, indices] = probabilities[row, indices]
        return selected
    raise ValueError(policy)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c2_cache = np.load(args.c2_cache, allow_pickle=False)
    d_cache = np.load(args.d_cache, allow_pickle=False)
    validate_cache_alignment(c2_cache, d_cache)

    role_names = c2_cache['role_names'].tolist()
    b_logits = c2_cache['b_logits'].astype(np.float32)
    c2_logits = c2_cache['c_logits'].astype(np.float32)
    d_logits = d_cache['c_logits'].astype(np.float32)
    weights = np.asarray([args.b_weight, args.c2_weight, args.d_weight], np.float32)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f'B/C2/D weights must be non-negative and sum to 1, got {weights}.')
    combined = (
        args.b_weight * zscore_rows(b_logits)
        + args.c2_weight * zscore_rows(c2_logits)
        + args.d_weight * zscore_rows(d_logits)
    )
    frame_probs = c2_cache['frame_probs'].astype(np.float32)

    frame_info, ontology_roles, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    if role_names != ontology_roles:
        raise ValueError('Pipeline role order differs from frame ontology.')
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0

    train_rows = read_json(args.train_file)
    frame_frequency = Counter(row['frame'] for row in train_rows)
    frame_bucket_ids = np.asarray(
        [BUCKETS.index(frame_bucket(frame_frequency[name])) for name in frame_names],
        dtype=np.int32,
    )
    top_frame = frame_probs.argmax(axis=1)
    sample_bucket_ids = frame_bucket_ids[top_frame]
    top2 = np.sort(np.partition(frame_probs, -2, axis=1)[:, -2:], axis=1)
    confidence = percentile_confidence(top2[:, 1] - top2[:, 0])

    dev_rows = read_json(args.dev_file)
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        dev_rows, role2idx,
        c2_cache['instance_indices'].astype(np.int64),
        c2_cache['starts'].astype(np.int64),
        c2_cache['ends'].astype(np.int64),
        len(role_names),
    )

    selection_policies = [
        {'name': 'all_frames', 'kind': 'all'},
        {'name': 'bucket_k_mild', 'kind': 'bucket_k', 'k_values': [10, 5, 3, 2, 1]},
        {'name': 'bucket_k_wide', 'kind': 'bucket_k', 'k_values': [20, 10, 5, 3, 2]},
        {'name': 'cumulative_090', 'kind': 'cumulative', 'threshold': 0.90, 'max_k': 20},
        {'name': 'cumulative_095', 'kind': 'cumulative', 'threshold': 0.95, 'max_k': 30},
    ]
    frequency_profiles = {
        'none': np.asarray([1, 1, 1, 1, 1], np.float32),
        'mild': np.asarray([0.35, 0.55, 0.75, 0.90, 1.0], np.float32),
        'strong': np.asarray([0.15, 0.35, 0.60, 0.85, 1.0], np.float32),
    }
    lows = parse_float_list(args.prior_lows)
    highs = parse_float_list(args.prior_highs)
    powers = parse_float_list(args.powers)

    rows = []
    best = None
    for policy in selection_policies:
        selected_probs = selected_frame_probabilities(
            frame_probs, policy, sample_bucket_ids, confidence
        )
        role_prior = np.clip(selected_probs @ membership, args.prior_epsilon, 1.0)
        log_prior = np.log(role_prior).astype(np.float32)
        selected_mass = selected_probs.sum(axis=1)
        selected_frames = (selected_probs > 0).sum(axis=1)
        for frequency_name, factors in frequency_profiles.items():
            sample_frequency_factor = factors[sample_bucket_ids]
            for low in lows:
                for high in highs:
                    if high < low:
                        continue
                    for power in powers:
                        prior_weight = (
                            low + (high - low) * np.power(confidence, power)
                        ) * sample_frequency_factor
                        prediction = (
                            combined + prior_weight[:, None] * log_prior
                        ).argmax(axis=1)
                        metric = evaluate(
                            prediction, gold_mask, boundary_match,
                            gold_tuples, gold_boundaries,
                        )
                        row = {
                            'selection_policy': policy['name'],
                            'frequency_profile': frequency_name,
                            'prior_low': low, 'prior_high': high, 'power': power,
                            'mean_prior_weight': float(prior_weight.mean()),
                            'mean_selected_frames': float(selected_frames.mean()),
                            'mean_selected_probability_mass': float(selected_mass.mean()),
                            **metric,
                        }
                        rows.append(row)
                        if best is None or (
                            row['exact_f1'], row['exact_recall'], row['exact_precision']
                        ) > (
                            best['exact_f1'], best['exact_recall'], best['exact_precision']
                        ):
                            best = row

    with (output_dir / 'grid.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'best_frequency_aware_routing': best,
        'reference_previous_B_C2_D': {
            'exact_tp': 3798,
            'exact_f1': 0.6084588272989426,
            'weights': {'B': 0.25, 'C2': 0.10, 'D': 0.65},
            'fixed_prior': 0.9,
        },
        'num_grid_rows': len(rows),
        'num_predicted_spans': len(combined),
        'bucket_order': list(BUCKETS),
        'frequency_profiles': {
            name: values.tolist() for name, values in frequency_profiles.items()
        },
        'selection_policies': selection_policies,
        'guarantee': (
            'All 1009 roles keep finite logits via epsilon; routing never hard-removes a role.'
        ),
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
