#!/usr/bin/env python3
"""Search normalized B/C ensemble and confidence-adaptive frame prior."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks.common.experiment_utils import REPO_ROOT
from .dataset_experiments import build_augmented_frame_roles


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pipeline_cache',
        default=str(TASK_DIR / 'analysis' / 'task3-full-pipeline-BC' /
                    'pipeline_logits.npz'),
    )
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset' / 'cfn-dev.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset' / 'cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset' / 'frame_info.json'))
    parser.add_argument('--weight_step', default=0.05, type=float)
    parser.add_argument('--fixed_prior_min', default=0.0, type=float)
    parser.add_argument('--fixed_prior_max', default=2.0, type=float)
    parser.add_argument('--fixed_prior_step', default=0.1, type=float)
    parser.add_argument('--adaptive_low_values', default='0,0.2,0.4,0.6,0.8,1.0')
    parser.add_argument('--adaptive_high_values', default='0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0')
    parser.add_argument('--adaptive_powers', default='0.5,1,2')
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument(
        '--output_dir',
        default=str(TASK_DIR / 'analysis' /
                    'task3-full-pipeline-BC-normalized-adaptive'),
    )
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def float_grid(start, stop, step):
    return np.round(np.arange(start, stop + step / 2.0, step), 10)


def zscore_rows(logits):
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / centered.std(axis=1, keepdims=True).clip(min=1e-6)


def percentile_confidence(values):
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.ones_like(values, dtype=np.float32)
    return (
        np.searchsorted(unique, values, side='right').astype(np.float32) - 1.0
    ) / float(len(unique) - 1)


def build_gold_mask(dev_rows, role2idx, instance_indices, starts, ends, num_roles):
    gold_by_boundary = {}
    gold_tuples = 0
    gold_boundaries = 0
    for instance_index, row in enumerate(dev_rows):
        for span in row.get('cfn_spans', []):
            boundary = (instance_index, int(span['start']), int(span['end']))
            if boundary not in gold_by_boundary:
                gold_by_boundary[boundary] = set()
                gold_boundaries += 1
            role_id = int(role2idx[span['fe_name']])
            if role_id not in gold_by_boundary[boundary]:
                gold_by_boundary[boundary].add(role_id)
                gold_tuples += 1
    mask = np.zeros((len(instance_indices), num_roles), dtype=bool)
    boundary_match = np.zeros(len(instance_indices), dtype=bool)
    for index, key in enumerate(zip(instance_indices, starts, ends)):
        roles = gold_by_boundary.get(tuple(int(x) for x in key), ())
        if roles:
            boundary_match[index] = True
            mask[index, list(roles)] = True
    return mask, boundary_match, gold_tuples, gold_boundaries


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(prediction, gold_role_mask, boundary_match, gold_tuples, gold_boundaries):
    indices = np.arange(len(prediction))
    tp = int(gold_role_mask[indices, prediction].sum())
    precision, recall, f1 = prf(tp, len(prediction), gold_tuples)
    boundary_tp = int(boundary_match.sum())
    bp, br, bf = prf(boundary_tp, len(prediction), gold_boundaries)
    return {
        'exact_precision': precision, 'exact_recall': recall, 'exact_f1': f1,
        'exact_tp': tp, 'predicted_tuples': int(len(prediction)),
        'gold_tuples': int(gold_tuples),
        'boundary_precision': bp, 'boundary_recall': br, 'boundary_f1': bf,
        'boundary_tp': boundary_tp, 'gold_boundaries': int(gold_boundaries),
        'role_acc_on_matched_boundaries': tp / boundary_tp if boundary_tp else 0.0,
    }


def is_better(row, best):
    return best is None or (
        row['exact_f1'], row['exact_recall'], row['exact_precision']
    ) > (
        best['exact_f1'], best['exact_recall'], best['exact_precision']
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.pipeline_cache)
    required = {
        'b_logits', 'c_logits', 'frame_probs', 'instance_indices',
        'sentence_ids', 'starts', 'ends', 'role_names',
    }
    missing = required.difference(cache.files)
    if missing:
        raise ValueError(f'Pipeline cache missing fields: {sorted(missing)}')
    b_logits = cache['b_logits'].astype(np.float32)
    c_logits = cache['c_logits'].astype(np.float32)
    if b_logits.shape != c_logits.shape:
        raise ValueError(f'B/C logits mismatch: {b_logits.shape} vs {c_logits.shape}')
    frame_probs = cache['frame_probs'].astype(np.float32)
    role_names = cache['role_names'].tolist()
    instance_indices = cache['instance_indices'].astype(np.int64)
    starts = cache['starts'].astype(np.int64)
    ends = cache['ends'].astype(np.int64)

    frame_info, ontology_roles, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    if role_names != ontology_roles:
        raise ValueError('Pipeline cache role order differs from frame ontology.')
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0
    legal_prior = np.clip(frame_probs @ membership, args.prior_epsilon, 1.0)
    log_prior = np.log(legal_prior).astype(np.float32)

    dev_rows = read_json(args.dev_file)
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        dev_rows, role2idx, instance_indices, starts, ends, len(role_names)
    )
    b_norm, c_norm = zscore_rows(b_logits), zscore_rows(c_logits)

    top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    entropy = -(frame_probs * np.log(frame_probs.clip(min=1e-12))).sum(axis=1)
    confidence_features = {
        'margin_percentile': percentile_confidence(margin),
        'negative_entropy_percentile': percentile_confidence(-entropy),
    }

    grid_rows = []
    best_fixed = None
    best_fixed_by_weight = []
    weights = float_grid(0.0, 1.0, args.weight_step)
    fixed_priors = float_grid(
        args.fixed_prior_min, args.fixed_prior_max, args.fixed_prior_step
    )
    for b_weight in weights:
        combined = b_weight * b_norm + (1.0 - b_weight) * c_norm
        best_for_weight = None
        for prior_weight in fixed_priors:
            prediction = (combined + prior_weight * log_prior).argmax(axis=1)
            row = {
                'strategy': 'fixed_prior', 'confidence': '',
                'b_weight': float(b_weight), 'c_weight': float(1.0 - b_weight),
                'prior_low': float(prior_weight), 'prior_high': float(prior_weight),
                'power': 0.0,
                **evaluate(
                    prediction, gold_mask, boundary_match,
                    gold_tuples, gold_boundaries,
                ),
            }
            grid_rows.append(row)
            if is_better(row, best_fixed):
                best_fixed = row
            if is_better(row, best_for_weight):
                best_for_weight = row
        best_fixed_by_weight.append(best_for_weight)

    # Limit the adaptive search to the three best fixed-prior ensemble weights.
    candidate_weight_rows = sorted(
        best_fixed_by_weight, key=lambda row: row['exact_f1'], reverse=True
    )[:3]
    lows = [float(value) for value in args.adaptive_low_values.split(',') if value]
    highs = [float(value) for value in args.adaptive_high_values.split(',') if value]
    powers = [float(value) for value in args.adaptive_powers.split(',') if value]
    best_adaptive = None
    for fixed_reference in candidate_weight_rows:
        b_weight = fixed_reference['b_weight']
        combined = b_weight * b_norm + (1.0 - b_weight) * c_norm
        for confidence_name, confidence in confidence_features.items():
            for low in lows:
                for high in highs:
                    if high < low:
                        continue
                    for power in powers:
                        sample_weight = low + (high - low) * np.power(confidence, power)
                        prediction = (
                            combined + sample_weight[:, None] * log_prior
                        ).argmax(axis=1)
                        row = {
                            'strategy': 'adaptive_prior',
                            'confidence': confidence_name,
                            'b_weight': float(b_weight),
                            'c_weight': float(1.0 - b_weight),
                            'prior_low': low, 'prior_high': high, 'power': power,
                            **evaluate(
                                prediction, gold_mask, boundary_match,
                                gold_tuples, gold_boundaries,
                            ),
                        }
                        grid_rows.append(row)
                        if is_better(row, best_adaptive):
                            best_adaptive = row

    best_overall = best_adaptive if is_better(best_adaptive, best_fixed) else best_fixed
    with (output_dir / 'grid.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)

    b_weight = best_overall['b_weight']
    combined = b_weight * b_norm + (1.0 - b_weight) * c_norm
    if best_overall['strategy'] == 'adaptive_prior':
        confidence = confidence_features[best_overall['confidence']]
        sample_weight = best_overall['prior_low'] + (
            best_overall['prior_high'] - best_overall['prior_low']
        ) * np.power(confidence, best_overall['power'])
    else:
        sample_weight = np.full(len(combined), best_overall['prior_low'], np.float32)
    prediction = (combined + sample_weight[:, None] * log_prior).argmax(axis=1)
    prediction_rows = [
        {
            'instance_index': int(instance_indices[index]),
            'sentence_id': int(cache['sentence_ids'][index]),
            'start': int(starts[index]), 'end': int(ends[index]),
            'role_id': int(role_id), 'role': role_names[int(role_id)],
            'frame_prior_weight': float(sample_weight[index]),
        }
        for index, role_id in enumerate(prediction)
    ]
    with (output_dir / 'best_predictions.json').open('w', encoding='utf8') as handle:
        json.dump(prediction_rows, handle, ensure_ascii=False, indent=2)

    summary = {
        'normalization': 'per_sample_zscore_over_1009_roles',
        'best_b_only_fixed_prior': next(
            row for row in best_fixed_by_weight if row['b_weight'] == 1.0
        ),
        'best_second_model_only_fixed_prior': next(
            row for row in best_fixed_by_weight if row['b_weight'] == 0.0
        ),
        'best_fixed_prior': best_fixed,
        'best_adaptive_prior': best_adaptive,
        'best_overall': best_overall,
        'candidate_adaptive_b_weights': [row['b_weight'] for row in candidate_weight_rows],
        'num_predicted_spans': int(len(b_logits)),
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
