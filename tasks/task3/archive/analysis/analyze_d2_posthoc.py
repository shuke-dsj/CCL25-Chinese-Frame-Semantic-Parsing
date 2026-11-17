#!/usr/bin/env python3
"""Post-hoc D2 analysis and targeted B/C2/D/D2 ensemble search."""

from __future__ import annotations

import argparse
import csv
import json
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c2_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-C2-frame/pipeline_logits.npz'))
    parser.add_argument('--d_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-D-soft-top5/pipeline_logits.npz'))
    parser.add_argument('--d2_epoch1_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-D2-epoch1/pipeline_logits.npz'))
    parser.add_argument('--d2_epoch2_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-D2-epoch2/pipeline_logits.npz'))
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset/cfn-dev.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    parser.add_argument('--output_dir', default=str(
        TASK_DIR / 'analysis/task3-D2-posthoc-and-four-model'))
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def validate_alignment(reference, other, name):
    for field in ('instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names'):
        if not np.array_equal(reference[field], other[field]):
            raise ValueError(f'{name} cache differs in {field}.')
    if not np.allclose(reference['frame_probs'], other['frame_probs'], atol=1e-7):
        raise ValueError(f'{name} cache differs in Task1 probabilities.')


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
    c2 = np.load(args.c2_cache, allow_pickle=False)
    d = np.load(args.d_cache, allow_pickle=False)
    e1 = np.load(args.d2_epoch1_cache, allow_pickle=False)
    e2 = np.load(args.d2_epoch2_cache, allow_pickle=False)
    for cache, name in ((d, 'D'), (e1, 'D2-e1'), (e2, 'D2-e2')):
        validate_alignment(c2, cache, name)

    frame_info, role_names, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    if c2['role_names'].tolist() != role_names:
        raise ValueError('Role order differs from ontology.')
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0
    frame_probs = c2['frame_probs'].astype(np.float32)
    log_prior = np.log(np.clip(frame_probs @ membership, 1e-8, 1.0)).astype(np.float32)
    dev_rows = read_json(args.dev_file)
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        dev_rows, role2idx,
        c2['instance_indices'].astype(np.int64),
        c2['starts'].astype(np.int64), c2['ends'].astype(np.int64),
        len(role_names),
    )

    b_norm = zscore_rows(c2['b_logits'].astype(np.float32))
    c2_norm = zscore_rows(c2['c_logits'].astype(np.float32))
    d_norm = zscore_rows(d['c_logits'].astype(np.float32))
    d2_e1_norm = zscore_rows(e1['c_logits'].astype(np.float32))
    d2_e2_norm = zscore_rows(e2['c_logits'].astype(np.float32))
    old_combined = 0.25 * b_norm + 0.10 * c2_norm + 0.65 * d_norm

    top2 = np.sort(np.partition(frame_probs, -2, axis=1)[:, -2:], axis=1)
    margin_confidence = percentile_confidence(top2[:, 1] - top2[:, 0])
    entropy = -(frame_probs * np.log(frame_probs.clip(min=1e-12))).sum(axis=1)
    confidence_features = {
        'margin_percentile': margin_confidence,
        'negative_entropy_percentile': percentile_confidence(-entropy),
    }

    grid = []
    best_fixed = None
    best_fixed_by_epoch = {}
    # D2 is added as a residual ensemble member on top of the previously
    # selected B/C2/D mixture.  This targeted search is less prone to Dev
    # overfitting than a completely unconstrained four-dimensional simplex.
    for epoch, d2_norm in ((1, d2_e1_norm), (2, d2_e2_norm)):
        epoch_best = None
        for d2_weight in np.round(np.arange(0.0, 0.6001, 0.025), 6):
            old_weight = 1.0 - d2_weight
            combined = old_weight * old_combined + d2_weight * d2_norm
            for prior in np.round(np.arange(0.0, 2.0001, 0.05), 6):
                prediction = (combined + prior * log_prior).argmax(axis=1)
                row = {
                    'strategy': 'fixed', 'epoch': epoch,
                    'old_B_C2_D_weight': float(old_weight),
                    'd2_weight': float(d2_weight),
                    'confidence': '', 'low': float(prior), 'high': float(prior),
                    'power': 0.0,
                    **evaluate(prediction, gold_mask, boundary_match,
                               gold_tuples, gold_boundaries),
                }
                grid.append(row)
                if is_better(row, epoch_best):
                    epoch_best = row
                if is_better(row, best_fixed):
                    best_fixed = row
        best_fixed_by_epoch[epoch] = epoch_best

    # Refine only around the three strongest fixed mixtures.
    fixed_candidates = sorted(
        [row for row in grid if row['strategy'] == 'fixed'],
        key=lambda row: row['exact_f1'], reverse=True,
    )
    unique_mixtures = []
    seen = set()
    for row in fixed_candidates:
        key = (row['epoch'], row['d2_weight'])
        if key not in seen:
            unique_mixtures.append(key)
            seen.add(key)
        if len(unique_mixtures) == 4:
            break

    best_adaptive = None
    lows = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    highs = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    powers = (0.5, 1.0, 2.0)
    for epoch, d2_weight in unique_mixtures:
        d2_norm = d2_e1_norm if epoch == 1 else d2_e2_norm
        old_weight = 1.0 - d2_weight
        combined = old_weight * old_combined + d2_weight * d2_norm
        for confidence_name, confidence in confidence_features.items():
            for low in lows:
                for high in highs:
                    if high < low:
                        continue
                    for power in powers:
                        prior = low + (high - low) * np.power(confidence, power)
                        prediction = (combined + prior[:, None] * log_prior).argmax(axis=1)
                        row = {
                            'strategy': 'adaptive', 'epoch': epoch,
                            'old_B_C2_D_weight': float(old_weight),
                            'd2_weight': float(d2_weight),
                            'confidence': confidence_name,
                            'low': low, 'high': high, 'power': power,
                            **evaluate(prediction, gold_mask, boundary_match,
                                       gold_tuples, gold_boundaries),
                        }
                        grid.append(row)
                        if is_better(row, best_adaptive):
                            best_adaptive = row

    best = best_adaptive if is_better(best_adaptive, best_fixed) else best_fixed
    d2_norm = d2_e1_norm if best['epoch'] == 1 else d2_e2_norm
    combined = (
        best['old_B_C2_D_weight'] * old_combined
        + best['d2_weight'] * d2_norm
    )
    if best['strategy'] == 'adaptive':
        confidence = confidence_features[best['confidence']]
        prior = best['low'] + (best['high'] - best['low']) * np.power(
            confidence, best['power'])
    else:
        prior = np.full(len(combined), best['low'], np.float32)
    best_prediction = (combined + prior[:, None] * log_prior).argmax(axis=1)
    old_prediction = (old_combined + 0.9 * log_prior).argmax(axis=1)
    indices = np.arange(len(best_prediction))
    old_correct = gold_mask[indices, old_prediction]
    best_correct = gold_mask[indices, best_prediction]

    with (output_dir / 'grid.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    summary = {
        'best_fixed': best_fixed,
        'best_fixed_by_epoch': best_fixed_by_epoch,
        'best_adaptive': best_adaptive,
        'best_overall': best,
        'reference_old_B_C2_D': {
            'weights': {'B': 0.25, 'C2': 0.10, 'D': 0.65},
            'prior': 0.9,
            **evaluate(old_prediction, gold_mask, boundary_match,
                       gold_tuples, gold_boundaries),
        },
        'complementarity_vs_old_ensemble': {
            'prediction_disagreements': int((old_prediction != best_prediction).sum()),
            'old_only_correct': int((old_correct & ~best_correct).sum()),
            'new_only_correct': int((~old_correct & best_correct).sum()),
            'both_correct': int((old_correct & best_correct).sum()),
            'oracle_union_tp': int((old_correct | best_correct).sum()),
        },
        'effective_four_model_weights': {
            'B': 0.25 * best['old_B_C2_D_weight'],
            'C2': 0.10 * best['old_B_C2_D_weight'],
            'D': 0.65 * best['old_B_C2_D_weight'],
            'D2': best['d2_weight'],
        },
        'adaptive_candidate_mixtures': [
            {'epoch': epoch, 'd2_weight': weight}
            for epoch, weight in unique_mixtures
        ],
        'num_spans': len(best_prediction),
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
