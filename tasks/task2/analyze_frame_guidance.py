#!/usr/bin/env python3
"""Measure whether Task1 predictions can safely guide frame-aware Task2/3.

This is a data/logit-only analysis: no model checkpoint or GPU is required.
It combines the saved Task1 baseline/prototype dev logits, measures top-k frame
recall, and quantifies how much the predicted frames reduce the candidate FE
inventory while retaining gold FE mentions.
"""
import argparse
import csv
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task1_logits',
        default='./tasks/task1/analysis/baseline-prototype-10ep/dev_logits.npz',
    )
    parser.add_argument(
        '--task1_summary',
        default='./tasks/task1/analysis/baseline-prototype-10ep/summary.json',
    )
    parser.add_argument('--train_file', default='./dataset/cfn-train.json')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--topk', default='1,3,5,10')
    parser.add_argument(
        '--output_dir', default='./tasks/task2/analysis/task2-frame-guidance-feasibility'
    )
    return parser.parse_args()


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate_frame_candidates(candidate_indices, dev_rows, frame_names, frame_to_fes,
                              total_gold_fe_mentions, global_fe_count):
    frame_hits = 0
    covered_mentions = 0
    all_mentions_covered_examples = 0
    candidate_role_counts = []
    candidate_frame_counts = []
    for index, row in enumerate(dev_rows):
        indices = list(dict.fromkeys(int(item) for item in candidate_indices[index]))
        predicted_frames = [frame_names[item] for item in indices]
        candidate_frame_counts.append(len(predicted_frames))
        frame_hits += int(row['frame'] in predicted_frames)
        candidate_fes = set()
        for name in predicted_frames:
            candidate_fes.update(frame_to_fes[name])
        candidate_role_counts.append(len(candidate_fes))
        gold_mentions = [span['fe_name'] for span in row['cfn_spans']]
        covered = sum(name in candidate_fes for name in gold_mentions)
        covered_mentions += covered
        all_mentions_covered_examples += int(covered == len(gold_mentions))
    mean_roles = float(np.mean(candidate_role_counts))
    return {
        'frame_recall': frame_hits / len(dev_rows),
        'gold_fe_mention_coverage': covered_mentions / max(total_gold_fe_mentions, 1),
        'all_gold_fe_covered_example_rate': all_mentions_covered_examples / len(dev_rows),
        'avg_candidate_frame_count': float(np.mean(candidate_frame_counts)),
        'avg_candidate_fe_count': mean_roles,
        'candidate_reduction_vs_global': 1.0 - mean_roles / global_fe_count,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    topk_values = sorted({int(value) for value in args.topk.split(',') if value.strip()})

    with open(args.frame_info_file, encoding='utf8') as handle:
        frame_info = json.load(handle)
    with open(args.train_file, encoding='utf8') as handle:
        train_rows = json.load(handle)
    with open(args.dev_file, encoding='utf8') as handle:
        dev_rows = json.load(handle)
    with open(args.task1_summary, encoding='utf8') as handle:
        task1_summary = json.load(handle)

    frame_names = [item['frame_name'] for item in frame_info]
    frame_to_idx = {name: index for index, name in enumerate(frame_names)}
    frame_to_fes = {
        item['frame_name']: {fe['fe_name'] for fe in item['fes']}
        for item in frame_info
    }
    global_fes = sorted({name for values in frame_to_fes.values() for name in values})
    seen_frames = {row['frame'] for row in train_rows}

    archive = np.load(args.task1_logits)
    baseline_logits = archive['baseline_logits']
    prototype_logits = archive['prototype_logits']
    gold = archive['gold'].astype(np.int64)
    sentence_ids = archive['sentence_ids']
    if len(dev_rows) != len(gold):
        raise RuntimeError(f'dev/logit length mismatch: {len(dev_rows)} vs {len(gold)}')
    if [row['sentence_id'] for row in dev_rows] != sentence_ids.tolist():
        raise RuntimeError('dev rows and saved Task1 logits have different sentence order')
    if baseline_logits.shape[1] != len(frame_names):
        raise RuntimeError(
            f'frame count mismatch: logits={baseline_logits.shape[1]} info={len(frame_names)}'
        )

    best = task1_summary['best_joint_hybrid']
    gamma = float(best['gamma'])
    baseline_weight = float(best['baseline_weight'])
    prototype_weight = float(best['prototype_weight'])
    seen_mask = np.asarray([name in seen_frames for name in frame_names], dtype=bool)
    calibrated_prototype = prototype_logits.copy()
    calibrated_prototype[:, seen_mask] -= gamma
    hybrid_scores = (
        baseline_weight * softmax(baseline_logits)
        + prototype_weight * softmax(calibrated_prototype)
    )

    ranked = np.argsort(-hybrid_scores, axis=1)
    gold_frames = [frame_names[index] for index in gold]
    missing_frame_rows = [row['sentence_id'] for row in dev_rows if row['frame'] not in frame_to_idx]
    if missing_frame_rows:
        raise RuntimeError(f'dev frames missing in frame_info, examples={missing_frame_rows[:5]}')

    total_gold_fe_mentions = sum(len(row['cfn_spans']) for row in dev_rows)
    global_fe_count = len(global_fes)
    per_k = []
    for k in topk_values:
        metrics = evaluate_frame_candidates(
            ranked[:, :k], dev_rows, frame_names, frame_to_fes,
            total_gold_fe_mentions, global_fe_count,
        )
        per_k.append({'top_k': k, **metrics})

    baseline_ranked = np.argsort(-baseline_logits, axis=1)
    prototype_ranked = np.argsort(-prototype_logits, axis=1)
    candidate_strategies = {}
    for name, candidate_indices in {
        'baseline_top5': baseline_ranked[:, :5],
        'prototype_top5': prototype_ranked[:, :5],
        'hybrid_top5': ranked[:, :5],
        'baseline_top10': baseline_ranked[:, :10],
        'baseline5_union_prototype5': np.asarray([
            list(dict.fromkeys(
                baseline_ranked[index, :5].tolist()
                + prototype_ranked[index, :5].tolist()
            ))
            for index in range(len(dev_rows))
        ], dtype=object),
    }.items():
        candidate_strategies[name] = evaluate_frame_candidates(
            candidate_indices, dev_rows, frame_names, frame_to_fes,
            total_gold_fe_mentions, global_fe_count,
        )

    oracle_candidate_counts = [len(frame_to_fes[row['frame']]) for row in dev_rows]
    oracle_covered = sum(
        span['fe_name'] in frame_to_fes[row['frame']]
        for row in dev_rows for span in row['cfn_spans']
    )

    # Character counts approximate WordPiece lengths for these mostly-Chinese
    # templates. Full FE definitions are intentionally excluded; AGED uses
    # them as separate training queries rather than concatenating all of them.
    template_length = {}
    full_fe_definition_length = {}
    for frame in frame_info:
        name = frame['frame_name']
        seen_fe_names = set()
        unique_fes = []
        for fe in frame['fes']:
            if fe['fe_name'] not in seen_fe_names:
                seen_fe_names.add(fe['fe_name'])
                unique_fes.append(fe)
        template = name + frame.get('frame_def', '') + '|'.join(
            fe['fe_name'] for fe in unique_fes
        )
        full_defs = template + ''.join(fe.get('fe_def', '') for fe in unique_fes)
        template_length[name] = len(template)
        full_fe_definition_length[name] = len(full_defs)
    gold_pair_lengths = [
        len(row['text']) + template_length[row['frame']] + 3 for row in dev_rows
    ]
    gold_full_definition_pair_lengths = [
        len(row['text']) + full_fe_definition_length[row['frame']] + 3 for row in dev_rows
    ]

    summary = {
        'num_examples': len(dev_rows),
        'num_frames': len(frame_names),
        'global_unique_fe_names': global_fe_count,
        'task1_hybrid': {
            'gamma': gamma,
            'baseline_weight': baseline_weight,
            'prototype_weight': prototype_weight,
            'top1_accuracy_recomputed': float(np.mean(ranked[:, 0] == gold)),
        },
        'topk_guidance': per_k,
        'candidate_strategies': candidate_strategies,
        'gold_frame_oracle': {
            'gold_fe_mention_coverage': oracle_covered / max(total_gold_fe_mentions, 1),
            'avg_candidate_fe_count': float(np.mean(oracle_candidate_counts)),
            'median_candidate_fe_count': percentile(oracle_candidate_counts, 50),
            'max_candidate_fe_count': int(max(oracle_candidate_counts)),
            'candidate_reduction_vs_global': 1.0 - float(np.mean(oracle_candidate_counts)) / global_fe_count,
        },
        'definition_input_feasibility': {
            'template': 'frame_name + frame_def + FE names',
            'pair_char_length_mean': float(np.mean(gold_pair_lengths)),
            'pair_char_length_p95': percentile(gold_pair_lengths, 95),
            'pair_char_length_max': int(max(gold_pair_lengths)),
            'pair_over_512_rate': float(np.mean(np.asarray(gold_pair_lengths) > 512)),
            'all_fe_definitions_pair_char_length_mean': float(
                np.mean(gold_full_definition_pair_lengths)
            ),
            'all_fe_definitions_pair_over_512_rate': float(
                np.mean(np.asarray(gold_full_definition_pair_lengths) > 512)
            ),
            'note': (
                'Frame definition plus FE names is the intended inference template. '
                'Concatenating every full FE definition is measured only to show why FE definitions '
                'should be separate training/query instances.'
            ),
        },
        'decision_note': (
            'Top-k frame recall measures Task1 error propagation. FE coverage measures whether the '
            'union of predicted-frame inventories still contains each gold role label; it is an '
            'ontology feasibility diagnostic, not a Task2 span F1.'
        ),
    }

    with open(os.path.join(args.output_dir, 'topk_guidance.csv'), 'w', newline='',
              encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_k[0].keys()))
        writer.writeheader()
        writer.writerows(per_k)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
