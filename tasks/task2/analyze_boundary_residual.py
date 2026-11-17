#!/usr/bin/env python3
"""Post-hoc H1 boundary-residual diagnostic for Task2.

The H1 model trains start/end token heads but deliberately leaves span
inference unchanged.  This script tests whether those learned heads contain
useful complementary evidence before we spend another five-epoch run on H2:

    score(i, j) = span_score(i, j)
                  + beta * (start_score(i) + end_score(j))

Both ``beta`` and the final safe-decode threshold are searched on dev.  The
script is analysis-only: it never changes or overwrites a checkpoint.
"""
import argparse
import csv
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from .analyze_threshold import collate, matrix_to_char_candidates, prf
from .dataset import Dataset
from .decoding import build_safe_span_mask
from .model_factory import build_task2_model


def float_grid(start, stop, step):
    if step <= 0:
        raise ValueError(f'grid step must be positive, got {step}')
    values = np.arange(start, stop + step / 2.0, step, dtype=np.float64)
    return np.round(values, 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Search a post-hoc start/end boundary residual for Task2 H1.'
    )
    parser.add_argument(
        '--checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-H1-boundary-aux-w010-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument('--boundary_loss_weight', type=float, default=0.1)
    parser.add_argument('--beta_min', type=float, default=-0.5)
    parser.add_argument('--beta_max', type=float, default=0.5)
    parser.add_argument('--beta_step', type=float, default=0.05)
    parser.add_argument('--threshold_min', type=float, default=-1.0)
    parser.add_argument('--threshold_max', type=float, default=0.5)
    parser.add_argument('--threshold_step', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--max_examples', type=int, default=0)
    parser.add_argument(
        '--output_dir',
        default='./tasks/task2/analysis/task2-H1-boundary-residual',
    )
    return parser.parse_args()


def load_rows(path, max_examples):
    with open(path, encoding='utf8') as handle:
        rows = json.load(handle)
    return rows[:max_examples] if max_examples > 0 else rows


def build_gold(row):
    target = row['target'][-1]
    coverage = np.zeros(len(row['text']), dtype=bool)
    spans = set()
    for span in row['cfn_spans']:
        if span['end'] < target['start'] or span['start'] > target['end']:
            start, end = int(span['start']), int(span['end'])
            spans.add((start, end))
            coverage[start:end + 1] = True
    return spans, coverage


def interval_coverage_scores(length, starts, ends, scores):
    matrix = np.full((length, length), -np.inf, dtype=np.float32)
    matrix[starts.astype(np.int64), ends.astype(np.int64)] = scores
    suffix_end = np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
    prefix_start = np.maximum.accumulate(suffix_end, axis=0)
    positions = np.arange(length)
    return prefix_start[positions, positions]


def counts_at_thresholds(values, thresholds):
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    return len(ordered) - np.searchsorted(ordered, thresholds, side='left')


def collect_items(args, dataset, rows, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate, device=device),
        drop_last=False,
    )
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.max_cls = dataset.dataset.max_cls if isinstance(dataset, Subset) else dataset.max_cls
    config.task2_boundary_loss_weight = args.boundary_loss_weight
    model = build_task2_model(config, 'boundary_auxiliary')
    state = torch.load(args.checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'checkpoint/model mismatch: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    model = model.to(device).eval()
    print(f'[load] missing=0 unexpected=0 checkpoint={args.checkpoint}')

    items = []
    row_offset = 0
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, sentence_ids, target_cls in tqdm(
                loader, desc='cache-H1-boundaries'):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=target,
                target_cls=target_cls,
                labels=None,
                device=device,
                for_test=True,
            )
            logits = output['logits']
            boundary_pair = (
                output['boundary_start_logits'].unsqueeze(2)
                + output['boundary_end_logits'].unsqueeze(1)
            )
            valid = build_safe_span_mask(attention_mask, target)
            for index in range(input_ids.shape[0]):
                span_scores, starts, ends = matrix_to_char_candidates(
                    logits[index], valid[index], target[index]
                )
                boundary_scores, boundary_starts, boundary_ends = matrix_to_char_candidates(
                    boundary_pair[index], valid[index], target[index]
                )
                if not (np.array_equal(starts, boundary_starts)
                        and np.array_equal(ends, boundary_ends)):
                    raise RuntimeError('span/boundary candidate coordinate mismatch')
                row = rows[row_offset + index]
                gold_spans, gold_coverage = build_gold(row)
                gold_mask = np.fromiter(
                    ((int(start), int(end)) in gold_spans
                     for start, end in zip(starts, ends)),
                    dtype=bool,
                    count=len(starts),
                )
                if int(gold_mask.sum()) != len(gold_spans):
                    raise RuntimeError(
                        f'gold/candidate mismatch at row {row_offset + index}: '
                        f'{int(gold_mask.sum())} candidates vs {len(gold_spans)} gold'
                    )
                items.append({
                    'length': len(row['text']),
                    'starts': starts,
                    'ends': ends,
                    'span_scores': span_scores,
                    'boundary_scores': boundary_scores,
                    'gold_mask': gold_mask,
                    'gold_coverage': gold_coverage,
                })
            row_offset += input_ids.shape[0]
    return items


def evaluate_beta(items, beta, thresholds):
    exact_tp = np.zeros(len(thresholds), dtype=np.int64)
    exact_pred = np.zeros(len(thresholds), dtype=np.int64)
    token_tp = np.zeros(len(thresholds), dtype=np.int64)
    token_pred = np.zeros(len(thresholds), dtype=np.int64)
    exact_gold = 0
    token_gold = 0

    for item in items:
        combined = item['span_scores'] + beta * item['boundary_scores']
        exact_pred += counts_at_thresholds(combined, thresholds)
        exact_tp += counts_at_thresholds(combined[item['gold_mask']], thresholds)
        exact_gold += int(item['gold_mask'].sum())

        coverage_scores = interval_coverage_scores(
            item['length'], item['starts'], item['ends'], combined
        )
        token_pred += counts_at_thresholds(coverage_scores, thresholds)
        token_tp += counts_at_thresholds(
            coverage_scores[item['gold_coverage']], thresholds
        )
        token_gold += int(item['gold_coverage'].sum())

    output = []
    for index, threshold in enumerate(thresholds):
        exact = prf(exact_tp[index], exact_pred[index], exact_gold)
        token = prf(token_tp[index], token_pred[index], token_gold)
        output.append({
            'beta': float(beta),
            'threshold': float(threshold),
            'exact_precision': exact['precision'],
            'exact_recall': exact['recall'],
            'exact_f1': exact['f1'],
            'exact_tp': exact['tp'],
            'predicted_spans': exact['predicted'],
            'gold_spans': exact['gold'],
            'token_precision': token['precision'],
            'token_recall': token['recall'],
            'token_f1': token['f1'],
            'token_tp': token['tp'],
            'predicted_tokens': token['predicted'],
            'gold_tokens': token['gold'],
        })
    return output


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    full_dataset = Dataset(args.dev_file, args.frame_info_file, tokenizer)
    rows = load_rows(args.dev_file, args.max_examples)
    dataset = (
        Subset(full_dataset, range(len(rows)))
        if args.max_examples > 0 else full_dataset
    )
    items = collect_items(args, dataset, rows, device)

    betas = float_grid(args.beta_min, args.beta_max, args.beta_step)
    thresholds = float_grid(
        args.threshold_min, args.threshold_max, args.threshold_step
    )
    # Always retain the original H1 calibrated operating point and no-residual
    # beta for an exact sanity comparison.
    betas = np.unique(np.concatenate([betas, np.asarray([0.0])]))
    thresholds = np.unique(
        np.concatenate([thresholds, np.asarray([-0.37, 0.0])])
    )
    grid = []
    for beta in tqdm(betas, desc='boundary-residual-grid'):
        grid.extend(evaluate_beta(items, float(beta), thresholds))

    best_token = max(grid, key=lambda row: row['token_f1'])
    best_exact = max(grid, key=lambda row: row['exact_f1'])
    beta_zero_rows = [row for row in grid if abs(row['beta']) < 1e-12]
    h1_calibrated = min(
        beta_zero_rows, key=lambda row: abs(row['threshold'] - (-0.37))
    )
    summary = {
        'device': str(device),
        'checkpoint': args.checkpoint,
        'num_examples': len(items),
        'h1_no_residual_at_calibrated_threshold': h1_calibrated,
        'best_token_overlap': best_token,
        'best_exact': best_exact,
        'token_f1_gain_over_h1': (
            best_token['token_f1'] - h1_calibrated['token_f1']
        ),
        'note': (
            'Analysis-only residual: span(i,j) + beta*(start(i)+end(j)). '
            'A positive best beta supports H2; beta=0 or negative evidence does not.'
        ),
    }
    with open(os.path.join(args.output_dir, 'grid.csv'), 'w',
              newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0].keys()))
        writer.writeheader()
        writer.writerows(grid)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w',
              encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w',
              encoding='utf8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
