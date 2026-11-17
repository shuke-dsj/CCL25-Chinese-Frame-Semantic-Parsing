#!/usr/bin/env python3
"""Centered-logit ensemble analysis for Task2 B/D/E checkpoints.

Each model score is first shifted by its independently tuned dev threshold:

    centered_score_m = raw_score_m - calibrated_threshold_m

The script then searches convex ensemble weights and a small final threshold.
Models are loaded and evaluated one at a time, so three BERT checkpoints do not
need to coexist on GPU. All final metrics use the safe span mask and the same
official-like token-coverage calculation as ``analyze_threshold.py``.
"""
import argparse
import csv
import gc
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


MODEL_ORDER = ('B', 'D', 'E')


def float_grid(start, stop, step):
    if step <= 0:
        raise ValueError(f'grid step must be positive, got {step}')
    values = np.arange(start, stop + step / 2.0, step, dtype=np.float64)
    return np.round(values, 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze centered-logit B/D/E Task2 ensembles on the dev set.'
    )
    parser.add_argument(
        '--baseline_checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    parser.add_argument(
        '--target_checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-D-target-conditioned-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument(
        '--content_checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-E-content-aware-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument('--baseline_threshold', type=float, default=-0.26)
    parser.add_argument('--target_threshold', type=float, default=-0.75)
    parser.add_argument('--content_threshold', type=float, default=-0.20)
    parser.add_argument('--pair_weight_step', type=float, default=0.05)
    parser.add_argument('--triple_weight_step', type=float, default=0.10)
    parser.add_argument('--final_threshold_min', type=float, default=-0.30)
    parser.add_argument('--final_threshold_max', type=float, default=0.30)
    parser.add_argument('--final_threshold_step', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--task2_content_dim', type=int, default=32)
    parser.add_argument('--task2_content_scale_init', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument(
        '--output_dir', default='./tasks/task2/analysis/task2-BDE-centered-ensemble'
    )
    parser.add_argument(
        '--max_examples', type=int, default=0,
        help='Debug only: use the first N examples. Zero uses the complete dev set.',
    )
    return parser.parse_args()


def load_raw_rows(dev_file, max_examples):
    with open(dev_file, encoding='utf8') as handle:
        rows = json.load(handle)
    if max_examples > 0:
        rows = rows[:max_examples]
    return rows


def build_gold_item(row):
    target_raw = row['target'][-1]
    gold_coverage = np.zeros(len(row['text']), dtype=bool)
    gold_spans = set()
    for span in row['cfn_spans']:
        # Dataset Task2 excludes spans that overlap the marked target.
        if span['end'] < target_raw['start'] or span['start'] > target_raw['end']:
            start, end = int(span['start']), int(span['end'])
            gold_spans.add((start, end))
            gold_coverage[start:end + 1] = True
    return gold_spans, gold_coverage


def collect_model_scores(args, dataset, raw_rows, model_key, variant, checkpoint, device):
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
    config.task2_content_dim = args.task2_content_dim
    config.task2_content_scale_init = args.task2_content_scale_init
    model = build_task2_model(config, variant)
    state = torch.load(checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'{model_key} checkpoint/model mismatch: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    model = model.to(device).eval()
    print(f'[load {model_key}] variant={variant} checkpoint={checkpoint}')

    output = []
    row_offset = 0
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, sentence_ids in tqdm(
                loader, desc=f'cache-{model_key}'):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=target,
                labels=None,
                device=device,
                for_test=True,
            )['logits']
            valid = build_safe_span_mask(attention_mask, target)
            for index in range(input_ids.shape[0]):
                scores, starts, ends = matrix_to_char_candidates(
                    logits[index], valid[index], target[index]
                )
                row = raw_rows[row_offset + index]
                output.append({
                    'sentence_id': row['sentence_id'],
                    'scores': scores,
                    'starts': starts,
                    'ends': ends,
                })
            row_offset += input_ids.shape[0]

    del state, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return output


def assemble_cache(raw_rows, score_cache):
    items = []
    for index, row in enumerate(raw_rows):
        reference = score_cache['B'][index]
        gold_spans, gold_coverage = build_gold_item(row)
        starts, ends = reference['starts'], reference['ends']
        for key in MODEL_ORDER:
            candidate = score_cache[key][index]
            if candidate['sentence_id'] != row['sentence_id']:
                raise RuntimeError(f'row order mismatch for model {key} at index {index}')
            if not (np.array_equal(starts, candidate['starts']) and
                    np.array_equal(ends, candidate['ends'])):
                raise RuntimeError(f'candidate coordinate mismatch for model {key} at index {index}')
        gold_mask = np.fromiter(
            ((int(start), int(end)) in gold_spans for start, end in zip(starts, ends)),
            dtype=bool,
            count=len(starts),
        )
        if int(gold_mask.sum()) != len(gold_spans):
            raise RuntimeError(
                f'gold/candidate mismatch at index {index}: '
                f'{int(gold_mask.sum())} candidates vs {len(gold_spans)} gold spans'
            )
        items.append({
            'length': len(row['text']),
            'starts': starts,
            'ends': ends,
            'gold_mask': gold_mask,
            'gold_coverage': gold_coverage,
            'scores': {key: score_cache[key][index]['scores'] for key in MODEL_ORDER},
        })
    return items


def interval_coverage_scores(length, starts, ends, span_scores):
    """Return max score of any candidate span covering each character.

    For score matrix S(i,j), a character t is covered at threshold q iff
    max_{i<=t<=j} S(i,j) >= q. Suffix maxima over end followed by prefix
    maxima over start compute all character maxima in O(length^2).
    """
    matrix = np.full((length, length), -np.inf, dtype=np.float32)
    matrix[starts.astype(np.int64), ends.astype(np.int64)] = span_scores
    suffix_end = np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
    prefix_start = np.maximum.accumulate(suffix_end, axis=0)
    positions = np.arange(length)
    return prefix_start[positions, positions]


def counts_at_thresholds(values, thresholds):
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    return len(ordered) - np.searchsorted(ordered, thresholds, side='left')


def evaluate_weights(items, centered_thresholds, weights, final_thresholds):
    exact_tp = np.zeros(len(final_thresholds), dtype=np.int64)
    exact_pred = np.zeros(len(final_thresholds), dtype=np.int64)
    token_tp = np.zeros(len(final_thresholds), dtype=np.int64)
    token_pred = np.zeros(len(final_thresholds), dtype=np.int64)
    exact_gold = 0
    token_gold = 0

    for item in items:
        combined = np.zeros_like(item['scores']['B'], dtype=np.float32)
        for key in MODEL_ORDER:
            weight = weights.get(key, 0.0)
            if weight:
                combined += weight * (item['scores'][key] - centered_thresholds[key])

        exact_pred += counts_at_thresholds(combined, final_thresholds)
        gold_scores = combined[item['gold_mask']]
        exact_tp += counts_at_thresholds(gold_scores, final_thresholds)
        exact_gold += int(item['gold_mask'].sum())

        coverage_scores = interval_coverage_scores(
            item['length'], item['starts'], item['ends'], combined
        )
        token_pred += counts_at_thresholds(coverage_scores, final_thresholds)
        token_tp += counts_at_thresholds(
            coverage_scores[item['gold_coverage']], final_thresholds
        )
        token_gold += int(item['gold_coverage'].sum())

    rows = []
    for index, threshold in enumerate(final_thresholds):
        exact = prf(exact_tp[index], exact_pred[index], exact_gold)
        token = prf(token_tp[index], token_pred[index], token_gold)
        rows.append({
            'weights': dict(weights),
            'final_threshold': float(threshold),
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
    return rows


def pair_weights(left, right, step):
    for left_weight in float_grid(0.0, 1.0, step):
        yield {
            left: float(left_weight),
            right: float(np.round(1.0 - left_weight, 10)),
        }


def triple_weights(step):
    values = float_grid(0.0, 1.0, step)
    for b_weight in values:
        for d_weight in values:
            e_weight = float(np.round(1.0 - b_weight - d_weight, 10))
            if e_weight < -1e-8:
                continue
            if e_weight < 0:
                e_weight = 0.0
            yield {'B': float(b_weight), 'D': float(d_weight), 'E': e_weight}


def public_row(family, row):
    weights = row['weights']
    return {
        'family': family,
        'weight_B': weights.get('B', 0.0),
        'weight_D': weights.get('D', 0.0),
        'weight_E': weights.get('E', 0.0),
        **{key: value for key, value in row.items() if key != 'weights'},
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    full_dataset = Dataset(args.dev_file, args.frame_info_file, tokenizer)
    raw_rows = load_raw_rows(args.dev_file, args.max_examples)
    if args.max_examples > 0:
        dataset = Subset(full_dataset, range(len(raw_rows)))
    else:
        dataset = full_dataset

    specs = {
        'B': ('baseline', args.baseline_checkpoint),
        'D': ('target_conditioned', args.target_checkpoint),
        'E': ('content_aware', args.content_checkpoint),
    }
    score_cache = {}
    for key in MODEL_ORDER:
        variant, checkpoint = specs[key]
        score_cache[key] = collect_model_scores(
            args, dataset, raw_rows, key, variant, checkpoint, device
        )
    items = assemble_cache(raw_rows, score_cache)
    del score_cache
    gc.collect()

    centered_thresholds = {
        'B': args.baseline_threshold,
        'D': args.target_threshold,
        'E': args.content_threshold,
    }
    final_thresholds = float_grid(
        args.final_threshold_min, args.final_threshold_max, args.final_threshold_step
    )
    # Keep the model-specific calibrated decision boundary available even when
    # a custom final-threshold grid would otherwise miss exactly zero.
    final_thresholds = np.unique(np.concatenate([final_thresholds, np.asarray([0.0])]))
    families = {
        'B+D': pair_weights('B', 'D', args.pair_weight_step),
        'B+E': pair_weights('B', 'E', args.pair_weight_step),
        'B+D+E': triple_weights(args.triple_weight_step),
    }
    all_rows = []
    family_summary = {}
    for family, generator in families.items():
        family_rows = []
        weights_list = list(generator)
        for weights in tqdm(weights_list, desc=f'ensemble-{family}'):
            evaluated = evaluate_weights(
                items, centered_thresholds, weights, final_thresholds
            )
            family_rows.extend(public_row(family, row) for row in evaluated)
        best_token = max(family_rows, key=lambda row: row['token_f1'])
        best_exact = max(family_rows, key=lambda row: row['exact_f1'])
        family_summary[family] = {
            'best_token_overlap': best_token,
            'best_exact': best_exact,
            'num_weight_settings': len(weights_list),
        }
        all_rows.extend(family_rows)

    # Sanity checks: a pure endpoint at final threshold zero must reproduce
    # that model's independently calibrated result.
    zero_index = int(np.argmin(np.abs(final_thresholds)))
    single_models = {}
    for key in MODEL_ORDER:
        rows = evaluate_weights(
            items, centered_thresholds, {key: 1.0}, final_thresholds
        )
        single_models[key] = public_row(f'{key}-only', rows[zero_index])

    best_overall = max(all_rows, key=lambda row: row['token_f1'])
    summary = {
        'device': str(device),
        'num_examples': len(items),
        'debug_subset': args.max_examples > 0,
        'checkpoints': {
            key: {'variant': specs[key][0], 'path': specs[key][1]}
            for key in MODEL_ORDER
        },
        'centering_thresholds': centered_thresholds,
        'search': {
            'pair_weight_step': args.pair_weight_step,
            'triple_weight_step': args.triple_weight_step,
            'final_threshold_min': args.final_threshold_min,
            'final_threshold_max': args.final_threshold_max,
            'final_threshold_step': args.final_threshold_step,
        },
        'single_model_centering_sanity': single_models,
        'families': family_summary,
        'best_overall_token_overlap': best_overall,
        'note': (
            'Each raw model score is shifted by its independently tuned threshold before blending. '
            'All metrics use safe decoding candidates. Token overlap is computed from the union of '
            'covered character positions for each example.'
        ),
    }

    csv_path = os.path.join(args.output_dir, 'ensemble_grid.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
