#!/usr/bin/env python3
"""Search Task2 thresholds using exact-span and token-overlap metrics."""
import argparse
import csv
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model_factory import build_task2_model


def collate(data, device=None):
    bs = len(data)
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device)
    attention_mask = torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device)
    target = [x[2] for x in data]
    labels = [x[3] for x in data]
    label_matrix = torch.zeros(bs, max_len, max_len, device=device)
    for i, spans in enumerate(labels):
        for start, end in spans:
            label_matrix[i, start, end] = 1
    target_cls = torch.tensor([x[5] for x in data], dtype=torch.long, device=device)
    return input_ids, attention_mask, target, label_matrix, [x[4] for x in data], target_cls


def prf(tp, predicted, gold):
    precision = tp / max(predicted, 1e-9)
    recall = tp / max(gold, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'predicted': int(predicted), 'gold': int(gold),
    }


def safe_mask(attention_mask, target):
    batch, length = attention_mask.shape
    output = torch.zeros(batch, length, length, dtype=torch.bool, device=attention_mask.device)
    for i, (target_start, target_end) in enumerate(target):
        seq_len = int(attention_mask[i].sum().item())
        # Valid characters left of target: token positions [1, target_start-1].
        if target_start > 1:
            n = target_start - 1
            output[i, 1:target_start, 1:target_start] = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=output.device)
            )
        # After inserting two markers, right-side characters begin at target_end+3.
        right_start = target_end + 3
        right_end = seq_len - 1  # exclude [SEP]
        if right_start < right_end:
            n = right_end - right_start
            output[i, right_start:right_end, right_start:right_end] = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=output.device)
            )
    return output


def legacy_mask(attention_mask):
    return torch.triu(attention_mask.unsqueeze(2).bool() & attention_mask.unsqueeze(1).bool())


def matrix_to_char_candidates(logits, valid_mask, target):
    """Return candidate scores and original character coordinates for one example."""
    indices = torch.nonzero(valid_mask, as_tuple=False)
    scores = logits[valid_mask].detach().cpu().numpy().astype(np.float32)
    starts, ends = [], []
    target_start, target_end = target
    for start, end in indices.detach().cpu().tolist():
        if end < target_start:
            starts.append(start - 1); ends.append(end - 1)
        else:  # safe_mask guarantees start >= target_end+3
            starts.append(start - 3); ends.append(end - 3)
    return scores, np.asarray(starts, dtype=np.int16), np.asarray(ends, dtype=np.int16)


def token_coverage(length, starts, ends, selected):
    if not np.any(selected):
        return np.zeros(length, dtype=bool)
    diff = np.zeros(length + 1, dtype=np.int32)
    chosen_starts = starts[selected].astype(np.int64)
    chosen_ends = ends[selected].astype(np.int64)
    np.add.at(diff, chosen_starts, 1)
    np.add.at(diff, chosen_ends + 1, -1)
    return np.cumsum(diff[:-1]) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./tasks/task2/experiment_record/legacy-saves/checkpoints/model_task2_best.bin')
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--threshold_min', default=-2.5, type=float)
    parser.add_argument('--threshold_max', default=2.5, type=float)
    parser.add_argument('--threshold_step', default=0.1, type=float)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--task2_model_variant', default='baseline',
                        choices=['baseline', 'target_conditioned', 'content_aware',
                                 'frame_prototype_conditioned', 'boundary_auxiliary'])
    parser.add_argument('--task2_content_dim', default=32, type=int)
    parser.add_argument('--task2_content_scale_init', default=0.1, type=float)
    parser.add_argument('--task2_frame_prototype_file',
                        default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    parser.add_argument('--task2_frame_query_dim', default=64, type=int)
    parser.add_argument('--task2_frame_scale_init', default=0.1, type=float)
    parser.add_argument('--task2_boundary_loss_weight', default=0.1, type=float)
    parser.add_argument('--output_dir', default='./tasks/task2/analysis/task2-threshold-baseline')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Dataset(args.dev_file, args.frame_info_file, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=device), drop_last=False)
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.max_cls = dataset.max_cls
    config.task2_content_dim = args.task2_content_dim
    config.task2_content_scale_init = args.task2_content_scale_init
    config.task2_frame_prototype_file = args.task2_frame_prototype_file
    config.task2_frame_query_dim = args.task2_frame_query_dim
    config.task2_frame_scale_init = args.task2_frame_scale_init
    config.task2_frame_names = list(dataset.label2cls.keys())
    config.task2_boundary_loss_weight = args.task2_boundary_loss_weight
    model = build_task2_model(config, args.task2_model_variant)
    message = model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print(f'[load] missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}')

    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2,
                           args.threshold_step, dtype=np.float64)
    exact_tp = np.zeros(len(thresholds), dtype=np.int64)
    exact_pred = np.zeros(len(thresholds), dtype=np.int64)
    exact_gold = 0
    legacy_counts = [0, 0, 0]
    candidates = []

    raw_rows = json.load(open(args.dev_file, encoding='utf8'))
    row_offset = 0
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, sentence_ids, target_cls in tqdm(
                loader, desc='task2-threshold'):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                           target_cls=target_cls, labels=None, device=device,
                           for_test=True)['logits']
            valid = safe_mask(attention_mask, target)
            gold = labels.bool()
            exact_gold += int(gold.sum().item())

            legacy_prediction = (logits >= 0) & legacy_mask(attention_mask)
            legacy_counts[0] += int((legacy_prediction & gold).sum().item())
            legacy_counts[1] += int(legacy_prediction.sum().item())
            legacy_counts[2] += int(gold.sum().item())

            # Exact-span counts are cheap to evaluate for all thresholds on GPU.
            for index, threshold in enumerate(thresholds):
                prediction = (logits >= float(threshold)) & valid
                exact_tp[index] += int((prediction & gold).sum().item())
                exact_pred[index] += int(prediction.sum().item())

            for i in range(input_ids.shape[0]):
                scores, starts, ends = matrix_to_char_candidates(logits[i], valid[i], target[i])
                row = raw_rows[row_offset + i]
                gold_coverage = np.zeros(len(row['text']), dtype=bool)
                gold_spans = []
                target_raw = row['target'][-1]
                for span in row['cfn_spans']:
                    if span['end'] < target_raw['start'] or span['start'] > target_raw['end']:
                        gold_coverage[span['start']:span['end'] + 1] = True
                        gold_spans.append((span['start'], span['end']))
                candidates.append({
                    'scores': scores, 'starts': starts, 'ends': ends,
                    'length': len(row['text']), 'gold_coverage': gold_coverage,
                    'gold_spans': gold_spans, 'sentence_id': row['sentence_id'],
                })
            row_offset += input_ids.shape[0]

    # Official-like token-overlap sweep. Aggregate per example so duplicate sentence_ids remain distinct.
    token_tp = np.zeros(len(thresholds), dtype=np.int64)
    token_pred = np.zeros(len(thresholds), dtype=np.int64)
    token_gold = sum(int(item['gold_coverage'].sum()) for item in candidates)
    for item in tqdm(candidates, desc='token-overlap-sweep'):
        for index, threshold in enumerate(thresholds):
            covered = token_coverage(item['length'], item['starts'], item['ends'],
                                     item['scores'] >= float(threshold))
            token_tp[index] += int((covered & item['gold_coverage']).sum())
            token_pred[index] += int(covered.sum())

    rows = []
    for index, threshold in enumerate(thresholds):
        exact = prf(exact_tp[index], exact_pred[index], exact_gold)
        token = prf(token_tp[index], token_pred[index], token_gold)
        rows.append({
            'threshold': float(threshold),
            'exact_precision': exact['precision'], 'exact_recall': exact['recall'], 'exact_f1': exact['f1'],
            'exact_tp': exact['tp'], 'predicted_spans': exact['predicted'], 'gold_spans': exact['gold'],
            'avg_predicted_spans': exact['predicted'] / len(dataset),
            'token_precision': token['precision'], 'token_recall': token['recall'], 'token_f1': token['f1'],
            'token_tp': token['tp'], 'predicted_tokens': token['predicted'], 'gold_tokens': token['gold'],
            'avg_predicted_tokens': token['predicted'] / len(dataset),
        })
    best_exact = max(rows, key=lambda row: row['exact_f1'])
    best_token = max(rows, key=lambda row: row['token_f1'])
    threshold_zero = next(
        (row for row in rows if abs(row['threshold']) < 1e-10), None
    )
    closest_to_zero = min(rows, key=lambda row: abs(row['threshold']))
    summary = {
        'legacy_threshold_0_exact': prf(*legacy_counts),
        'safe_threshold_0': threshold_zero,
        'closest_to_zero_in_grid': closest_to_zero,
        'best_safe_exact': best_exact,
        'best_safe_token_overlap': best_token,
        'num_examples': len(dataset),
        'note': ('Exact requires identical start/end. Token-overlap uses the union of covered character '
                 'positions per example and is an official-like implementation based on the README. '
                 'safe_threshold_0 is null when zero is outside the requested threshold grid.'),
    }
    with open(os.path.join(args.output_dir, 'threshold_grid.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

