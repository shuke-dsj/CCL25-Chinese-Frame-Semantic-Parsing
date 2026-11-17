#!/usr/bin/env python3
"""Threshold calibration for G1 oracle-frame definition-guided Task2."""
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

from .analyze_threshold import matrix_to_char_candidates, prf, token_coverage
from .dataset_frame_guided import FrameGuidedDataset, frame_guided_collate
from .decoding import build_safe_span_mask
from .model_frame_guided import FrameGuidedGlobalPointer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--threshold_min', default=-3.0, type=float)
    parser.add_argument('--threshold_max', default=1.0, type=float)
    parser.add_argument('--threshold_step', default=0.05, type=float)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--task2_role_query_dim', default=64, type=int)
    parser.add_argument('--task2_role_scale_init', default=0.1, type=float)
    parser.add_argument(
        '--output_dir',
        default='./tasks/task2/analysis/task2-G1-definition-guided-threshold',
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = FrameGuidedDataset(
        args.dev_file, args.frame_info_file, tokenizer, max_length=args.max_length
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(
            frame_guided_collate, pad_token_id=tokenizer.pad_token_id, device=device
        ),
        drop_last=False,
    )

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.task2_role_query_dim = args.task2_role_query_dim
    config.task2_role_scale_init = args.task2_role_scale_init
    model = FrameGuidedGlobalPointer(config)
    message = model.load_state_dict(
        torch.load(args.checkpoint, map_location='cpu'), strict=False
    )
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'G1 checkpoint mismatch: missing={message.missing_keys} '
            f'unexpected={message.unexpected_keys}'
        )
    model = model.to(device).eval()
    print(f'[G1 load] checkpoint={args.checkpoint} device={device}')

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
        dtype=np.float64,
    )
    exact_tp = np.zeros(len(thresholds), dtype=np.int64)
    exact_pred = np.zeros(len(thresholds), dtype=np.int64)
    exact_gold = 0
    candidates = []
    raw_rows = json.load(open(args.dev_file, encoding='utf8'))
    row_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='G1-threshold'):
            output = model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                target=batch['target'], sentence_end=batch['sentence_end'],
                role_spans=batch['role_spans'], role_mask=batch['role_mask'],
                labels=None, device=device, for_test=True,
            )
            logits = output['logits']
            sentence_length = logits.shape[-1]
            positions = torch.arange(sentence_length, device=device).unsqueeze(0)
            sentence_attention = (
                positions <= batch['sentence_end'].unsqueeze(1)
            ).long()
            valid = build_safe_span_mask(sentence_attention, batch['target'])
            gold = batch['labels'].any(dim=1)
            exact_gold += int(gold.sum().item())

            for threshold_index, threshold in enumerate(thresholds):
                prediction = (logits >= float(threshold)) & valid
                exact_tp[threshold_index] += int((prediction & gold).sum().item())
                exact_pred[threshold_index] += int(prediction.sum().item())

            for item_index in range(logits.shape[0]):
                scores, starts, ends = matrix_to_char_candidates(
                    logits[item_index], valid[item_index], batch['target'][item_index]
                )
                row = raw_rows[row_offset + item_index]
                gold_coverage = np.zeros(len(row['text']), dtype=bool)
                target_raw = row['target'][-1]
                for span in row['cfn_spans']:
                    if (span['end'] < target_raw['start']
                            or span['start'] > target_raw['end']):
                        gold_coverage[span['start']:span['end'] + 1] = True
                candidates.append({
                    'scores': scores,
                    'starts': starts,
                    'ends': ends,
                    'length': len(row['text']),
                    'gold_coverage': gold_coverage,
                })
            row_offset += logits.shape[0]

    token_tp = np.zeros(len(thresholds), dtype=np.int64)
    token_pred = np.zeros(len(thresholds), dtype=np.int64)
    token_gold = sum(int(item['gold_coverage'].sum()) for item in candidates)
    for item in tqdm(candidates, desc='G1-token-overlap-sweep'):
        for threshold_index, threshold in enumerate(thresholds):
            covered = token_coverage(
                item['length'], item['starts'], item['ends'],
                item['scores'] >= float(threshold),
            )
            token_tp[threshold_index] += int(
                (covered & item['gold_coverage']).sum()
            )
            token_pred[threshold_index] += int(covered.sum())

    rows = []
    for threshold_index, threshold in enumerate(thresholds):
        exact = prf(exact_tp[threshold_index], exact_pred[threshold_index], exact_gold)
        token = prf(token_tp[threshold_index], token_pred[threshold_index], token_gold)
        rows.append({
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

    threshold_zero = next(
        (row for row in rows if abs(row['threshold']) < 1e-10), None
    )
    closest_to_zero = min(rows, key=lambda row: abs(row['threshold']))
    best_exact = max(rows, key=lambda row: row['exact_f1'])
    best_token = max(rows, key=lambda row: row['token_f1'])
    summary = {
        'checkpoint': args.checkpoint,
        'safe_threshold_0': threshold_zero,
        'closest_to_zero_in_grid': closest_to_zero,
        'best_safe_exact': best_exact,
        'best_safe_token_overlap': best_token,
        'num_examples': len(dataset),
        'oracle_frame': True,
        'note': (
            'G1 uses gold/oracle frames. Token-overlap is the union of covered '
            'character positions per example.'
        ),
    }
    with open(
        os.path.join(args.output_dir, 'threshold_grid.csv'),
        'w', newline='', encoding='utf-8-sig',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(
        os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf8'
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf8'
    ) as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
