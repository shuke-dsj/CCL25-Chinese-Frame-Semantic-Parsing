#!/usr/bin/env python3
"""Evaluate task1 checkpoints, frequency buckets, and probability ensembles."""
import argparse
import codecs
import csv
import json
import os
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.task1.dataset import Dataset
from tasks.task1.model import Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_checkpoint', required=True)
    p.add_argument('--aware_checkpoint', required=True)
    p.add_argument('--aware_layers', default='last4', choices=['last', 'last4'])
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--train_file', default='./dataset/cfn-train.json')
    p.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    p.add_argument('--label_file', default='./dataset/frame_info.json')
    p.add_argument('--output_dir', default='./tasks/task1/analysis/task1-ensemble')
    return p.parse_args()


def collate(data, device=None):
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device)
    attention_mask = torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device)
    target = [x[2] for x in data]
    labels = torch.tensor([x[3] for x in data], dtype=torch.long, device=device)
    sentence_ids = [x[4] for x in data]
    return input_ids, attention_mask, target, labels, sentence_ids


def load_model(checkpoint, config_file, num_labels, variant, aware_layers, device):
    config = BertConfig.from_json_file(config_file)
    config.num_labels = num_labels
    config.model_variant = variant
    config.target_aware_layers = aware_layers
    model = Model(config)
    state = torch.load(checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f'[{variant}] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    return model.to(device)


def collect_logits(model, loader, device, desc):
    model.eval()
    logits_all, labels_all, ids_all = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, sentence_ids in tqdm(loader, desc=desc):
            out = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                        labels=None, device=device, for_test=True)
            logits_all.append(out['logits'].detach().cpu())
            labels_all.append(labels.detach().cpu())
            ids_all.extend(sentence_ids)
    return torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy(), ids_all


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def acc(pred, gold):
    return float((pred == gold).mean())


def freq_bucket(n):
    if n == 0:
        return 'unseen_0'
    if n <= 5:
        return 'rare_1_5'
    if n <= 15:
        return 'few_6_15'
    if n <= 50:
        return 'medium_16_50'
    return 'head_gt_50'


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dev = Dataset(args.dev_file, args.label_file, tokenizer)
    loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=device), drop_last=False)

    baseline = load_model(args.baseline_checkpoint, args.config_file, dev.num_labels,
                          'rope', 'last', device)
    base_logits, gold, sentence_ids = collect_logits(baseline, loader, device, 'baseline')
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    aware = load_model(args.aware_checkpoint, args.config_file, dev.num_labels,
                       'target_aware', args.aware_layers, device)
    aware_logits, gold2, sentence_ids2 = collect_logits(aware, loader, device, 'target-aware')
    if not np.array_equal(gold, gold2) or sentence_ids != sentence_ids2:
        raise RuntimeError('Checkpoint evaluations are not aligned.')

    base_prob, aware_prob = softmax(base_logits), softmax(aware_logits)
    base_pred, aware_pred = base_prob.argmax(1), aware_prob.argmax(1)
    grid = []
    best = None
    for alpha in np.linspace(0, 1, 41):
        # alpha is the baseline probability weight.
        pred = (alpha * base_prob + (1 - alpha) * aware_prob).argmax(1)
        row = {'baseline_weight': float(alpha), 'aware_weight': float(1-alpha), 'val_acc': acc(pred, gold)}
        grid.append(row)
        if best is None or row['val_acc'] > best['val_acc']:
            best = row
    ensemble_prob = best['baseline_weight'] * base_prob + best['aware_weight'] * aware_prob
    ensemble_pred = ensemble_prob.argmax(1)

    with open(os.path.join(args.output_dir, 'ensemble_grid.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['baseline_weight', 'aware_weight', 'val_acc'])
        w.writeheader(); w.writerows(grid)

    with codecs.open(args.train_file, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    train_freq = Counter(x['frame'] for x in train_data)
    idx2label = dev.idx2label
    bucket_stats = defaultdict(lambda: {'count': 0, 'baseline_correct': 0, 'aware_correct': 0, 'ensemble_correct': 0})
    rows = []
    for i, gold_idx in enumerate(gold):
        gold_name = idx2label[int(gold_idx)]
        bucket = freq_bucket(train_freq[gold_name])
        st = bucket_stats[bucket]
        st['count'] += 1
        st['baseline_correct'] += int(base_pred[i] == gold_idx)
        st['aware_correct'] += int(aware_pred[i] == gold_idx)
        st['ensemble_correct'] += int(ensemble_pred[i] == gold_idx)
        rows.append({
            'sentence_id': sentence_ids[i], 'gold': gold_name,
            'train_frequency': train_freq[gold_name], 'frequency_bucket': bucket,
            'baseline_pred': idx2label[int(base_pred[i])],
            'aware_pred': idx2label[int(aware_pred[i])],
            'ensemble_pred': idx2label[int(ensemble_pred[i])],
            'baseline_correct': int(base_pred[i] == gold_idx),
            'aware_correct': int(aware_pred[i] == gold_idx),
            'ensemble_correct': int(ensemble_pred[i] == gold_idx),
        })
    with open(os.path.join(args.output_dir, 'dev_predictions.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    bucket_rows = []
    for bucket in ['unseen_0', 'rare_1_5', 'few_6_15', 'medium_16_50', 'head_gt_50']:
        st = bucket_stats[bucket]
        n = st['count']
        bucket_rows.append({
            'bucket': bucket, 'count': n,
            'baseline_acc': st['baseline_correct'] / n if n else 0.0,
            'aware_acc': st['aware_correct'] / n if n else 0.0,
            'ensemble_acc': st['ensemble_correct'] / n if n else 0.0,
        })
    with open(os.path.join(args.output_dir, 'frequency_bucket_metrics.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        w.writeheader(); w.writerows(bucket_rows)

    summary = {
        'baseline_checkpoint': args.baseline_checkpoint,
        'aware_checkpoint': args.aware_checkpoint,
        'aware_layers': args.aware_layers,
        'baseline_acc': acc(base_pred, gold),
        'aware_acc': acc(aware_pred, gold),
        'best_ensemble': best,
        'baseline_only_correct': int(((base_pred == gold) & (aware_pred != gold)).sum()),
        'aware_only_correct': int(((aware_pred == gold) & (base_pred != gold)).sum()),
        'both_wrong': int(((base_pred != gold) & (aware_pred != gold)).sum()),
        'num_examples': int(len(gold)),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

