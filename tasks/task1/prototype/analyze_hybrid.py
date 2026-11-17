#!/usr/bin/env python3
"""Analyze Baseline + Frame-Prototype hybrids on task1 dev data."""
import argparse
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
from .model import FramePrototypeModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_checkpoint', required=True)
    p.add_argument('--prototype_checkpoint', required=True)
    p.add_argument('--prototype_file', default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--train_file', default='./dataset/cfn-train.json')
    p.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    p.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    p.add_argument('--output_dir', default='./tasks/task1/analysis/baseline-prototype-10ep')
    p.add_argument('--gamma_max', default=5.0, type=float)
    p.add_argument('--gamma_step', default=0.1, type=float)
    p.add_argument('--weight_step', default=0.025, type=float)
    return p.parse_args()


def collate(data, device=None):
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    return (
        torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device),
        torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device),
        [x[2] for x in data],
        torch.tensor([x[3] for x in data], dtype=torch.long, device=device),
        [x[4] for x in data],
    )


def load_baseline(args, num_labels, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_labels
    config.model_variant = 'rope'
    config.target_aware_layers = 'last'
    model = Model(config)
    msg = model.load_state_dict(torch.load(args.baseline_checkpoint, map_location='cpu'), strict=False)
    print(f'[baseline load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    return model.to(device)


def load_prototype(args, num_labels, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_labels
    model = FramePrototypeModel(config, args.prototype_file)
    msg = model.load_state_dict(torch.load(args.prototype_checkpoint, map_location='cpu'), strict=False)
    print(f'[prototype load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    return model.to(device)


def collect_logits(model, loader, device, kind):
    model.eval(); all_logits=[]; all_labels=[]; ids=[]
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, sentence_ids in tqdm(loader, desc=kind):
            if kind == 'baseline':
                out = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                            labels=None, device=device, for_test=True)
            else:
                out = model(input_ids, attention_mask, target, labels=None, for_test=True)
            all_logits.append(out['logits'].cpu())
            all_labels.append(labels.cpu()); ids.extend(sentence_ids)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy(), ids


def softmax(x):
    x = x - x.max(1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(1, keepdims=True)


def accuracy(pred, gold):
    return float((pred == gold).mean())


def topk_accuracy(scores, gold, k=5):
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean([gold[i] in top[i] for i in range(len(gold))]))


def bucket(freq):
    if freq == 0: return 'unseen_0'
    if freq <= 5: return 'rare_1_5'
    if freq <= 15: return 'few_6_15'
    if freq <= 50: return 'medium_16_50'
    return 'head_gt_50'


def evaluate_buckets(predictions, gold, idx2label, train_freq):
    stats = defaultdict(lambda: [0, 0])
    for pred, label in zip(predictions, gold):
        b = bucket(train_freq[idx2label[int(label)]])
        stats[b][0] += int(pred == label); stats[b][1] += 1
    return {b: {'correct': c, 'count': n, 'acc': c/n if n else 0.0} for b, (c,n) in stats.items()}


def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dev = Dataset(args.dev_file, args.frame_info_file, tokenizer)
    loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=device), drop_last=False)

    baseline = load_baseline(args, dev.num_labels, device)
    base_logits, gold, sentence_ids = collect_logits(baseline, loader, device, 'baseline')
    del baseline
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    prototype = load_prototype(args, dev.num_labels, device)
    proto_logits, gold2, ids2 = collect_logits(prototype, loader, device, 'prototype')
    if not np.array_equal(gold, gold2) or sentence_ids != ids2:
        raise RuntimeError('Baseline/prototype outputs are not aligned.')
    np.savez_compressed(os.path.join(args.output_dir, 'dev_logits.npz'),
                        baseline_logits=base_logits, prototype_logits=proto_logits,
                        gold=gold, sentence_ids=np.asarray(sentence_ids))

    train_rows = json.load(open(args.train_file, encoding='utf8'))
    train_freq = Counter(x['frame'] for x in train_rows)
    seen_mask = np.array([name in train_freq for name in dev.idx2label], dtype=bool)
    unseen_mask = ~seen_mask
    gold_unseen = unseen_mask[gold]

    base_prob = softmax(base_logits); base_pred = base_prob.argmax(1)
    proto_prob = softmax(proto_logits); proto_pred = proto_prob.argmax(1)

    gamma_rows=[]; best_gamma=None
    gamma_values = np.arange(0.0, args.gamma_max + args.gamma_step/2, args.gamma_step)
    for gamma in gamma_values:
        calibrated = proto_logits.copy(); calibrated[:, seen_mask] -= gamma
        pred = calibrated.argmax(1)
        row = {
            'gamma': float(gamma), 'overall_acc': accuracy(pred, gold),
            'seen_acc': accuracy(pred[~gold_unseen], gold[~gold_unseen]),
            'unseen_acc': accuracy(pred[gold_unseen], gold[gold_unseen]),
            'unseen_top5': topk_accuracy(calibrated[gold_unseen], gold[gold_unseen], 5),
        }
        gamma_rows.append(row)
        if best_gamma is None or row['overall_acc'] > best_gamma['overall_acc']:
            best_gamma = row

    grid=[]; best_joint=None
    weight_values = np.arange(0.0, 1.0 + args.weight_step/2, args.weight_step)
    for gamma in gamma_values:
        calibrated = proto_logits.copy(); calibrated[:, seen_mask] -= gamma
        calibrated_prob = softmax(calibrated)
        for base_weight in weight_values:
            scores = base_weight * base_prob + (1-base_weight) * calibrated_prob
            pred = scores.argmax(1)
            row = {'gamma': float(gamma), 'baseline_weight': float(base_weight),
                   'prototype_weight': float(1-base_weight), 'val_acc': accuracy(pred, gold)}
            grid.append(row)
            if best_joint is None or row['val_acc'] > best_joint['val_acc']:
                best_joint = row

    best_cal_logits = proto_logits.copy(); best_cal_logits[:, seen_mask] -= best_joint['gamma']
    best_scores = best_joint['baseline_weight']*base_prob + best_joint['prototype_weight']*softmax(best_cal_logits)
    best_pred = best_scores.argmax(1)
    cal_proto_logits = proto_logits.copy(); cal_proto_logits[:, seen_mask] -= best_gamma['gamma']
    cal_proto_pred = cal_proto_logits.argmax(1)

    base_top5 = np.argpartition(base_logits, -5, axis=1)[:, -5:]
    proto_top5 = np.argpartition(proto_logits, -5, axis=1)[:, -5:]
    union_hit = np.mean([gold[i] in set(base_top5[i]).union(set(proto_top5[i])) for i in range(len(gold))])

    summary = {
        'baseline_acc': accuracy(base_pred, gold), 'baseline_top5': topk_accuracy(base_logits, gold),
        'prototype_acc': accuracy(proto_pred, gold), 'prototype_top5': topk_accuracy(proto_logits, gold),
        'prototype_unseen_acc': accuracy(proto_pred[gold_unseen], gold[gold_unseen]),
        'prototype_unseen_top5': topk_accuracy(proto_logits[gold_unseen], gold[gold_unseen]),
        'best_prototype_calibration': best_gamma,
        'best_joint_hybrid': best_joint,
        'baseline_only_correct': int(((base_pred==gold)&(proto_pred!=gold)).sum()),
        'prototype_only_correct': int(((proto_pred==gold)&(base_pred!=gold)).sum()),
        'both_wrong': int(((proto_pred!=gold)&(base_pred!=gold)).sum()),
        'oracle_union_acc': float(np.mean((base_pred==gold)|(proto_pred==gold))),
        'top5_candidate_union_recall': float(union_hit), 'num_examples': int(len(gold)),
        'bucket_metrics': {
            'baseline': evaluate_buckets(base_pred,gold,dev.idx2label,train_freq),
            'prototype': evaluate_buckets(proto_pred,gold,dev.idx2label,train_freq),
            'calibrated_prototype': evaluate_buckets(cal_proto_pred,gold,dev.idx2label,train_freq),
            'best_hybrid': evaluate_buckets(best_pred,gold,dev.idx2label,train_freq),
        },
    }
    with open(os.path.join(args.output_dir,'summary.json'),'w',encoding='utf8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    for name, rows, fields in [
        ('prototype_calibration_grid.csv',gamma_rows,list(gamma_rows[0])),
        ('hybrid_grid.csv',grid,list(grid[0])),
    ]:
        with open(os.path.join(args.output_dir,name),'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

