#!/usr/bin/env python3
"""Refine B/D2 by injecting small C2/D weights; Dev analysis only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.analyze_pipeline_ensemble import build_gold_mask, evaluate, percentile_confidence, zscore_rows
from tasks.task3.dataset_experiments import build_augmented_frame_roles

TASK_DIR = REPO_ROOT / 'tasks/task3'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--c2_cache', default=str(TASK_DIR / 'analysis/task3-full-pipeline-B-C2-frame/pipeline_logits.npz'))
    p.add_argument('--d_cache', default=str(TASK_DIR / 'analysis/task3-full-pipeline-B-D-soft-top5/pipeline_logits.npz'))
    p.add_argument('--d2_epoch1_cache', default=str(TASK_DIR / 'analysis/task3-full-pipeline-B-D2-epoch1/pipeline_logits.npz'))
    p.add_argument('--d2_epoch2_cache', default=str(TASK_DIR / 'analysis/task3-full-pipeline-B-D2-epoch2/pipeline_logits.npz'))
    p.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset/cfn-dev.json'))
    p.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    p.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    p.add_argument('--output_dir', default=str(TASK_DIR / 'analysis/task3-D2-four-model-refine'))
    return p.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def better(row, best):
    return best is None or (row['exact_f1'], row['exact_recall']) > (best['exact_f1'], best['exact_recall'])


def main():
    a = parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    c2c, dc = np.load(a.c2_cache), np.load(a.d_cache)
    e1, e2 = np.load(a.d2_epoch1_cache), np.load(a.d2_epoch2_cache)
    for name, cache in [('D', dc), ('D2e1', e1), ('D2e2', e2)]:
        for field in ('instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names'):
            if not np.array_equal(c2c[field], cache[field]):
                raise ValueError(f'{name} differs in {field}')
    frames, roles, role2idx, frame_roles, augmented = build_augmented_frame_roles(
        a.frame_info_file, a.train_file)
    f2i = {row['frame_name']: i for i, row in enumerate(frames)}
    membership = np.zeros((len(frames), len(roles)), np.float32)
    for frame, ids in frame_roles.items(): membership[f2i[frame], ids] = 1
    probs = c2c['frame_probs'].astype(np.float32)
    log_prior = np.log(np.clip(probs @ membership, 1e-8, 1)).astype(np.float32)
    gold, boundary, ngold, nbound = build_gold_mask(
        read_json(a.dev_file), role2idx, c2c['instance_indices'], c2c['starts'],
        c2c['ends'], len(roles))
    b = zscore_rows(c2c['b_logits'].astype(np.float32))
    c2 = zscore_rows(c2c['c_logits'].astype(np.float32))
    d = zscore_rows(dc['c_logits'].astype(np.float32))
    d2s = {1: zscore_rows(e1['c_logits'].astype(np.float32)),
           2: zscore_rows(e2['c_logits'].astype(np.float32))}
    top2 = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
    confs = {
        'margin_percentile': percentile_confidence(top2[:, 1] - top2[:, 0]),
        'negative_entropy_percentile': percentile_confidence(
            (probs * np.log(probs.clip(min=1e-12))).sum(axis=1)),
    }
    rows, best_fixed = [], None
    ratios = (0.60, 0.65, 0.70)
    injections = (0.0, 0.05, 0.10, 0.15, 0.20)
    for epoch, d2 in d2s.items():
        for b_ratio in ratios:
            for c2w in injections:
                for dw in injections:
                    base = 1.0 - c2w - dw
                    bw, d2w = base * b_ratio, base * (1.0 - b_ratio)
                    combined = bw*b + c2w*c2 + dw*d + d2w*d2
                    for prior in np.round(np.arange(0, 2.0001, .05), 6):
                        pred = (combined + prior*log_prior).argmax(1)
                        row = {'strategy':'fixed','epoch':epoch,'b_weight':bw,
                               'c2_weight':c2w,'d_weight':dw,'d2_weight':d2w,
                               'confidence':'','low':float(prior),'high':float(prior),'power':0.0,
                               **evaluate(pred,gold,boundary,ngold,nbound)}
                        rows.append(row)
                        if better(row,best_fixed): best_fixed=row
    candidates=[]; seen=set()
    for row in sorted(rows,key=lambda x:x['exact_f1'],reverse=True):
        key=(row['epoch'],row['b_weight'],row['c2_weight'],row['d_weight'],row['d2_weight'])
        if key not in seen: candidates.append(key); seen.add(key)
        if len(candidates)==6: break
    best_adaptive=None
    for epoch,bw,c2w,dw,d2w in candidates:
        combined=bw*b+c2w*c2+dw*d+d2w*d2s[epoch]
        for cname,conf in confs.items():
            for low in (0,.2,.4,.6,.8,1.0):
                for high in (.6,.8,1,1.2,1.4,1.6,1.8,2):
                    if high<low: continue
                    for power in (.5,1,2):
                        prior=low+(high-low)*np.power(conf,power)
                        pred=(combined+prior[:,None]*log_prior).argmax(1)
                        row={'strategy':'adaptive','epoch':epoch,'b_weight':bw,
                             'c2_weight':c2w,'d_weight':dw,'d2_weight':d2w,
                             'confidence':cname,'low':low,'high':high,'power':power,
                             **evaluate(pred,gold,boundary,ngold,nbound)}
                        rows.append(row)
                        if better(row,best_adaptive): best_adaptive=row
    best=best_adaptive if better(best_adaptive,best_fixed) else best_fixed
    with (out/'grid.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'best_fixed':best_fixed,'best_adaptive':best_adaptive,'best_overall':best,
             'adaptive_candidate_weights':[{'epoch':x[0],'B':x[1],'C2':x[2],'D':x[3],'D2':x[4]} for x in candidates],
             'reference_B_D2_epoch1':{'tp':3803,'f1':0.6092598526113425,
                 'weights':{'B':.65,'D2':.35},'prior':{'low':.4,'high':1.8,'power':1}},
             'augmented_frame_role_licenses':augmented,'run_config':vars(a)}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
