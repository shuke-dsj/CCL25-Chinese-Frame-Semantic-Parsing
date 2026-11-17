#!/usr/bin/env python3
"""Search robust Task1→Task3 frame-guided decoding on cached Dev logits.

Hard top-1 masking is an oracle upper bound but can amplify Task1 errors.  This
script compares three inexpensive alternatives without retraining Task3:

1. top-k frame candidate union;
2. finite penalties for roles outside the top-k frame union;
3. a soft legal-role prior marginalised over the full Task1 distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks.task3.frame_utils import load_frame_ontology
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.common.experiment_utils import REPO_ROOT


TASK_DIR = REPO_ROOT / "tasks" / "task3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task1_logits", default=str(REPO_ROOT / "tasks" / "task1" / "analysis" / "baseline-prototype-10ep" / "dev_logits.npz"))
    p.add_argument("--task3_logits", default=str(TASK_DIR / "analysis" / "task3-baseline-audit" / "dev_logits.npz"))
    p.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset" / "frame_info.json"))
    p.add_argument("--train_file", default=str(REPO_ROOT / "dataset" / "cfn-train.json"))
    p.add_argument("--dev_file", default=str(REPO_ROOT / "dataset" / "cfn-dev.json"))
    p.add_argument("--prototype_gamma", default=2.7, type=float)
    p.add_argument("--baseline_weight", default=0.5, type=float)
    p.add_argument("--topk_values", default="1,2,3,5,10")
    p.add_argument("--penalties", default="0,0.25,0.5,0.75,1,1.5,2,3,4,5,7.5,10,20")
    p.add_argument("--prior_weights", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5")
    p.add_argument("--output_dir", default=str(TASK_DIR / "analysis" / "task3-frame-guidance-sweep"))
    return p.parse_args()


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(scores, labels, frame_correct=None):
    pred = scores.argmax(axis=1)
    top5 = np.argpartition(scores, -min(5, scores.shape[1]), axis=1)[:, -min(5, scores.shape[1]):]
    hit = pred == labels
    out = {
        "acc": float(hit.mean()),
        "top5_acc": float(np.mean([labels[i] in top5[i] for i in range(len(labels))])),
        "correct": int(hit.sum()),
        "count": int(len(labels)),
    }
    if frame_correct is not None:
        out["task1_frame_correct_count"] = int(frame_correct.sum())
        out["acc_when_task1_frame_correct"] = float(hit[frame_correct].mean()) if frame_correct.any() else 0.0
        wrong = ~frame_correct
        out["acc_when_task1_frame_wrong"] = float(hit[wrong].mean()) if wrong.any() else 0.0
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the exact candidate inventory used by Experiments B/C.  The
    # official ontology misses one Train-observed frame/FE licence; omitting
    # that augmentation here would make post-training decoding inconsistent
    # with masked-CE training.
    frame_info, role_names, role2idx, frame_to_role_ids, augmented_licenses = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [x["frame_name"] for x in frame_info]
    frame2idx = {name: i for i, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0

    train = json.load(open(args.train_file, encoding="utf8"))
    seen_frames = {x["frame"] for x in train}
    seen_mask = np.asarray([name in seen_frames for name in frame_names], dtype=bool)
    dev = json.load(open(args.dev_file, encoding="utf8"))
    # Legacy audit caches excluded target-overlap spans and did not save source
    # instance indices. New B/C caches include them explicitly and retain all
    # 5947 Dev annotations. Support both formats without using lossy SID maps.
    legacy_instance_indices = []
    legacy_expected_sids = []
    for instance_idx, line in enumerate(dev):
        target = line["target"][-1]
        for span in line.get("cfn_spans", []):
            if not (span["end"] < target["start"] or span["start"] > target["end"]):
                continue
            legacy_instance_indices.append(instance_idx)
            legacy_expected_sids.append(int(line["sentence_id"]))

    t1 = np.load(args.task1_logits)
    base = t1["baseline_logits"].astype(np.float32)
    proto = t1["prototype_logits"].astype(np.float32).copy()
    proto[:, seen_mask] -= args.prototype_gamma
    base_weight = args.baseline_weight
    frame_prob = base_weight * softmax(base) + (1.0 - base_weight) * softmax(proto)
    t1_sids = t1["sentence_ids"].astype(np.int64)

    t3 = np.load(args.task3_logits)
    logits = t3["logits"].astype(np.float32)
    labels = t3["labels"].astype(np.int64)
    sids = t3["sentence_ids"].astype(np.int64)
    if 'instance_indices' in t3.files:
        t1_rows = t3['instance_indices'].astype(np.int64)
        expected_sids = np.asarray(
            [int(dev[index]['sentence_id']) for index in t1_rows], dtype=np.int64
        )
    else:
        t1_rows = np.asarray(legacy_instance_indices, dtype=np.int64)
        expected_sids = np.asarray(legacy_expected_sids, dtype=np.int64)
    if len(t1_rows) != len(sids) or not np.array_equal(sids, expected_sids):
        raise RuntimeError('Task1/Task3 cached Dev rows are not aligned by source instance order.')
    if not np.array_equal(t1_sids, np.asarray([int(x["sentence_id"]) for x in dev])):
        raise RuntimeError('Task1 logits do not follow cfn-dev instance order.')
    sample_frame_prob = frame_prob[t1_rows]
    gold_frames = np.asarray([frame2idx[dev[i]["frame"]] for i in t1_rows], dtype=np.int64)
    top1_frames = sample_frame_prob.argmax(axis=1)
    frame_correct = top1_frames == gold_frames

    oracle_mask = membership[gold_frames].astype(bool)
    hard_oracle = np.where(oracle_mask, logits, -1e9)
    global_result = metrics(logits, labels, frame_correct)
    oracle_result = metrics(hard_oracle, labels, frame_correct)

    topk_values = [int(x) for x in args.topk_values.split(",") if x]
    penalties = [float(x) for x in args.penalties.split(",") if x]
    prior_weights = [float(x) for x in args.prior_weights.split(",") if x]
    max_k = max(topk_values)
    top_frames = np.argpartition(sample_frame_prob, -max_k, axis=1)[:, -max_k:]
    # Sort only the selected candidates to make nested top-k sets deterministic.
    selected_prob = np.take_along_axis(sample_frame_prob, top_frames, axis=1)
    order = np.argsort(-selected_prob, axis=1)
    top_frames = np.take_along_axis(top_frames, order, axis=1)

    grid = []
    best_penalty = None
    hard_topk = {}
    for k in topk_values:
        candidate_mask = membership[top_frames[:, :k]].max(axis=1).astype(bool)
        hard_score = np.where(candidate_mask, logits, -1e9)
        hard_topk[str(k)] = {
            **metrics(hard_score, labels, frame_correct),
            "mean_candidate_roles": float(candidate_mask.sum(axis=1).mean()),
            "gold_frame_recall": float(np.any(top_frames[:, :k] == gold_frames[:, None], axis=1).mean()),
        }
        for penalty in penalties:
            score = logits - penalty * (~candidate_mask)
            row = {
                "strategy": "topk_finite_penalty", "topk": k, "penalty": penalty,
                "prior_weight": "", "mean_candidate_roles": float(candidate_mask.sum(axis=1).mean()),
                "gold_frame_recall": float(np.any(top_frames[:, :k] == gold_frames[:, None], axis=1).mean()),
                **metrics(score, labels, frame_correct),
            }
            grid.append(row)
            if best_penalty is None or row["acc"] > best_penalty["acc"]:
                best_penalty = row

    # p(role is legal | x) = sum_frame p(frame|x) I(role in FE(frame)).
    legal_prior_by_sentence = frame_prob @ membership
    sample_legal_prior = legal_prior_by_sentence[t1_rows]
    log_prior = np.log(np.clip(sample_legal_prior, 1e-8, 1.0))
    best_prior = None
    for weight in prior_weights:
        score = logits + weight * log_prior
        row = {
            "strategy": "marginal_legal_role_prior", "topk": "", "penalty": "",
            "prior_weight": weight, "mean_candidate_roles": "", "gold_frame_recall": "",
            **metrics(score, labels, frame_correct),
        }
        grid.append(row)
        if best_prior is None or row["acc"] > best_prior["acc"]:
            best_prior = row

    with (out_dir / "grid.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader(); writer.writerows(grid)

    summary = {
        "task1_hybrid": {
            "prototype_gamma": args.prototype_gamma,
            "baseline_weight": base_weight,
            "prototype_weight": 1.0 - base_weight,
            "sentence_frame_acc": float((frame_prob.argmax(1) == t1["gold"]).mean()),
        },
        "global_task3": global_result,
        "oracle_frame_hard_mask": oracle_result,
        "predicted_frame_hard_topk": hard_topk,
        "best_finite_penalty": best_penalty,
        "best_marginal_legal_role_prior": best_prior,
        "num_argument_examples": int(len(labels)),
        "augmented_frame_role_licenses": augmented_licenses,
        "run_config": vars(args),
    }
    with (out_dir / "summary.json").open("w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
