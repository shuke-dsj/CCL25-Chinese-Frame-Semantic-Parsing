#!/usr/bin/env python3
"""I0: audit confidence-based rejection on the best B+C2+D2 Dev pipeline.

This is deliberately a zero-training diagnostic.  It answers whether the
existing role ensemble already exposes a confidence signal that can remove
false Task2 boundaries / wrong roles without sacrificing more than a small
amount of Task3 recall.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks.common.experiment_utils import REPO_ROOT
TASK_DIR = REPO_ROOT / "tasks" / "task3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c2_cache",
        default=str(TASK_DIR / "analysis/task3-full-pipeline-B-C2-frame/pipeline_logits.npz"),
    )
    parser.add_argument(
        "--d2_cache",
        default=str(TASK_DIR / "analysis/task3-full-pipeline-B-D2-epoch1/pipeline_logits.npz"),
    )
    parser.add_argument("--dev_file", default=str(REPO_ROOT / "dataset/cfn-dev.json"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    parser.add_argument("--b_weight", default=0.585, type=float)
    parser.add_argument("--c2_weight", default=0.100, type=float)
    parser.add_argument("--d2_weight", default=0.315, type=float)
    parser.add_argument("--prior_low", default=0.6, type=float)
    parser.add_argument("--prior_high", default=1.8, type=float)
    parser.add_argument("--prior_power", default=2.0, type=float)
    parser.add_argument(
        "--max_recall_drop", default=0.003, type=float,
        help="Absolute recall drop allowed when selecting a precision-oriented threshold.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(TASK_DIR / "analysis/task3-I0-precision-rejection"),
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)


def zscore_rows(logits):
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / centered.std(axis=1, keepdims=True).clip(min=1e-6)


def percentile_confidence(values):
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.ones_like(values, dtype=np.float32)
    return (
        np.searchsorted(unique, values, side="right").astype(np.float32) - 1.0
    ) / float(len(unique) - 1)


def build_augmented_frame_roles(frame_info_file, train_file):
    frame_info = read_json(frame_info_file)
    role_names, role2idx, frame_to_role_ids = [], {}, {}
    for frame in frame_info:
        ids = []
        for fe in frame.get("fes", []):
            role = fe["fe_name"]
            if role not in role2idx:
                role2idx[role] = len(role_names)
                role_names.append(role)
            role_id = role2idx[role]
            if role_id not in ids:
                ids.append(role_id)
        frame_to_role_ids[frame["frame_name"]] = ids
    added = []
    for row in read_json(train_file):
        licensed = frame_to_role_ids[row["frame"]]
        for span in row.get("cfn_spans", []):
            role_id = role2idx[span["fe_name"]]
            if role_id not in licensed:
                licensed.append(role_id)
                added.append({
                    "sentence_id": int(row["sentence_id"]),
                    "frame": row["frame"],
                    "role": span["fe_name"],
                })
    return frame_info, role_names, role2idx, frame_to_role_ids, added


def build_gold_mask(dev_rows, role2idx, instance_indices, starts, ends, num_roles):
    gold_by_boundary = {}
    gold_tuples = 0
    for instance_index, row in enumerate(dev_rows):
        for span in row.get("cfn_spans", []):
            key = (instance_index, int(span["start"]), int(span["end"]))
            roles = gold_by_boundary.setdefault(key, set())
            role_id = int(role2idx[span["fe_name"]])
            if role_id not in roles:
                roles.add(role_id)
                gold_tuples += 1
    mask = np.zeros((len(instance_indices), num_roles), dtype=bool)
    boundary_match = np.zeros(len(instance_indices), dtype=bool)
    for index, key in enumerate(zip(instance_indices, starts, ends)):
        roles = gold_by_boundary.get(tuple(int(value) for value in key), ())
        if roles:
            boundary_match[index] = True
            mask[index, list(roles)] = True
    return mask, boundary_match, gold_tuples, len(gold_by_boundary)


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def check_alignment(reference, other):
    for key in ("instance_indices", "sentence_ids", "starts", "ends", "role_names"):
        if key not in reference.files or key not in other.files:
            raise ValueError(f"Pipeline cache is missing {key!r}.")
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"C2/D2 pipeline caches are not aligned on {key}.")


def metrics_for_keep(keep, correct_tuple, boundary_match, gold_tuples, gold_boundaries):
    predicted = int(keep.sum())
    tp = int((keep & correct_tuple).sum())
    boundary_tp = int((keep & boundary_match).sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold_tuples if gold_tuples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "retained_predictions": predicted,
        "filtered_predictions": int(len(keep) - predicted),
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_tp": boundary_tp,
        "boundary_precision": boundary_tp / predicted if predicted else 0.0,
        "boundary_recall": boundary_tp / gold_boundaries if gold_boundaries else 0.0,
        "boundary_false_positive_filtered": int(((~boundary_match) & (~keep)).sum()),
        "wrong_role_filtered": int((boundary_match & (~correct_tuple) & (~keep)).sum()),
        "correct_tuple_filtered": int((correct_tuple & (~keep)).sum()),
    }


def sweep_feature(name, values, correct_tuple, boundary_match, gold_tuples, gold_boundaries):
    """Retain rows with a high feature value and evaluate every useful cutoff."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Feature {name} contains NaN/Inf.")
    order = np.argsort(-values, kind="stable")
    candidates = [0]
    # Threshold only after the final member of a tied score group.
    candidates.extend((np.flatnonzero(values[order][:-1] != values[order][1:]) + 1).tolist())
    candidates.append(len(values))
    rows = []
    for retained in candidates:
        keep = np.zeros(len(values), dtype=bool)
        keep[order[:retained]] = True
        threshold = float("inf") if retained == 0 else float(values[order[retained - 1]])
        rows.append({
            "feature": name,
            "threshold": threshold,
            **metrics_for_keep(
                keep, correct_tuple, boundary_match, gold_tuples, gold_boundaries
            ),
        })
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c2 = np.load(args.c2_cache, allow_pickle=True)
    d2 = np.load(args.d2_cache, allow_pickle=True)
    check_alignment(c2, d2)

    if not np.isclose(args.b_weight + args.c2_weight + args.d2_weight, 1.0):
        raise ValueError("B/C2/D2 weights must sum to 1.")
    role_names = c2["role_names"].tolist()
    b = zscore_rows(c2["b_logits"].astype(np.float32))
    c2_logits = zscore_rows(c2["c_logits"].astype(np.float32))
    d2_logits = zscore_rows(d2["c_logits"].astype(np.float32))
    frame_probs = c2["frame_probs"].astype(np.float32)

    frame_info, ontology_roles, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    if ontology_roles != role_names:
        raise ValueError("Pipeline role order differs from frame_info ontology.")
    frame2idx = {row["frame_name"]: i for i, row in enumerate(frame_info)}
    membership = np.zeros((len(frame_info), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0
    legal_prior = np.clip(frame_probs @ membership, 1e-8, 1.0)

    top2_frame = np.partition(frame_probs, -2, axis=1)[:, -2:]
    frame_margin = top2_frame[:, 1] - top2_frame[:, 0]
    frame_confidence = percentile_confidence(frame_margin)
    prior_weight = args.prior_low + (args.prior_high - args.prior_low) * (
        frame_confidence ** args.prior_power
    )
    combined = args.b_weight * b + args.c2_weight * c2_logits + args.d2_weight * d2_logits
    final_scores = combined + prior_weight[:, None] * np.log(legal_prior)
    prediction = final_scores.argmax(axis=1)

    dev_rows = read_json(args.dev_file)
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        dev_rows,
        role2idx,
        c2["instance_indices"].astype(np.int64),
        c2["starts"].astype(np.int64),
        c2["ends"].astype(np.int64),
        len(role_names),
    )
    row_indices = np.arange(len(prediction))
    correct_tuple = gold_mask[row_indices, prediction]

    probabilities = softmax(final_scores)
    sorted_scores = np.partition(final_scores, -2, axis=1)[:, -2:]
    role_margin = sorted_scores[:, 1] - sorted_scores[:, 0]
    entropy = -(probabilities * np.log(probabilities.clip(min=1e-12))).sum(axis=1)
    individual_prediction = np.stack(
        [b.argmax(axis=1), c2_logits.argmax(axis=1), d2_logits.argmax(axis=1)], axis=1
    )
    agreement = (individual_prediction == prediction[:, None]).mean(axis=1)
    features = {
        "final_top1_score": final_scores[row_indices, prediction],
        "role_margin": role_margin,
        "role_max_probability": probabilities[row_indices, prediction],
        "negative_role_entropy": -entropy,
        "model_agreement": agreement,
        "frame_margin": frame_margin,
        "negative_frame_entropy": (
            frame_probs * np.log(frame_probs.clip(min=1e-12))
        ).sum(axis=1),
        "predicted_role_legal_prior": legal_prior[row_indices, prediction],
    }

    baseline = metrics_for_keep(
        np.ones(len(prediction), dtype=bool),
        correct_tuple,
        boundary_match,
        gold_tuples,
        gold_boundaries,
    )
    all_rows = []
    best_by_feature = {}
    minimum_recall = baseline["recall"] - args.max_recall_drop
    for name, values in features.items():
        rows = sweep_feature(
            name, values, correct_tuple, boundary_match, gold_tuples, gold_boundaries
        )
        all_rows.extend(rows)
        eligible = [row for row in rows if row["recall"] >= minimum_recall]
        best_by_feature[name] = max(
            eligible,
            key=lambda row: (row["precision"], row["f1"], row["retained_predictions"]),
        )

    best_constrained = max(
        best_by_feature.values(),
        key=lambda row: (row["precision"], row["f1"], row["retained_predictions"]),
    )
    best_f1 = max(all_rows, key=lambda row: (row["f1"], row["precision"]))
    fieldnames = list(all_rows[0])
    with (output_dir / "feature_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    false_boundary = ~boundary_match
    incorrect_tuple = ~correct_tuple
    oracle_boundary_keep = ~false_boundary
    oracle_tuple_keep = ~incorrect_tuple
    summary = {
        "reference_ensemble": {
            "weights": {"B": args.b_weight, "C2": args.c2_weight, "D2": args.d2_weight},
            "adaptive_prior": {
                "confidence": "margin_percentile",
                "low": args.prior_low,
                "high": args.prior_high,
                "power": args.prior_power,
            },
            **baseline,
        },
        "recall_constraint": {
            "max_absolute_drop": args.max_recall_drop,
            "minimum_recall": minimum_recall,
        },
        "best_precision_under_recall_constraint": best_constrained,
        "best_by_feature_under_constraint": best_by_feature,
        "best_unconstrained_f1": best_f1,
        "oracle_remove_false_boundaries": metrics_for_keep(
            oracle_boundary_keep, correct_tuple, boundary_match, gold_tuples, gold_boundaries
        ),
        "oracle_remove_all_incorrect_tuples": metrics_for_keep(
            oracle_tuple_keep, correct_tuple, boundary_match, gold_tuples, gold_boundaries
        ),
        "error_counts": {
            "correct_tuple": int(correct_tuple.sum()),
            "matched_boundary_wrong_role": int((boundary_match & ~correct_tuple).sum()),
            "false_boundary": int(false_boundary.sum()),
        },
        "augmented_frame_role_licenses": augmented,
        "run_config": vars(args),
    }
    with (output_dir / "summary.json").open("w", encoding="utf8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
