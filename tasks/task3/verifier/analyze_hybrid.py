#!/usr/bin/env python3
"""Combine I1 validity confidence with the strong B+C2+D2 role ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.analyze_precision_rejection import (
    build_augmented_frame_roles,
    build_gold_mask,
    metrics_for_keep,
    percentile_confidence,
    read_json,
    zscore_rows,
)


TASK_DIR = REPO_ROOT / "tasks/task3"


def sweep_feature_grid(
    name, values, correct_tuple, boundary_match, gold_tuples, gold_boundaries
):
    """Use a dense retained-count grid; exact all-unique sweeps are unnecessary here."""
    values = np.asarray(values)
    order = np.argsort(-values, kind="stable")
    retained_counts = np.unique(np.linspace(0, len(values), 501).round().astype(int))
    retained_counts = np.unique(np.concatenate([retained_counts, [len(values)]]))
    rows = []
    for retained in retained_counts:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2_cache", default=str(
        TASK_DIR / "analysis/task3-full-pipeline-B-C2-frame/pipeline_logits.npz"
    ))
    parser.add_argument("--d2_cache", default=str(
        TASK_DIR / "analysis/task3-full-pipeline-B-D2-epoch1/pipeline_logits.npz"
    ))
    parser.add_argument("--verifier_caches", default=",".join(str(
        TASK_DIR / "experiment_record/task3-I1B-frame-conditioned-verifier-neg2-3ep-result"
        / f"dev_pipeline_logits_epoch_{epoch}.npz"
    ) for epoch in (1, 2, 3)))
    parser.add_argument("--dev_file", default=str(REPO_ROOT / "dataset/cfn-dev.json"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    parser.add_argument("--max_recall_drop", default=0.003, type=float)
    parser.add_argument("--output_dir", default=str(
        TASK_DIR / "analysis/task3-I1B-validity-B-C2-D2-hybrid"
    ))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c2, d2 = np.load(args.c2_cache), np.load(args.d2_cache)
    for key in ("instance_indices", "sentence_ids", "starts", "ends", "role_names"):
        if not np.array_equal(c2[key], d2[key]):
            raise ValueError(f"C2/D2 cache mismatch on {key}.")
    frame_info, roles, role2idx, frame_roles, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    if roles != c2["role_names"].tolist():
        raise ValueError("Role ontology differs from pipeline cache.")
    frame2idx = {row["frame_name"]: index for index, row in enumerate(frame_info)}
    membership = np.zeros((len(frame_info), len(roles)), dtype=np.float32)
    for frame, role_ids in frame_roles.items():
        membership[frame2idx[frame], role_ids] = 1.0
    frame_probs = c2["frame_probs"].astype(np.float32)
    legal_prior = np.clip(frame_probs @ membership, 1e-8, 1.0)
    frame_top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    frame_margin_rank = percentile_confidence(frame_top2[:, 1] - frame_top2[:, 0])
    prior_weight = 0.6 + 1.2 * frame_margin_rank ** 2
    role_scores = (
        0.585 * zscore_rows(c2["b_logits"].astype(np.float32))
        + 0.100 * zscore_rows(c2["c_logits"].astype(np.float32))
        + 0.315 * zscore_rows(d2["c_logits"].astype(np.float32))
        + prior_weight[:, None] * np.log(legal_prior)
    )
    prediction = role_scores.argmax(axis=1)
    indices = np.arange(len(prediction))
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        read_json(args.dev_file), role2idx,
        c2["instance_indices"], c2["starts"], c2["ends"], len(roles),
    )
    correct_tuple = gold_mask[indices, prediction]
    reference = metrics_for_keep(
        np.ones(len(prediction), dtype=bool), correct_tuple, boundary_match,
        gold_tuples, gold_boundaries,
    )
    minimum_recall = reference["recall"] - args.max_recall_drop
    final_top1_rank = percentile_confidence(role_scores[indices, prediction])

    verifier_paths = [Path(value) for value in args.verifier_caches.split(",") if value]
    all_rows, summaries = [], []
    for verifier_path in verifier_paths:
        verifier = np.load(verifier_path)
        for key, pipeline_key in (
            ("instance_indices", "instance_indices"), ("sentence_ids", "sentence_ids"),
            ("starts", "starts"), ("ends", "ends"),
        ):
            if not np.array_equal(verifier[key], c2[pipeline_key]):
                raise ValueError(f"Verifier cache {verifier_path} differs on {key}.")
        validity = verifier["validity_probabilities"].astype(np.float32)
        validity_rank = percentile_confidence(validity)
        features = {"validity_only": validity_rank}
        for verifier_weight in np.linspace(0.0, 1.0, 11):
            features[f"joint_rank_v{verifier_weight:.2f}"] = (
                verifier_weight * validity_rank
                + (1.0 - verifier_weight) * final_top1_rank
            )
        epoch_rows = []
        for feature, values in features.items():
            rows = sweep_feature_grid(
                f"{verifier_path.stem}:{feature}", values, correct_tuple,
                boundary_match, gold_tuples, gold_boundaries,
            )
            epoch_rows.extend(rows)
            all_rows.extend(rows)
        eligible = [row for row in epoch_rows if row["recall"] >= minimum_recall]
        summaries.append({
            "cache": str(verifier_path),
            "best_precision_under_recall_constraint": max(
                eligible, key=lambda row: (row["precision"], row["f1"])
            ),
            "best_f1": max(epoch_rows, key=lambda row: (row["f1"], row["precision"])),
        })
    best_constrained = max(
        (item["best_precision_under_recall_constraint"] for item in summaries),
        key=lambda row: (row["precision"], row["f1"]),
    )
    best_f1 = max(
        (item["best_f1"] for item in summaries),
        key=lambda row: (row["f1"], row["precision"]),
    )
    with (output_dir / "grid.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "reference_B_C2_D2": reference,
        "recall_constraint": {
            "max_drop": args.max_recall_drop, "minimum_recall": minimum_recall,
        },
        "per_verifier_cache": summaries,
        "best_overall_under_recall_constraint": best_constrained,
        "best_overall_f1": best_f1,
        "augmented_frame_role_licenses": augmented,
        "run_config": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
