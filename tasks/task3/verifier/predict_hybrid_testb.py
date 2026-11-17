#!/usr/bin/env python3
"""Generate the final Test-B B+C2+D2 + calibrated I1B-R rejection submission.

No Test-A model/data is evaluated.  Historical A JSON files are copied
byte-for-byte only because Tianchi expects the legacy six-file zip layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.archive.prediction.predict_b_c2_d2_ensemble import (
    percentile_confidence,
    zscore_rows,
)
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.predict_soft_frame_prior import SoftPriorDataset
from tasks.task3.verifier.dataset import encode_with_target_argument_markers
from tasks.task3.verifier.model import FrameConditionedSpanRoleVerifier


TASK_DIR = REPO_ROOT / "tasks/task3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default=str(REPO_ROOT / "dataset/cfn-test-B.json"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    parser.add_argument("--pretrained_model_dir", default=str(REPO_ROOT / "chinese_bert_wwm_ext"))
    parser.add_argument("--task1_probability_file", default=str(
        REPO_ROOT / "tasks/task1/predictions/prototype-hybrid-soft-prior/B_task1_frame_probs.npz"
    ))
    parser.add_argument("--task1_submission_file", default=str(
        REPO_ROOT / "tasks/task1/predictions/prototype-hybrid-soft-prior/B_task1_test.json"
    ))
    parser.add_argument("--task2_prediction_file", default=str(
        REPO_ROOT / "tasks/task2/predictions/task2-BDEH-centered-040-020-020-020-qm034/B_task2_test.json"
    ))
    parser.add_argument("--role_logits_cache", default=str(
        TASK_DIR / "predictions/task3-B-C2-D2-e1-zscore-adaptive/test_logits.npz"
    ))
    parser.add_argument("--reference_task3_file", default=str(
        TASK_DIR / "predictions/task3-B-C2-D2-e1-zscore-adaptive/B_task3_test.json"
    ))
    parser.add_argument("--verifier_checkpoint", default=str(
        TASK_DIR / "experiment_record/task3-I1B-R-hardmix-pw15-1ep-result/checkpoints/task3_verifier_best.bin"
    ))
    parser.add_argument("--b_weight", default=0.585, type=float)
    parser.add_argument("--c2_weight", default=0.100, type=float)
    parser.add_argument("--d2_weight", default=0.315, type=float)
    parser.add_argument("--prior_low", default=0.6, type=float)
    parser.add_argument("--prior_high", default=1.8, type=float)
    parser.add_argument("--prior_power", default=2.0, type=float)
    parser.add_argument("--prior_epsilon", default=1e-8, type=float)
    parser.add_argument("--ensemble_rank_weight", default=0.9, type=float)
    parser.add_argument("--validity_rank_weight", default=0.1, type=float)
    parser.add_argument("--dev_retained", default=6406, type=int)
    parser.add_argument("--dev_total", default=6537, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--historical_a_dir", default=str(
        TASK_DIR / "predictions/submission-task3-C-w300"
    ))
    parser.add_argument("--output_dir", default=str(
        TASK_DIR / "predictions/task3-B-C2-D2-I1BR-rank-reject2pct"
    ))
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifierTestDataset(Dataset):
    def __init__(self, soft_prior_dataset, tokenizer, max_length=512):
        self.items = soft_prior_dataset.items
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        input_ids, attention_mask, marker_positions = encode_with_target_argument_markers(
            self.tokenizer,
            row["text"],
            row["target_start"], row["target_end"],
            row["start"], row["end"],
            self.max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_positions": marker_positions,
            "frame_probabilities": row["frame_prob"],
            "row": row,
        }


def collate(items, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in items)
    input_ids, attention_masks = [], []
    for item in items:
        padding = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(item["attention_mask"] + [0] * padding)
    positions = {
        name: torch.tensor(
            [item["marker_positions"][name] for item in items], dtype=torch.long
        )
        for name in ("target_start", "target_end", "argument_start", "argument_end")
    }
    frame_probabilities = torch.tensor(
        np.stack([item["frame_probabilities"] for item in items]),
        dtype=torch.float32,
    )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "marker_positions": positions,
        "frame_probabilities": frame_probabilities,
        "frame_ids": frame_probabilities.argmax(dim=1),
        "rows": [item["row"] for item in items],
    }


def to_device(batch, device):
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "frame_ids": batch["frame_ids"].to(device),
        "frame_probabilities": batch["frame_probabilities"].to(device),
        "marker_positions": {
            key: value.to(device) for key, value in batch["marker_positions"].items()
        },
    }


def load_verifier(args, num_roles, num_frames, device):
    checkpoint_path = Path(args.verifier_checkpoint)
    with checkpoint_path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu")
    run = checkpoint.get("run_config", {})
    if checkpoint.get("model_variant") != "frame_conditioned_verifier":
        raise ValueError("The final predictor requires an I1B-R frame-conditioned checkpoint.")
    config = BertConfig.from_pretrained(args.pretrained_model_dir)
    model = FrameConditionedSpanRoleVerifier(
        config,
        num_roles=num_roles,
        num_frames=num_frames,
        variant="frame_conditioned_verifier",
        projection_dim=int(run.get("projection_dim", 512)),
        frame_dim=int(run.get("frame_dim", 128)),
        frame_condition_dropout=float(run.get("frame_condition_dropout", 0.2)),
    )
    message = model.load_state_dict(checkpoint["model_state"], strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"Verifier checkpoint mismatch: {message}")
    return model.to(device).eval(), checkpoint


def collect_validity(model, loader, device):
    logits, rows = [], []
    with torch.no_grad():
        for cpu_batch in tqdm(loader, desc="I1B-R-TestB-validity"):
            batch = to_device(cpu_batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                marker_positions=batch["marker_positions"],
                frame_ids=batch["frame_ids"],
                frame_probabilities=batch["frame_probabilities"],
            )
            logits.append(output["validity_logits"].cpu().float().numpy())
            rows.extend(batch["rows"])
    return np.concatenate(logits), rows


def write_json_rows(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf8", newline="\n") as handle:
        handle.write(
            "[\n" + ",\n".join(
                "   " + json.dumps(row, ensure_ascii=False) for row in rows
            ) + "\n]"
        )


def package_submission(args, output_dir, task3_file):
    submission_dir = output_dir / "submission_files"
    submission_dir.mkdir(parents=True, exist_ok=True)
    split = "A" if Path(args.test_file).name == "cfn-test-A.json" else "B"
    if split == "A":
        sources = {
            "A_task1_test.json": Path(args.task1_submission_file),
            "A_task2_test.json": Path(args.task2_prediction_file),
            "A_task3_test.json": task3_file,
        }
        zip_path = output_dir / "submission-task3-testA-I1BR.zip"
    else:
        sources = {
            "A_task1_test.json": Path(args.historical_a_dir) / "A_task1_test.json",
            "A_task2_test.json": Path(args.historical_a_dir) / "A_task2_test.json",
            "A_task3_test.json": Path(args.historical_a_dir) / "A_task3_test.json",
            "B_task1_test.json": Path(args.task1_submission_file),
            "B_task2_test.json": Path(args.task2_prediction_file),
            "B_task3_test.json": task3_file,
        }
        zip_path = output_dir / "submission-task3-B-C2-D2-I1BR-rank-reject2pct.zip"
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Submission component missing: {source}")
        shutil.copyfile(source, submission_dir / name)
    zip_path = output_dir / "submission-task3-B-C2-D2-I1BR-rank-reject2pct.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(submission_dir / name, arcname=name)
    with zipfile.ZipFile(zip_path) as archive:
        if sorted(archive.namelist()) != sorted(sources) or archive.testzip() is not None:
            raise RuntimeError("Submission zip integrity check failed.")
    return submission_dir, zip_path, sorted(sources)


def main():
    args = parse_args()
    split = "A" if Path(args.test_file).name == "cfn-test-A.json" else "B"
    if Path(args.test_file).name not in {"cfn-test-A.json", "cfn-test-B.json"}:
        raise ValueError("test_file must be cfn-test-A.json or cfn-test-B.json")
    if not np.isclose(args.ensemble_rank_weight + args.validity_rank_weight, 1.0):
        raise ValueError("Ensemble/validity rank weights must sum to one.")
    if not 0 < args.dev_retained <= args.dev_total:
        raise ValueError("Invalid Dev retained/total calibration counts.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_info, role_names, _, frame_to_roles, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    frame_names = [row["frame_name"] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_roles.items():
        membership[frame2idx[frame], role_ids] = 1.0

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    soft_dataset = SoftPriorDataset(
        args.test_file,
        args.task1_probability_file,
        args.task2_prediction_file,
        tokenizer,
    )
    if soft_dataset.frame_names != frame_names:
        raise ValueError(f"Task1 Test-{split} frame order differs from frame_info.")
    verifier_dataset = VerifierTestDataset(soft_dataset, tokenizer, args.max_length)
    loader = DataLoader(
        verifier_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )
    model, checkpoint = load_verifier(
        args, len(role_names), len(frame_names), device
    )
    validity_logits, rows = collect_validity(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(rows) != len(soft_dataset) or not np.isfinite(validity_logits).all():
        raise RuntimeError(f"Verifier Test-{split} outputs are incomplete or non-finite.")
    np.savez_compressed(
        output_dir / "test_validity_logits.npz",
        validity_logits=validity_logits.astype(np.float32),
        sentence_ids=np.asarray([row["sentence_id"] for row in rows], dtype=np.int64),
        starts=np.asarray([row["start"] for row in rows], dtype=np.int32),
        ends=np.asarray([row["end"] for row in rows], dtype=np.int32),
    )

    cache = np.load(args.role_logits_cache, allow_pickle=False)
    required = {"b_logits", "c2_logits", "d2_logits"}
    if required.difference(cache.files):
        raise ValueError(f"Role cache missing: {sorted(required.difference(cache.files))}")
    b_logits = cache["b_logits"].astype(np.float32)
    c2_logits = cache["c2_logits"].astype(np.float32)
    d2_logits = cache["d2_logits"].astype(np.float32)
    expected_shape = (len(rows), len(role_names))
    for name, logits in (("B", b_logits), ("C2", c2_logits), ("D2", d2_logits)):
        if logits.shape != expected_shape or not np.isfinite(logits).all():
            raise ValueError(f"{name} Test-B logits invalid: {logits.shape}")

    frame_probs = np.stack([row["frame_prob"] for row in rows]).astype(np.float32)
    legal_prior = np.clip(frame_probs @ membership, args.prior_epsilon, 1.0)
    frame_top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    frame_confidence = percentile_confidence(frame_top2[:, 1] - frame_top2[:, 0])
    prior_weight = args.prior_low + (
        args.prior_high - args.prior_low
    ) * frame_confidence ** args.prior_power
    role_scores = (
        args.b_weight * zscore_rows(b_logits)
        + args.c2_weight * zscore_rows(c2_logits)
        + args.d2_weight * zscore_rows(d2_logits)
        + prior_weight[:, None] * np.log(legal_prior)
    )
    role_prediction = role_scores.argmax(axis=1)
    indices = np.arange(len(rows))
    role_top1 = role_scores[indices, role_prediction]

    # Reproduce the historical role output before applying rejection.
    reference_rows = read_json(args.reference_task3_file)
    reconstructed = [
        [row["sentence_id"], row["start"], row["end"], role_names[int(role_id)]]
        for row, role_id in zip(rows, role_prediction)
    ]
    if reconstructed != reference_rows:
        changed = sum(
            left != right for left, right in zip(reconstructed, reference_rows)
        ) if len(reconstructed) == len(reference_rows) else None
        raise RuntimeError(
            f"Reconstructed B+C2+D2 output differs from its provenance file; changed={changed}."
        )

    role_rank = percentile_confidence(role_top1)
    validity_rank = percentile_confidence(validity_logits)
    joint_score = (
        args.ensemble_rank_weight * role_rank
        + args.validity_rank_weight * validity_rank
    )
    keep_count = int(round(len(rows) * args.dev_retained / args.dev_total))
    keep_count = min(max(keep_count, 1), len(rows))
    order = np.argsort(-joint_score, kind="stable")
    keep = np.zeros(len(rows), dtype=bool)
    keep[order[:keep_count]] = True
    output_rows = [row for index, row in enumerate(reconstructed) if keep[index]]
    rejected_rows = []
    score_rows = []
    for index, (source, prediction_row) in enumerate(zip(rows, reconstructed)):
        record = {
            "candidate_index": index,
            "sentence_id": int(source["sentence_id"]),
            "start": int(source["start"]),
            "end": int(source["end"]),
            "role": prediction_row[3],
            "kept": int(keep[index]),
            "joint_score": float(joint_score[index]),
            "ensemble_rank": float(role_rank[index]),
            "validity_rank": float(validity_rank[index]),
            "validity_logit": float(validity_logits[index]),
        }
        score_rows.append(record)
        if not keep[index]:
            rejected_rows.append(record)

    task3_file = output_dir / f"{split}_task3_test.json"
    write_json_rows(task3_file, output_rows)
    write_json_rows(output_dir / "rejected_candidates.json", rejected_rows)
    with (output_dir / "candidate_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    if read_json(task3_file) != output_rows:
        raise RuntimeError(f"Filtered {split}_task3_test.json round-trip failed.")
    # Every output row is a retained subsequence of the original Task2 order.
    task2_rows = read_json(args.task2_prediction_file)
    expected_boundaries = [list(row[:3]) for index, row in enumerate(task2_rows) if keep[index]]
    if [list(row[:3]) for row in output_rows] != expected_boundaries:
        raise RuntimeError("Filtered Task3 rows are not an ordered Task2 subsequence.")

    submission_dir, zip_path, members = package_submission(
        args, output_dir, task3_file
    )
    validity_probability = 1.0 / (1.0 + np.exp(-np.clip(validity_logits, -50, 50)))
    summary = {
        "strategy": "B_C2_D2_role_plus_I1BR_rank_calibrated_rejection",
        "test_split": split,
        "role_model": {
            "weights": {"B": args.b_weight, "C2": args.c2_weight, "D2": args.d2_weight},
            "adaptive_prior": {
                "low": args.prior_low, "high": args.prior_high,
                "power": args.prior_power,
            },
            "reconstructed_reference_exactly": True,
        },
        "calibration": {
            "method": "within_split_percentile_rank_and_fixed_dev_retention_rate",
            "ensemble_rank_weight": args.ensemble_rank_weight,
            "validity_rank_weight": args.validity_rank_weight,
            "dev_retained": args.dev_retained,
            "dev_total": args.dev_total,
            "retention_rate": args.dev_retained / args.dev_total,
            "test_rejection_cutoff_joint_score": float(joint_score[order[keep_count - 1]]),
            "raw_probability_threshold_not_used": True,
        },
        "candidate_counts": {
            "task2_candidates": len(rows),
            "task3_retained": len(output_rows),
            "task3_rejected": len(rejected_rows),
        },
        "validity_distribution": {
            "logit_min": float(validity_logits.min()),
            "logit_mean": float(validity_logits.mean()),
            "logit_max": float(validity_logits.max()),
            "probability_min": float(validity_probability.min()),
            "probability_mean": float(validity_probability.mean()),
            "probability_max": float(validity_probability.max()),
        },
        "checkpoint": {
            "path": str(Path(args.verifier_checkpoint).resolve()),
            "source_epoch": checkpoint.get("epoch"),
            "model_variant": checkpoint.get("model_variant"),
            "sha256": sha256(args.verifier_checkpoint),
        },
        "outputs": {
            "task3_file": str(task3_file.resolve()),
            "validity_cache": str((output_dir / "test_validity_logits.npz").resolve()),
            "score_csv": str((output_dir / "candidate_scores.csv").resolve()),
            "rejected_json": str((output_dir / "rejected_candidates.json").resolve()),
            "submission_zip": str(zip_path.resolve()),
        },
        "integrity": {
            "task3_is_ordered_task2_subsequence": True,
            "task3_json_round_trip": True,
            "zip_members": members,
            "zip_crc_ok": True,
            "task3_sha256": sha256(task3_file),
            "zip_sha256": sha256(zip_path),
            "component_sha256": {
                name: sha256(submission_dir / name) for name in members
            },
        },
        "duplicate_sentence_id_groups": soft_dataset.duplicate_sentence_ids,
        "task2_rows_using_identical_duplicate_instances": (
            soft_dataset.collapsed_identical_duplicate_rows
        ),
        "target_overlapping_task2_spans": soft_dataset.target_overlapping_task2_spans,
        "historical_a_files_reused_without_inference": True,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "augmented_frame_role_licenses": augmented,
        "run_config": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
