#!/usr/bin/env python3
"""Audit the Task3 data and evaluate legacy vs frame-constrained decoding.

This script is intentionally analysis-only: it does not modify checkpoints or
training code.  All outputs are written under ``tasks/task3/analysis``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.task3.frame_utils import apply_frame_mask, load_frame_ontology
from tasks.task3.model import Model
from tasks.common.experiment_utils import REPO_ROOT


TASK_DIR = REPO_ROOT / "tasks" / "task3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset" / "cfn-train.json"))
    parser.add_argument("--dev_file", default=str(REPO_ROOT / "dataset" / "cfn-dev.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset" / "frame_info.json"))
    parser.add_argument("--vocab_file", default=str(REPO_ROOT / "chinese_bert_wwm_ext" / "vocab.txt"))
    parser.add_argument("--config_file", default=str(REPO_ROOT / "chinese_bert_wwm_ext" / "config.json"))
    parser.add_argument(
        "--checkpoint_path",
        default=str(TASK_DIR / "experiment_record" / "legacy-saves" / "checkpoints" / "model_task3_best.bin"),
    )
    parser.add_argument("--output_dir", default=str(TASK_DIR / "analysis" / "task3-baseline-audit"))
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--skip_model", action="store_true", help="run the data audit only")
    parser.add_argument("--max_examples", default=None, type=int, help="model smoke-test subset")
    parser.add_argument(
        "--task1_predictions",
        default=None,
        help="optional Task1 official JSON or dev analysis CSV for predicted-frame masking",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def load_ordered_frame_predictions(path, dev_data):
    """Load one Task1 frame per cfn-dev *instance*, preserving duplicates."""
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        rows = read_json(path_obj)
        sentence_ids = [int(x[0]) for x in rows]
        frames = [x[1] for x in rows]
    elif path_obj.suffix.lower() == ".csv":
        with path_obj.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        candidates = (
            "hybrid_pred", "best_hybrid_pred", "final_pred", "ensemble_pred",
            "prototype_pred", "aware_pred", "baseline_pred", "pred", "frame",
        )
        pred_col = next((c for c in candidates if rows and c in rows[0]), None)
        if pred_col is None:
            raise ValueError(f"No prediction column from {candidates} in {path}")
        sentence_ids = [int(x["sentence_id"]) for x in rows]
        frames = [x[pred_col] for x in rows]
    else:
        raise ValueError(f"Unsupported Task1 prediction format: {path}")
    expected_ids = [int(x["sentence_id"]) for x in dev_data]
    if sentence_ids != expected_ids:
        raise ValueError(
            "Task1 predictions must follow cfn-dev instance order; sentence_id alone "
            "is not unique and cannot be used as a dictionary key."
        )
    return frames


def bucket(freq: int) -> str:
    if freq == 0:
        return "unseen_0"
    if freq <= 5:
        return "rare_1_5"
    if freq <= 15:
        return "few_6_15"
    if freq <= 50:
        return "medium_16_50"
    return "head_gt_50"


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def summarize_counter(counter: Counter):
    values = sorted(counter.elements())
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "min": values[0],
        "mean": float(sum(values) / len(values)),
        "median": float(np.median(values)),
        "max": values[-1],
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def data_audit(train_data, dev_data, frame_to_roles, label2idx, output_dir: Path):
    train_role_freq = Counter(
        span["fe_name"] for line in train_data for span in line.get("cfn_spans", [])
    )
    train_frame_freq = Counter(line.get("frame", "") for line in train_data)

    split_summary = {}
    all_multirole_rows = []
    overlap_rows = []
    for split, data in (("train", train_data), ("dev", dev_data)):
        annotations = 0
        overlap_target = 0
        illegal_gold = 0
        unknown_frames = 0
        group_roles = defaultdict(set)
        role_freq = Counter()
        frame_freq = Counter()
        for line in data:
            frame = line.get("frame", "")
            frame_freq[frame] += 1
            if frame not in frame_to_roles:
                unknown_frames += 1
            target = line["target"][-1]
            legal = set(frame_to_roles.get(frame, []))
            for span in line.get("cfn_spans", []):
                annotations += 1
                role = span["fe_name"]
                role_freq[role] += 1
                group_roles[(int(line["sentence_id"]), span["start"], span["end"])].add(role)
                if not (span["end"] < target["start"] or span["start"] > target["end"]):
                    overlap_target += 1
                    overlap_rows.append({
                        "split": split, "sentence_id": int(line["sentence_id"]),
                        "frame": frame, "target_start": target["start"],
                        "target_end": target["end"], "argument_start": span["start"],
                        "argument_end": span["end"], "role": role,
                    })
                if role not in legal:
                    illegal_gold += 1

        multi_groups = {k: v for k, v in group_roles.items() if len(v) > 1}
        for (sid, start, end), roles in sorted(multi_groups.items()):
            all_multirole_rows.append({
                "split": split, "sentence_id": sid, "start": start, "end": end,
                "num_roles": len(roles), "roles": " | ".join(sorted(roles)),
            })
        unique_span_groups = len(group_roles)
        split_summary[split] = {
            "sentences": len(data),
            "annotations": annotations,
            "unique_argument_boundaries": unique_span_groups,
            "multi_role_boundary_groups": len(multi_groups),
            "multi_role_annotations": sum(len(v) for v in multi_groups.values()),
            "single_label_decoder_oracle_recall_ceiling": safe_div(unique_span_groups, annotations),
            "target_overlapping_arguments": overlap_target,
            "unknown_frames": unknown_frames,
            "gold_roles_not_licensed_by_frame_info": illegal_gold,
            "unique_frames": len(frame_freq),
            "unique_roles": len(role_freq),
        }

    candidate_rows = []
    for frame, roles in frame_to_roles.items():
        candidate_rows.append({
            "frame": frame,
            "candidate_roles": len(roles),
            "train_sentences": train_frame_freq.get(frame, 0),
        })
    write_csv(
        output_dir / "frame_candidate_sizes.csv", candidate_rows,
        ["frame", "candidate_roles", "train_sentences"],
    )
    write_csv(
        output_dir / "multi_role_same_span.csv", all_multirole_rows,
        ["split", "sentence_id", "start", "end", "num_roles", "roles"],
    )
    write_csv(
        output_dir / "target_overlapping_arguments.csv", overlap_rows,
        ["split", "sentence_id", "frame", "target_start", "target_end",
         "argument_start", "argument_end", "role"],
    )

    candidate_sizes = [len(x) for x in frame_to_roles.values()]
    return {
        "splits": split_summary,
        "ontology": {
            "frames": len(frame_to_roles),
            "global_unique_roles": len(label2idx),
            "candidate_roles_per_frame": {
                "min": min(candidate_sizes) if candidate_sizes else 0,
                "mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
                "median": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
                "max": max(candidate_sizes) if candidate_sizes else 0,
            },
            "mean_candidate_reduction_ratio": (
                1.0 - float(np.mean(candidate_sizes)) / len(label2idx)
                if candidate_sizes and label2idx else 0.0
            ),
        },
        "train_role_frequency": dict(train_role_freq),
    }


class AuditDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, label2idx, max_examples=None):
        self.records = []
        self.excluded_target_overlap = 0
        for instance_index, line in enumerate(data):
            text = line["text"]
            raw_target = line["target"][-1]
            marked_target = [raw_target["start"] + 1, raw_target["end"] + 1]
            for span in line.get("cfn_spans", []):
                if span["end"] < raw_target["start"]:
                    marked_span = [span["start"] + 1, span["end"] + 1]
                elif span["start"] > raw_target["end"]:
                    marked_span = [span["start"] + 3, span["end"] + 3]
                else:
                    # The legacy Dataset accidentally reuses the previous
                    # ``label_idx`` for these cases. Exclude them from the clean
                    # model audit and report the count instead of evaluating a
                    # knowingly wrong argument boundary.
                    self.excluded_target_overlap += 1
                    continue
                self.records.append({
                    "text": text,
                    "marked_target": marked_target,
                    "marked_span": marked_span,
                    "label": label2idx[span["fe_name"]],
                    "role": span["fe_name"],
                    "frame": line["frame"],
                    "sentence_id": int(line["sentence_id"]),
                    "instance_index": instance_index,
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                })
        if max_examples is not None:
            self.records = self.records[:max_examples]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        encoded = self.tokenizer.encode_plus(list(row["text"]))
        input_ids = encoded.data["input_ids"]
        attention_mask = encoded.data["attention_mask"]
        target = row["marked_target"]
        input_ids = (
            input_ids[:target[0]] + [1] + input_ids[target[0]:target[1] + 1]
            + [2] + input_ids[target[1] + 1:]
        )
        attention_mask = attention_mask + [1, 1]
        return input_ids, attention_mask, row


def collate(batch, device):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.tensor(
        [x[0] + [0] * (max_len - len(x[0])) for x in batch],
        dtype=torch.long, device=device,
    )
    attention_mask = torch.tensor(
        [x[1] + [0] * (max_len - len(x[1])) for x in batch],
        dtype=torch.long, device=device,
    )
    rows = [x[2] for x in batch]
    spans = [row["marked_span"] for row in rows]
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    return input_ids, attention_mask, spans, labels, rows


def init_stats():
    return {"correct": 0, "top3": 0, "top5": 0, "count": 0}


def update_stats(stats, logits, labels):
    k = min(5, logits.shape[-1])
    top = torch.topk(logits, k=k, dim=-1).indices
    stats["correct"] += (top[:, 0] == labels).sum().item()
    stats["top3"] += (top[:, :min(3, k)] == labels[:, None]).any(dim=1).sum().item()
    stats["top5"] += (top == labels[:, None]).any(dim=1).sum().item()
    stats["count"] += labels.numel()


def finalize_stats(stats):
    return {
        "acc": safe_div(stats["correct"], stats["count"]),
        "top3_acc": safe_div(stats["top3"], stats["count"]),
        "top5_acc": safe_div(stats["top5"], stats["count"]),
        **stats,
    }


def evaluate_model(args, dev_data, idx2label, label2idx, frame_to_role_ids, train_role_freq, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = AuditDataset(dev_data, tokenizer, label2idx, args.max_examples)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=partial(collate, device=device), drop_last=False,
    )

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = len(idx2label)
    model = Model(config)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    msg = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    predicted_frames = (
        load_ordered_frame_predictions(args.task1_predictions, dev_data)
        if args.task1_predictions else None
    )

    global_stats = init_stats()
    oracle_stats = init_stats()
    predicted_stats = init_stats() if predicted_frames is not None else None
    bucket_stats = defaultdict(lambda: {"global": init_stats(), "oracle_frame": init_stats()})
    role_stats = defaultdict(lambda: {"global_correct": 0, "oracle_correct": 0, "count": 0})
    rows_out = []
    cached_logits = []
    cached_labels = []
    cached_sentence_ids = []
    cached_starts = []
    cached_ends = []
    loss_sum = 0.0
    unknown_predicted_frames = 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for input_ids, attention_mask, spans, labels, rows in tqdm(loader, desc="task3-audit"):
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, target=spans,
                labels=labels, device=device, for_test=False,
            )
            logits = output["logits"]
            cached_logits.append(logits.detach().cpu().to(torch.float32).numpy())
            cached_labels.append(labels.detach().cpu().numpy())
            cached_sentence_ids.extend(r["sentence_id"] for r in rows)
            cached_starts.extend(r["start"] for r in rows)
            cached_ends.extend(r["end"] for r in rows)
            loss_sum += output["loss"].item() * labels.numel()
            oracle_logits, oracle_mask = apply_frame_mask(
                logits, [r["frame"] for r in rows], frame_to_role_ids, "error"
            )
            update_stats(global_stats, logits, labels)
            update_stats(oracle_stats, oracle_logits, labels)

            pred_logits = None
            used_predicted_frames = None
            if predicted_frames is not None:
                used_predicted_frames = [predicted_frames[r["instance_index"]] for r in rows]
                unknown_predicted_frames += sum(f not in frame_to_role_ids for f in used_predicted_frames)
                pred_logits, _ = apply_frame_mask(
                    logits, used_predicted_frames, frame_to_role_ids, "global"
                )
                update_stats(predicted_stats, pred_logits, labels)

            global_pred = logits.argmax(dim=-1)
            oracle_pred = oracle_logits.argmax(dim=-1)
            predicted_pred = pred_logits.argmax(dim=-1) if pred_logits is not None else None
            for i, row in enumerate(rows):
                freq_bucket = bucket(train_role_freq.get(row["role"], 0))
                update_stats(bucket_stats[freq_bucket]["global"], logits[i:i + 1], labels[i:i + 1])
                update_stats(bucket_stats[freq_bucket]["oracle_frame"], oracle_logits[i:i + 1], labels[i:i + 1])
                rs = role_stats[row["role"]]
                rs["count"] += 1
                rs["global_correct"] += int(global_pred[i] == labels[i])
                rs["oracle_correct"] += int(oracle_pred[i] == labels[i])
                rows_out.append({
                    "sentence_id": row["sentence_id"], "start": row["start"], "end": row["end"],
                    "frame": row["frame"], "gold_role": row["role"],
                    "train_frequency": train_role_freq.get(row["role"], 0), "frequency_bucket": freq_bucket,
                    "global_pred": idx2label[int(global_pred[i])],
                    "oracle_frame_pred": idx2label[int(oracle_pred[i])],
                    "predicted_frame": used_predicted_frames[i] if used_predicted_frames else "",
                    "predicted_frame_pred": idx2label[int(predicted_pred[i])] if predicted_pred is not None else "",
                    "global_correct": int(global_pred[i] == labels[i]),
                    "oracle_frame_correct": int(oracle_pred[i] == labels[i]),
                    "predicted_frame_correct": int(predicted_pred[i] == labels[i]) if predicted_pred is not None else "",
                    "oracle_candidate_count": int(oracle_mask[i].sum().item()),
                })

    elapsed = time.perf_counter() - start_time
    write_csv(
        output_dir / "dev_predictions.csv", rows_out,
        ["sentence_id", "start", "end", "frame", "gold_role", "train_frequency", "frequency_bucket",
         "global_pred", "oracle_frame_pred", "predicted_frame", "predicted_frame_pred",
         "global_correct", "oracle_frame_correct", "predicted_frame_correct", "oracle_candidate_count"],
    )
    np.savez_compressed(
        output_dir / "dev_logits.npz",
        logits=np.concatenate(cached_logits, axis=0),
        labels=np.concatenate(cached_labels, axis=0),
        sentence_ids=np.asarray(cached_sentence_ids, dtype=np.int64),
        starts=np.asarray(cached_starts, dtype=np.int32),
        ends=np.asarray(cached_ends, dtype=np.int32),
    )
    role_rows = []
    for role, stat in sorted(role_stats.items(), key=lambda x: (-x[1]["count"], x[0])):
        role_rows.append({
            "role": role, "train_frequency": train_role_freq.get(role, 0), "dev_count": stat["count"],
            "global_acc": safe_div(stat["global_correct"], stat["count"]),
            "oracle_frame_acc": safe_div(stat["oracle_correct"], stat["count"]),
        })
    write_csv(
        output_dir / "role_metrics.csv", role_rows,
        ["role", "train_frequency", "dev_count", "global_acc", "oracle_frame_acc"],
    )

    dense_params = model.dense.weight.numel() + model.dense.bias.numel()
    total_params = sum(p.numel() for p in model.parameters())
    result = {
        "device": str(device),
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "checkpoint_load": {"missing": msg.missing_keys, "unexpected": msg.unexpected_keys},
        "examples": len(dataset),
        "excluded_target_overlapping_examples": dataset.excluded_target_overlap,
        "mean_loss": safe_div(loss_sum, len(dataset)),
        "global": finalize_stats(global_stats),
        "oracle_frame_constrained": finalize_stats(oracle_stats),
        "predicted_frame_constrained": finalize_stats(predicted_stats) if predicted_stats else None,
        "predicted_frame_missing_or_unknown_examples": unknown_predicted_frames,
        "frequency_buckets": {
            name: {variant: finalize_stats(stat) for variant, stat in variants.items()}
            for name, variants in bucket_stats.items()
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "examples_per_second": safe_div(len(dataset), elapsed),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
            ),
            "total_parameters": total_params,
            "dense_head_parameters": dense_params,
            "dense_head_parameter_ratio": safe_div(dense_params, total_params),
            "dense_output_width": config.num_labels * model.inner_dim * 2,
        },
    }
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = read_json(args.train_file)
    dev_data = read_json(args.dev_file)
    _, idx2label, label2idx, frame_to_roles, frame_to_role_ids = load_frame_ontology(args.frame_info_file)
    summary = data_audit(train_data, dev_data, frame_to_roles, label2idx, output_dir)

    if not args.skip_model:
        summary["model_evaluation"] = evaluate_model(
            args, dev_data, idx2label, label2idx, frame_to_role_ids,
            Counter(summary.pop("train_role_frequency")), output_dir,
        )
    else:
        # Keep the potentially large role-frequency mapping out of summary.json.
        summary.pop("train_role_frequency", None)

    summary["run_config"] = vars(args)
    with (output_dir / "summary.json").open("w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
