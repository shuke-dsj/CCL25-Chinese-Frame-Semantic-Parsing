"""Datasets and marker-safe coordinate mapping for Task3 verifier models."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


MARKERS = {
    "target_start": "[unused1]",
    "target_end": "[unused2]",
    "argument_start": "[unused3]",
    "argument_end": "[unused4]",
}


def read_json(path):
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)


def marker_token_ids(tokenizer):
    ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in MARKERS.items()}
    missing = [name for name, value in ids.items() if value == tokenizer.unk_token_id]
    if missing:
        raise ValueError(f"BERT vocabulary is missing marker tokens: {missing}")
    return ids


def encode_with_target_argument_markers(
    tokenizer, text, target_start, target_end, argument_start, argument_end, max_length=512
):
    """Insert four markers using raw character events, including overlap cases.

    The baseline handled only arguments strictly left/right of the target.
    Event-based insertion has no such branch: nested, identical and crossing
    spans all receive deterministic marker locations.
    """
    n = len(text)
    spans = {
        "target": (int(target_start), int(target_end)),
        "argument": (int(argument_start), int(argument_end)),
    }
    for name, (start, end) in spans.items():
        if not (0 <= start <= end < n):
            raise ValueError(f"Invalid raw {name} span {(start, end)} for text length {n}.")
    encoded = tokenizer.encode_plus(list(text))
    base_ids = list(encoded.data["input_ids"])
    if len(base_ids) != n + 2:
        raise ValueError(
            "Character/token alignment is not 1:1: "
            f"text length={n}, encoded length={len(base_ids)}."
        )
    char_ids = base_ids[1:-1]
    ids = marker_token_ids(tokenizer)
    input_ids = [base_ids[0]]
    marker_positions = {}
    for position, character_id in enumerate(char_ids):
        # Fixed start order and reverse end order preserve nesting when spans coincide.
        if position == target_start:
            marker_positions["target_start"] = len(input_ids)
            input_ids.append(ids["target_start"])
        if position == argument_start:
            marker_positions["argument_start"] = len(input_ids)
            input_ids.append(ids["argument_start"])
        input_ids.append(character_id)
        if position == argument_end:
            marker_positions["argument_end"] = len(input_ids)
            input_ids.append(ids["argument_end"])
        if position == target_end:
            marker_positions["target_end"] = len(input_ids)
            input_ids.append(ids["target_end"])
    input_ids.append(base_ids[-1])
    if len(input_ids) > max_length:
        raise ValueError(
            f"Marked sequence length {len(input_ids)} exceeds max_length={max_length}."
        )
    if set(marker_positions) != set(MARKERS):
        raise RuntimeError(f"Marker insertion failed: {marker_positions}")
    return input_ids, [1] * len(input_ids), marker_positions


def _overlaps(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)


def _base_record(instance_index, row, start, end):
    target = row["target"][-1]
    return {
        "instance_index": int(instance_index),
        "sentence_id": int(row["sentence_id"]),
        "text": row["text"],
        "frame": row["frame"],
        "target_start": int(target["start"]),
        "target_end": int(target["end"]),
        "argument_start": int(start),
        "argument_end": int(end),
    }


def build_hard_negative_records(
    rows, label2idx, negative_ratio=2.0, seed=42,
    boundary_negative_fraction=1.0,
):
    if not 0.0 <= float(boundary_negative_fraction) <= 1.0:
        raise ValueError("boundary_negative_fraction must be within [0,1].")
    records = []
    stats = Counter()
    for instance_index, row in enumerate(rows):
        spans = row.get("cfn_spans", [])
        gold_boundaries = {
            (int(span["start"]), int(span["end"])) for span in spans
        }
        positive_lengths = [end - start + 1 for start, end in gold_boundaries] or [1]
        for span in spans:
            record = _base_record(instance_index, row, span["start"], span["end"])
            record.update({
                "valid_label": 1,
                "role_label": int(label2idx[span["fe_name"]]),
                "gold_role_ids": [int(label2idx[span["fe_name"]])],
                "sample_type": "positive",
            })
            records.append(record)
            stats["positive_count"] += 1

        desired = int(math.ceil(len(spans) * float(negative_ratio)))
        if desired <= 0:
            continue
        n = len(row["text"])
        candidates = []
        seen = set(gold_boundaries)

        def add(start, end, sample_type):
            start, end = int(start), int(end)
            if not (0 <= start <= end < n):
                return
            boundary = (start, end)
            if boundary in seen:
                stats["duplicate_removed"] += 1
                return
            seen.add(boundary)
            candidates.append((start, end, sample_type))

        # Boundary perturbations are high-value hard negatives for Task2 errors.
        for start, end in sorted(gold_boundaries):
            for new_start, new_end in (
                (start - 1, end), (start + 1, end),
                (start, end - 1), (start, end + 1),
                (start - 1, end - 1), (start + 1, end + 1),
            ):
                add(new_start, new_end, "boundary_perturbation")

        rng = random.Random(int(seed) * 1000003 + instance_index)
        rng.shuffle(candidates)
        boundary_quota = min(
            len(candidates), int(round(desired * float(boundary_negative_fraction)))
        )
        candidates = candidates[:boundary_quota]
        words = row.get("word", [])
        attempts = 0
        while len(candidates) < desired and attempts < desired * 100 + 100:
            attempts += 1
            if words and rng.random() < 0.6:
                left = rng.randrange(len(words))
                right = rng.randrange(left, min(len(words), left + 5))
                start, end = int(words[left]["start"]), int(words[right]["end"])
                sample_type = "random_word_span"
            else:
                width = max(1, rng.choice(positive_lengths))
                width = min(width, n)
                start = rng.randrange(max(n - width + 1, 1))
                end = start + width - 1
                sample_type = "random_character_span"
            add(start, end, sample_type)

        for start, end, sample_type in candidates[:desired]:
            record = _base_record(instance_index, row, start, end)
            record.update({
                "valid_label": 0,
                "role_label": -100,
                "gold_role_ids": [],
                "sample_type": sample_type,
            })
            records.append(record)
            stats["negative_count"] += 1
            stats[sample_type + "_count"] += 1
            if _overlaps(
                start, end, record["target_start"], record["target_end"]
            ):
                stats["negative_target_overlap_count"] += 1

    stats["total_count"] = len(records)
    return records, dict(stats)


def build_pipeline_records(rows, candidate_file, label2idx):
    candidates = read_json(candidate_file)
    gold_by_boundary = {}
    gold_tuple_count = 0
    gold_boundary_count = 0
    for instance_index, row in enumerate(rows):
        for span in row.get("cfn_spans", []):
            boundary = (instance_index, int(span["start"]), int(span["end"]))
            if boundary not in gold_by_boundary:
                gold_by_boundary[boundary] = []
                gold_boundary_count += 1
            role_id = int(label2idx[span["fe_name"]])
            if role_id not in gold_by_boundary[boundary]:
                gold_by_boundary[boundary].append(role_id)
                gold_tuple_count += 1
    records = []
    for candidate_index, candidate in enumerate(candidates):
        instance_index = int(candidate["instance_index"])
        if not 0 <= instance_index < len(rows):
            raise IndexError(f"Candidate instance_index is out of range: {candidate}")
        row = rows[instance_index]
        if int(candidate["sentence_id"]) != int(row["sentence_id"]):
            raise ValueError(f"Candidate/source sentence_id mismatch: {candidate}")
        start, end = int(candidate["start"]), int(candidate["end"])
        gold_roles = gold_by_boundary.get((instance_index, start, end), [])
        record = _base_record(instance_index, row, start, end)
        record.update({
            "candidate_index": candidate_index,
            "valid_label": int(bool(gold_roles)),
            "role_label": int(gold_roles[0]) if gold_roles else -100,
            "gold_role_ids": list(gold_roles),
            "sample_type": "pipeline_candidate",
            "task2_score": float(candidate.get("task2_score", 0.0)),
        })
        records.append(record)
    stats = {
        "total_count": len(records),
        "positive_count": sum(row["valid_label"] for row in records),
        "negative_count": sum(not row["valid_label"] for row in records),
        "gold_tuple_count": gold_tuple_count,
        "gold_boundary_count": gold_boundary_count,
    }
    return records, stats


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True).clip(min=1e-12)


def load_task1_hybrid_probabilities(
    path, source_rows, train_rows, frame_names, gamma=2.7, baseline_weight=0.5
):
    cache = np.load(path)
    required = {"baseline_logits", "prototype_logits", "sentence_ids"}
    if required.difference(cache.files):
        raise ValueError(f"Task1 cache missing: {sorted(required.difference(cache.files))}")
    if len(cache["sentence_ids"]) != len(source_rows):
        raise ValueError("Task1 cache/source instance counts differ.")
    expected_ids = np.asarray([int(row["sentence_id"]) for row in source_rows])
    if not np.array_equal(cache["sentence_ids"].astype(np.int64), expected_ids):
        raise ValueError("Task1 cache is not aligned with source JSON instance order.")
    if cache["baseline_logits"].shape[1] != len(frame_names):
        raise ValueError("Task1 frame dimension differs from frame_info order.")
    seen_frames = {row["frame"] for row in train_rows}
    seen_mask = np.asarray([name in seen_frames for name in frame_names], dtype=bool)
    prototype = cache["prototype_logits"].astype(np.float32).copy()
    prototype[:, seen_mask] -= float(gamma)
    baseline_prob = _softmax(cache["baseline_logits"].astype(np.float32))
    prototype_prob = _softmax(prototype)
    probabilities = (
        float(baseline_weight) * baseline_prob
        + (1.0 - float(baseline_weight)) * prototype_prob
    )
    return probabilities.astype(np.float32)


class SpanRoleVerifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_file,
        tokenizer,
        label2idx,
        frame2idx,
        max_length=512,
        candidate_file=None,
        negative_ratio=2.0,
        seed=42,
        task1_probabilities=None,
        boundary_negative_fraction=1.0,
        max_examples=None,
    ):
        self.source_file = str(source_file)
        self.raw_rows = read_json(source_file)
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.frame2idx = frame2idx
        self.max_length = int(max_length)
        self.task1_probabilities = task1_probabilities
        if candidate_file:
            self.records, self.stats = build_pipeline_records(
                self.raw_rows, candidate_file, label2idx
            )
        else:
            self.records, self.stats = build_hard_negative_records(
                self.raw_rows, label2idx, negative_ratio, seed,
                boundary_negative_fraction=boundary_negative_fraction,
            )
        if max_examples is not None:
            self.records = self.records[: int(max_examples)]
            self.stats = dict(self.stats)
            self.stats["truncated_total_count"] = len(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        input_ids, attention_mask, marker_positions = encode_with_target_argument_markers(
            self.tokenizer,
            row["text"],
            row["target_start"], row["target_end"],
            row["argument_start"], row["argument_end"],
            self.max_length,
        )
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_positions": marker_positions,
            "frame_id": int(self.frame2idx[row["frame"]]),
            "row": row,
        }
        if self.task1_probabilities is not None:
            output["frame_probabilities"] = self.task1_probabilities[row["instance_index"]]
        return output


def make_collate(frame_to_role_ids_by_id, num_roles, frame_topk=5, pad_token_id=0):
    def collate(items):
        max_len = max(len(item["input_ids"]) for item in items)
        input_ids, attention_masks = [], []
        for item in items:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_masks.append(item["attention_mask"] + [0] * pad)
        positions = {
            name: torch.tensor(
                [item["marker_positions"][name] for item in items], dtype=torch.long
            )
            for name in MARKERS
        }
        frame_ids = torch.tensor([item["frame_id"] for item in items], dtype=torch.long)
        frame_probabilities = None
        if "frame_probabilities" in items[0]:
            frame_probabilities = torch.tensor(
                np.stack([item["frame_probabilities"] for item in items]),
                dtype=torch.float32,
            )
        candidate_mask = torch.zeros((len(items), num_roles), dtype=torch.bool)
        for index, item in enumerate(items):
            if frame_probabilities is None:
                candidate_frames = [item["frame_id"]]
            else:
                k = min(int(frame_topk), frame_probabilities.shape[1])
                candidate_frames = torch.topk(frame_probabilities[index], k=k).indices.tolist()
            role_ids = set()
            for frame_id in candidate_frames:
                role_ids.update(frame_to_role_ids_by_id[int(frame_id)])
            if not role_ids:
                candidate_mask[index] = True
            else:
                candidate_mask[index, list(role_ids)] = True
        rows = [item["row"] for item in items]
        labels = torch.tensor([row["role_label"] for row in rows], dtype=torch.long)
        # Train uses the gold frame inventory; observed data licenses should
        # guarantee that every positive role is available.
        for index, label in enumerate(labels.tolist()):
            if label >= 0 and frame_probabilities is None and not candidate_mask[index, label]:
                raise ValueError(f"Positive role absent from gold-frame candidates: {rows[index]}")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "marker_positions": positions,
            "frame_ids": frame_ids,
            "frame_probabilities": frame_probabilities,
            "candidate_mask": candidate_mask,
            "valid_labels": torch.tensor(
                [row["valid_label"] for row in rows], dtype=torch.float32
            ),
            "role_labels": labels,
            "rows": rows,
        }
    return collate
