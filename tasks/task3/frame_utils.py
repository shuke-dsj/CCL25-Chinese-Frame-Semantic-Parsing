"""Frame/FE ontology helpers shared by Task3 analysis and prediction.

The legacy Task3 baseline builds the global FE label order by scanning
``frame_info.json`` from top to bottom.  Keep that exact convention here so
old checkpoints remain compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch


def load_frame_ontology(frame_info_file: str):
    with open(frame_info_file, "r", encoding="utf8") as f:
        frame_info = json.load(f)

    idx2label: List[str] = []
    label2idx: Dict[str, int] = {}
    frame_to_roles: Dict[str, List[str]] = {}

    for frame in frame_info:
        frame_name = frame["frame_name"]
        roles: List[str] = []
        seen_in_frame: Set[str] = set()
        for fe in frame.get("fes", []):
            role = fe["fe_name"]
            if role not in label2idx:
                label2idx[role] = len(idx2label)
                idx2label.append(role)
            if role not in seen_in_frame:
                roles.append(role)
                seen_in_frame.add(role)
        frame_to_roles[frame_name] = roles

    frame_to_role_ids = {
        frame: [label2idx[role] for role in roles]
        for frame, roles in frame_to_roles.items()
    }
    return frame_info, idx2label, label2idx, frame_to_roles, frame_to_role_ids


def build_frame_mask(
    frames: Sequence[str],
    frame_to_role_ids: Mapping[str, Sequence[int]],
    num_labels: int,
    device,
    unknown_policy: str = "global",
) -> torch.Tensor:
    """Return a Boolean ``[batch, num_labels]`` legal-role mask.

    ``global`` deliberately falls back to all roles for a missing/unknown
    frame.  This is safer than returning an all-false row and makes pipeline
    failure explicit in the analysis rather than silently forcing label 0.
    """
    if unknown_policy not in {"global", "error"}:
        raise ValueError(f"unknown_policy must be global/error, got {unknown_policy}")
    mask = torch.zeros((len(frames), num_labels), dtype=torch.bool, device=device)
    for i, frame in enumerate(frames):
        role_ids = frame_to_role_ids.get(frame)
        if role_ids:
            mask[i, torch.as_tensor(role_ids, dtype=torch.long, device=device)] = True
        elif unknown_policy == "global":
            mask[i] = True
        else:
            raise KeyError(f"Unknown frame or empty FE inventory: {frame!r}")
    return mask


def apply_frame_mask(
    logits: torch.Tensor,
    frames: Sequence[str],
    frame_to_role_ids: Mapping[str, Sequence[int]],
    unknown_policy: str = "global",
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = build_frame_mask(
        frames, frame_to_role_ids, logits.shape[-1], logits.device, unknown_policy
    )
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min), mask


def load_sentence_frame_predictions(path: str) -> Dict[int, str]:
    """Load Task1 predictions from official JSON or an analysis CSV.

    Official JSON rows are ``[sentence_id, frame_name]``.  For CSV files the
    preferred prediction columns are searched from best/final to baseline.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        with path_obj.open("r", encoding="utf8") as f:
            rows = json.load(f)
        return {int(row[0]): row[1] for row in rows}

    if path_obj.suffix.lower() == ".csv":
        import csv

        with path_obj.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        candidates = (
            "hybrid_pred", "best_hybrid_pred", "final_pred", "ensemble_pred",
            "prototype_pred", "aware_pred", "baseline_pred", "pred", "frame",
        )
        pred_col = next((c for c in candidates if c in rows[0]), None)
        if pred_col is None or "sentence_id" not in rows[0]:
            raise ValueError(
                f"CSV must contain sentence_id and one of {candidates}; got {list(rows[0])}"
            )
        return {int(row["sentence_id"]): row[pred_col] for row in rows}

    raise ValueError(f"Unsupported Task1 prediction format: {path_obj}")
