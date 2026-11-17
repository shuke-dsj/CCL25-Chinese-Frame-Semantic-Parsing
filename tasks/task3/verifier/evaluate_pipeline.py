#!/usr/bin/env python3
"""Re-evaluate an I1 checkpoint on the fixed Task1+Task2 Dev pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.verifier.dataset import (
    SpanRoleVerifierDataset,
    load_task1_hybrid_probabilities,
    make_collate,
    read_json,
)
from tasks.task3.verifier.model import FrameConditionedSpanRoleVerifier
from tasks.task3.verifier.train import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--dev_file", default=str(REPO_ROOT / "dataset/cfn-dev.json"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    parser.add_argument(
        "--dev_candidate_file",
        default=str(REPO_ROOT / "tasks/task2/analysis/task2-BDEH-dev-pipeline-spans/task2_instance_predictions.json"),
    )
    parser.add_argument(
        "--task1_dev_logits",
        default=str(REPO_ROOT / "tasks/task1/analysis/baseline-prototype-10ep/dev_logits.npz"),
    )
    parser.add_argument("--output_dir", default=str(
        REPO_ROOT / "tasks/task3/analysis/task3-I1-checkpoint-evaluation"
    ))
    parser.add_argument("--reference_recall", default=0.6399865478392467, type=float)
    parser.add_argument("--max_recall_drop", default=0.003, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    run = checkpoint["run_config"]
    pretrained_dir = run.get("pretrained_model_dir", str(REPO_ROOT / "chinese_bert_wwm_ext"))
    tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
    frame_info, roles, role2idx, frame_to_roles, _ = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    frames = [row["frame_name"] for row in frame_info]
    if checkpoint["role_names"] != roles or checkpoint["frame_names"] != frames:
        raise ValueError("Checkpoint ontology order differs from current frame_info.")
    frame2idx = {name: index for index, name in enumerate(frames)}
    train_rows, dev_rows = read_json(args.train_file), read_json(args.dev_file)
    probabilities = load_task1_hybrid_probabilities(
        args.task1_dev_logits,
        dev_rows,
        train_rows,
        frames,
        run.get("task1_prototype_gamma", 2.7),
        run.get("task1_baseline_weight", 0.5),
    )
    dataset = SpanRoleVerifierDataset(
        args.dev_file,
        tokenizer,
        role2idx,
        frame2idx,
        max_length=run.get("max_length", 512),
        candidate_file=args.dev_candidate_file,
        task1_probabilities=probabilities,
    )
    by_id = {frame2idx[name]: role_ids for name, role_ids in frame_to_roles.items()}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(
            by_id,
            len(roles),
            frame_topk=run.get("frame_topk", 5),
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    config = BertConfig.from_pretrained(pretrained_dir)
    model = FrameConditionedSpanRoleVerifier(
        config,
        num_roles=len(roles),
        num_frames=len(frames),
        variant=checkpoint["model_variant"],
        projection_dim=run.get("projection_dim", 512),
        frame_dim=run.get("frame_dim", 128),
        frame_condition_dropout=run.get("frame_condition_dropout", 0.2),
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint load mismatch: {incompatible}")
    model.to(device)
    evaluation_args = SimpleNamespace(
        valid_pos_weight=run.get("valid_pos_weight", 2.0),
        focal_gamma=run.get("focal_gamma", 0.0),
        global_role_loss_weight=run.get("global_role_loss_weight", 0.2),
        reference_recall=args.reference_recall,
        max_recall_drop=args.max_recall_drop,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        model, loader, device, evaluation_args, "best", output_dir
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "model_variant": checkpoint["model_variant"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_training_threshold": checkpoint["best_threshold"],
        **metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
