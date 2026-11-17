#!/usr/bin/env python3
"""Fast data/model checks for both I1 verifier variants."""

from __future__ import annotations

import argparse
import gc
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT, set_seed
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.verifier.dataset import (
    SpanRoleVerifierDataset,
    encode_with_target_argument_markers,
    make_collate,
)
from tasks.task3.verifier.model import FrameConditionedSpanRoleVerifier
from tasks.task3.verifier.train import forward_model, objective_loss, to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--pretrained_model_dir", default=str(REPO_ROOT / "chinese_bert_wwm_ext"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    return parser.parse_args()


def check_marker_cases(tokenizer):
    cases = [
        ("甲乙丙丁戊", 0, 1, 3, 4, "target_before_argument"),
        ("甲乙丙丁戊", 3, 4, 0, 1, "argument_before_target"),
        ("甲乙丙丁戊", 1, 3, 2, 2, "nested_overlap"),
        ("甲乙丙丁戊", 2, 2, 2, 2, "identical_single_character"),
        ("甲乙丙丁戊", 1, 2, 2, 3, "crossing_overlap"),
    ]
    output = {}
    for text, ts, te, arg_s, arg_e, name in cases:
        ids, mask, positions = encode_with_target_argument_markers(
            tokenizer, text, ts, te, arg_s, arg_e
        )
        assert len(ids) == len(text) + 6
        assert len(set(positions.values())) == 4
        assert len(ids) == len(mask)
        output[name] = positions
    return output


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    marker_cases = check_marker_cases(tokenizer)
    frame_info, roles, role2idx, frame_to_roles, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    frames = [row["frame_name"] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frames)}
    by_id = {frame2idx[name]: role_ids for name, role_ids in frame_to_roles.items()}
    dataset = SpanRoleVerifierDataset(
        args.train_file,
        tokenizer,
        role2idx,
        frame2idx,
        negative_ratio=1.0,
        seed=42,
        max_examples=max(args.batch_size, 4),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(by_id, len(roles), frame_topk=5, pad_token_id=tokenizer.pad_token_id),
    )
    cpu_batch = next(iter(loader))
    pretrained = BertModel.from_pretrained(args.pretrained_model_dir)
    pretrained_state = pretrained.state_dict()
    del pretrained
    summaries = []
    for variant in ("marker_verifier", "frame_conditioned_verifier"):
        config = BertConfig.from_pretrained(args.pretrained_model_dir)
        model = FrameConditionedSpanRoleVerifier(
            config,
            num_roles=len(roles),
            num_frames=len(frames),
            variant=variant,
            projection_dim=128,
            frame_dim=32,
        )
        model.bert.load_state_dict(pretrained_state, strict=True)
        model.to(device).train()
        batch = to_device(cpu_batch, device)
        outputs = forward_model(model, batch, training=True, train_gold_frame_mass=0.9)
        losses = objective_loss(outputs, batch, valid_pos_weight=2.0, global_role_loss_weight=0.2)
        losses["loss"].backward()
        gradients = {
            "validity_head": float(model.validity_head.weight.grad.norm().item()),
            "role_head": float(model.role_head.weight.grad.norm().item()),
        }
        assert all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradients.values())
        summaries.append({
            "variant": variant,
            "input_shape": list(batch["input_ids"].shape),
            "validity_shape": list(outputs["validity_logits"].shape),
            "role_shape": list(outputs["role_logits"].shape),
            "loss": float(losses["loss"].item()),
            "gradients": gradients,
        })
        del model, batch, outputs, losses
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({
        "device": str(device),
        "marker_cases": marker_cases,
        "dataset_statistics": dataset.stats,
        "augmented_frame_role_licenses": augmented,
        "models": summaries,
        "status": "OK: marker mapping, generic-BERT load, forward/backward passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
