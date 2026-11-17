#!/usr/bin/env python3
"""Train I1-A/I1-B precision-oriented Task3 span-role verifiers from BERT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from tasks.common.experiment_utils import (
    ExperimentLogger,
    REPO_ROOT,
    save_checkpoint,
    set_seed,
)
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.verifier.dataset import (
    SpanRoleVerifierDataset,
    load_task1_hybrid_probabilities,
    make_collate,
    read_json,
)
from tasks.task3.verifier.model import FrameConditionedSpanRoleVerifier


TASK_DIR = REPO_ROOT / "tasks" / "task3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_variant", required=True,
        choices=["marker_verifier", "frame_conditioned_verifier"],
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=12, type=int)
    parser.add_argument("--accumulate_gradients", default=1, type=int)
    parser.add_argument("--encoder_lr", default=2e-5, type=float)
    parser.add_argument("--head_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--negative_ratio", default=2.0, type=float)
    parser.add_argument(
        "--boundary_negative_fraction", default=1.0, type=float,
        help="Fraction of hard negatives drawn from gold-boundary perturbations; the rest are random word/character spans.",
    )
    parser.add_argument("--valid_pos_weight", default=2.0, type=float)
    parser.add_argument("--focal_gamma", default=0.0, type=float)
    parser.add_argument("--global_role_loss_weight", default=0.2, type=float)
    parser.add_argument("--frame_topk", default=5, type=int)
    parser.add_argument("--frame_condition_dropout", default=0.2, type=float)
    parser.add_argument("--train_gold_frame_mass", default=0.9, type=float)
    parser.add_argument("--projection_dim", default=512, type=int)
    parser.add_argument("--frame_dim", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_train_examples", default=None, type=int)
    parser.add_argument("--max_dev_examples", default=None, type=int)
    parser.add_argument(
        "--warm_start_checkpoint", default=None,
        help="Optional I1 checkpoint for a controlled continuation/fine-tuning run.",
    )
    parser.add_argument("--pretrained_model_dir", default=str(REPO_ROOT / "chinese_bert_wwm_ext"))
    parser.add_argument("--train_file", default=str(REPO_ROOT / "dataset/cfn-train.json"))
    parser.add_argument("--dev_file", default=str(REPO_ROOT / "dataset/cfn-dev.json"))
    parser.add_argument("--frame_info_file", default=str(REPO_ROOT / "dataset/frame_info.json"))
    parser.add_argument(
        "--dev_candidate_file",
        default=str(REPO_ROOT / "tasks/task2/analysis/task2-BDEH-dev-pipeline-spans/task2_instance_predictions.json"),
    )
    parser.add_argument(
        "--task1_dev_logits",
        default=str(REPO_ROOT / "tasks/task1/analysis/baseline-prototype-10ep/dev_logits.npz"),
    )
    parser.add_argument("--task1_prototype_gamma", default=2.7, type=float)
    parser.add_argument("--task1_baseline_weight", default=0.5, type=float)
    parser.add_argument("--reference_recall", default=0.6399865478392467, type=float)
    parser.add_argument("--max_recall_drop", default=0.003, type=float)
    return parser.parse_args()


def to_device(batch, device):
    output = dict(batch)
    for key in (
        "input_ids", "attention_mask", "frame_ids", "frame_probabilities",
        "candidate_mask", "valid_labels", "role_labels",
    ):
        if output.get(key) is not None:
            output[key] = output[key].to(device)
    output["marker_positions"] = {
        key: value.to(device) for key, value in batch["marker_positions"].items()
    }
    return output


def training_frame_probabilities(batch, num_frames, gold_mass):
    if batch["frame_probabilities"] is not None:
        return batch["frame_probabilities"]
    if not 0 <= gold_mass <= 1:
        raise ValueError("train_gold_frame_mass must be in [0,1].")
    fallback = (1.0 - gold_mass) / max(num_frames, 1)
    probabilities = torch.full(
        (len(batch["frame_ids"]), num_frames),
        fallback,
        dtype=torch.float32,
        device=batch["frame_ids"].device,
    )
    probabilities.scatter_add_(
        1,
        batch["frame_ids"][:, None],
        torch.full(
            (len(batch["frame_ids"]), 1),
            gold_mass,
            device=batch["frame_ids"].device,
        ),
    )
    return probabilities


def forward_model(model, batch, training=False, train_gold_frame_mass=0.9):
    frame_probabilities = batch["frame_probabilities"]
    if model.variant == "frame_conditioned_verifier" and training:
        frame_probabilities = training_frame_probabilities(
            batch, model.num_frames, train_gold_frame_mass
        )
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        marker_positions=batch["marker_positions"],
        frame_ids=batch["frame_ids"],
        frame_probabilities=frame_probabilities,
    )


def objective_loss(
    outputs,
    batch,
    valid_pos_weight=2.0,
    focal_gamma=0.0,
    global_role_loss_weight=0.2,
):
    validity_logits = outputs["validity_logits"]
    valid_labels = batch["valid_labels"]
    raw_validity = F.binary_cross_entropy_with_logits(
        validity_logits,
        valid_labels,
        pos_weight=validity_logits.new_tensor(float(valid_pos_weight)),
        reduction="none",
    )
    if focal_gamma > 0:
        probability = torch.sigmoid(validity_logits)
        pt = probability * valid_labels + (1 - probability) * (1 - valid_labels)
        raw_validity = raw_validity * (1 - pt).pow(float(focal_gamma))
    validity_loss = raw_validity.mean()

    role_logits = outputs["role_logits"]
    positive = batch["role_labels"] >= 0
    frame_role_loss = role_logits.new_zeros(())
    global_role_loss = role_logits.new_zeros(())
    role_count = int(positive.sum().item())
    if role_count:
        labels = batch["role_labels"][positive]
        positive_logits = role_logits[positive]
        global_role_loss = F.cross_entropy(positive_logits, labels)
        mask = batch["candidate_mask"][positive]
        legal_gold = mask.gather(1, labels[:, None]).squeeze(1)
        if legal_gold.any():
            masked = positive_logits[legal_gold].masked_fill(
                ~mask[legal_gold], torch.finfo(role_logits.dtype).min
            )
            frame_role_loss = F.cross_entropy(masked, labels[legal_gold])
        else:
            # Honest Dev top-k masks may exclude all gold roles in a tiny batch.
            frame_role_loss = global_role_loss.detach() * 0.0
    total = validity_loss + frame_role_loss + global_role_loss_weight * global_role_loss
    return {
        "loss": total,
        "validity_loss": validity_loss,
        "frame_role_loss": frame_role_loss,
        "global_role_loss": global_role_loss,
        "role_count": role_count,
    }


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def threshold_metrics(
    thresholds, validity_probabilities, role_predictions, rows, gold_tuples, gold_boundaries
):
    role_correct = np.asarray(
        [int(prediction) in row["gold_role_ids"] for prediction, row in zip(role_predictions, rows)],
        dtype=bool,
    )
    boundary_correct = np.asarray([bool(row["gold_role_ids"]) for row in rows], dtype=bool)
    metrics = []
    for threshold in thresholds:
        keep = validity_probabilities >= float(threshold)
        predicted = int(keep.sum())
        tuple_tp = int((keep & role_correct).sum())
        boundary_tp = int((keep & boundary_correct).sum())
        p, r, f1 = prf(tuple_tp, predicted, gold_tuples)
        bp, br, bf1 = prf(boundary_tp, predicted, gold_boundaries)
        metrics.append({
            "threshold": float(threshold),
            "predicted_tuples": predicted,
            "tuple_tp": tuple_tp,
            "tuple_precision": p,
            "tuple_recall": r,
            "tuple_f1": f1,
            "boundary_tp": boundary_tp,
            "boundary_precision": bp,
            "boundary_recall": br,
            "boundary_f1": bf1,
            "role_acc_on_matched_boundaries": tuple_tp / boundary_tp if boundary_tp else 0.0,
            "false_boundaries_rejected": int(((~boundary_correct) & (~keep)).sum()),
            "correct_tuples_rejected": int((role_correct & (~keep)).sum()),
        })
    return metrics


def evaluate(model, loader, device, args, epoch, result_dir):
    model.eval()
    total_loss = total_validity = total_frame_role = total_global_role = 0.0
    count = 0
    validity_probabilities, role_logits_cache, candidate_masks, rows = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="I1-eval"):
            candidate_masks.append(batch["candidate_mask"].numpy())
            batch = to_device(batch, device)
            outputs = forward_model(model, batch, training=False)
            losses = objective_loss(
                outputs, batch, args.valid_pos_weight, args.focal_gamma,
                args.global_role_loss_weight,
            )
            batch_size = len(batch["rows"])
            count += batch_size
            total_loss += float(losses["loss"].item()) * batch_size
            total_validity += float(losses["validity_loss"].item()) * batch_size
            total_frame_role += float(losses["frame_role_loss"].item()) * batch_size
            total_global_role += float(losses["global_role_loss"].item()) * batch_size
            validity_probabilities.append(
                torch.sigmoid(outputs["validity_logits"]).cpu().float().numpy()
            )
            role_logits_cache.append(outputs["role_logits"].cpu().float().numpy())
            rows.extend(batch["rows"])
    validity_probabilities = np.concatenate(validity_probabilities)
    role_logits = np.concatenate(role_logits_cache)
    candidate_mask = np.concatenate(candidate_masks, axis=0)
    masked_role_logits = np.where(candidate_mask, role_logits, -1e30)
    role_predictions = masked_role_logits.argmax(axis=1)

    dense = np.linspace(0.0, 1.0, 201)
    quantiles = np.quantile(validity_probabilities, np.linspace(0.0, 1.0, 201))
    thresholds = np.unique(np.concatenate([dense, quantiles, [0.5]]))
    grid = threshold_metrics(
        thresholds,
        validity_probabilities,
        role_predictions,
        rows,
        loader.dataset.stats["gold_tuple_count"],
        loader.dataset.stats["gold_boundary_count"],
    )
    best_f1 = max(grid, key=lambda row: (row["tuple_f1"], row["tuple_precision"]))
    eligible = [
        row for row in grid
        if row["tuple_recall"] >= args.reference_recall - args.max_recall_drop
    ]
    best_constrained = (
        max(eligible, key=lambda row: (row["tuple_precision"], row["tuple_f1"]))
        if eligible else None
    )
    all_retained = min(grid, key=lambda row: abs(row["threshold"]))
    boundary_correct = np.asarray([bool(row["gold_role_ids"]) for row in rows], dtype=bool)
    validity_prediction = validity_probabilities >= 0.5
    vtp = int((validity_prediction & boundary_correct).sum())
    vp, vr, vf1 = prf(
        vtp, int(validity_prediction.sum()), loader.dataset.stats["gold_boundary_count"]
    )
    matched = boundary_correct.sum()
    role_correct = sum(
        int(prediction) in row["gold_role_ids"]
        for prediction, row in zip(role_predictions, rows)
    )

    result_dir = Path(result_dir)
    with (result_dir / f"threshold_sweep_epoch_{epoch}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    np.savez_compressed(
        result_dir / f"dev_pipeline_logits_epoch_{epoch}.npz",
        validity_probabilities=validity_probabilities.astype(np.float32),
        role_logits=role_logits.astype(np.float32),
        role_predictions=role_predictions.astype(np.int32),
        instance_indices=np.asarray([row["instance_index"] for row in rows], dtype=np.int32),
        sentence_ids=np.asarray([row["sentence_id"] for row in rows], dtype=np.int64),
        starts=np.asarray([row["argument_start"] for row in rows], dtype=np.int32),
        ends=np.asarray([row["argument_end"] for row in rows], dtype=np.int32),
    )
    return {
        "val_loss": total_loss / max(count, 1),
        "val_validity_loss": total_validity / max(count, 1),
        "val_frame_role_loss": total_frame_role / max(count, 1),
        "val_global_role_loss": total_global_role / max(count, 1),
        "validity_precision_at_050": vp,
        "validity_recall_at_050": vr,
        "validity_f1_at_050": vf1,
        "role_acc_on_all_matched_boundaries": role_correct / max(int(matched), 1),
        "all_retained": all_retained,
        "best_f1": best_f1,
        "best_precision_under_reference_recall": best_constrained,
    }


def build_model(args, num_roles, num_frames):
    config = BertConfig.from_pretrained(args.pretrained_model_dir)
    model = FrameConditionedSpanRoleVerifier(
        config,
        num_roles=num_roles,
        num_frames=num_frames,
        variant=args.model_variant,
        projection_dim=args.projection_dim,
        frame_dim=args.frame_dim,
        frame_condition_dropout=args.frame_condition_dropout,
    )
    pretrained = BertModel.from_pretrained(args.pretrained_model_dir)
    incompatible = model.bert.load_state_dict(pretrained.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Generic BERT load mismatch: {incompatible}")
    del pretrained
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires CUDA.")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    frame_info, role_names, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [row["frame_name"] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    frame_to_role_ids_by_id = {
        frame2idx[name]: role_ids for name, role_ids in frame_to_role_ids.items()
    }
    train_rows, dev_rows = read_json(args.train_file), read_json(args.dev_file)
    task1_probabilities = load_task1_hybrid_probabilities(
        args.task1_dev_logits,
        dev_rows,
        train_rows,
        frame_names,
        args.task1_prototype_gamma,
        args.task1_baseline_weight,
    )
    train_dataset = SpanRoleVerifierDataset(
        args.train_file, tokenizer, role2idx, frame2idx,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
        boundary_negative_fraction=args.boundary_negative_fraction,
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    dev_dataset = SpanRoleVerifierDataset(
        args.dev_file, tokenizer, role2idx, frame2idx,
        max_length=args.max_length,
        candidate_file=args.dev_candidate_file,
        task1_probabilities=task1_probabilities,
        max_examples=args.max_dev_examples,
    )
    collate = make_collate(
        frame_to_role_ids_by_id,
        len(role_names),
        frame_topk=args.frame_topk,
        pad_token_id=tokenizer.pad_token_id,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = build_model(args, len(role_names), len(frame_names)).to(device)
    if args.warm_start_checkpoint:
        warm_path = Path(args.warm_start_checkpoint)
        if not warm_path.is_absolute():
            warm_path = REPO_ROOT / warm_path
        warm_checkpoint = torch.load(str(warm_path), map_location="cpu")
        warm_state = warm_checkpoint.get("model_state", warm_checkpoint)
        incompatible = model.load_state_dict(warm_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"I1 warm-start mismatch: {incompatible}")
        print(json.dumps({
            "warm_start": str(warm_path),
            "source_epoch": warm_checkpoint.get("epoch"),
            "source_variant": warm_checkpoint.get("model_variant"),
            "missing": 0,
            "unexpected": 0,
        }, ensure_ascii=False))
    encoder_parameters = list(model.bert.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulate_gradients)
    total_updates = updates_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_proportion),
        num_training_steps=total_updates,
    )
    metric_fields = [
        "epoch", "train_loss", "train_validity_loss", "train_frame_role_loss",
        "train_global_role_loss", "val_loss", "val_validity_loss",
        "val_frame_role_loss", "val_global_role_loss",
        "validity_precision_at_050", "validity_recall_at_050", "validity_f1_at_050",
        "role_acc_on_all_matched_boundaries", "best_threshold", "pipeline_precision",
        "pipeline_recall", "pipeline_f1", "pipeline_tp", "pipeline_predicted",
        "constrained_threshold", "constrained_precision", "constrained_recall",
        "constrained_f1", "epoch_seconds", "peak_gpu_memory_mb", "encoder_lr", "head_lr",
    ]
    logger = ExperimentLogger(args, "task3_verifier", metric_fields)
    (logger.result_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "train": train_dataset.stats,
                "dev_pipeline": dev_dataset.stats,
                "augmented_frame_role_licenses": augmented,
                "role_count": len(role_names),
                "frame_count": len(frame_names),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    print(json.dumps({
        "I1_config": {
            "variant": args.model_variant,
            "device": str(device),
            "train_records": len(train_dataset),
            "dev_pipeline_candidates": len(dev_dataset),
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "initialization": "generic_chinese_bert_wwm_ext_only",
        }
    }, ensure_ascii=False))

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    best_f1 = -1.0
    history = []
    optimizer.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        totals = {"loss": 0.0, "validity_loss": 0.0, "frame_role_loss": 0.0, "global_role_loss": 0.0}
        seen = 0
        progress = tqdm(train_loader, desc=f"I1-{args.model_variant}-epoch_{epoch}")
        for step, cpu_batch in enumerate(progress, start=1):
            batch = to_device(cpu_batch, device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = forward_model(
                    model, batch, training=True,
                    train_gold_frame_mass=args.train_gold_frame_mass,
                )
                losses = objective_loss(
                    outputs, batch, args.valid_pos_weight, args.focal_gamma,
                    args.global_role_loss_weight,
                )
                scaled_loss = losses["loss"] / args.accumulate_gradients
            scaler.scale(scaled_loss).backward()
            if step % args.accumulate_gradients == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler deliberately skips optimizer.step() when the
                # current FP16 gradients contain Inf/NaN (commonly on the
                # very first update while its dynamic scale is still high).
                # Advancing the LR scheduler on a skipped optimizer update
                # triggers PyTorch's "scheduler.step before optimizer.step"
                # warning and shifts the whole schedule by one.  A decreased
                # scale is the public signal that this update was skipped.
                optimizer_was_run = scaler.get_scale() >= scale_before_step
                if optimizer_was_run:
                    scheduler.step()
                optimizer.zero_grad()
            batch_size = len(batch["rows"])
            seen += batch_size
            for key in totals:
                totals[key] += float(losses[key].item()) * batch_size
            if step % args.log_interval == 0:
                progress.set_postfix(loss=f"{totals['loss'] / seen:.4f}")

        evaluation = evaluate(
            model, dev_loader, device, args, epoch, logger.result_dir
        )
        selected = evaluation["best_f1"]
        constrained = evaluation["best_precision_under_reference_recall"]
        if selected["tuple_f1"] > best_f1:
            best_f1 = selected["tuple_f1"]
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "model_variant": args.model_variant,
                    "num_roles": len(role_names),
                    "num_frames": len(frame_names),
                    "role_names": role_names,
                    "frame_names": frame_names,
                    "best_threshold": selected["threshold"],
                    "epoch": epoch,
                    "run_config": vars(args),
                },
                logger.checkpoint_dir / "task3_verifier_best.bin",
            )
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(seen, 1),
            "train_validity_loss": totals["validity_loss"] / max(seen, 1),
            "train_frame_role_loss": totals["frame_role_loss"] / max(seen, 1),
            "train_global_role_loss": totals["global_role_loss"] / max(seen, 1),
            **{key: evaluation[key] for key in (
                "val_loss", "val_validity_loss", "val_frame_role_loss",
                "val_global_role_loss", "validity_precision_at_050",
                "validity_recall_at_050", "validity_f1_at_050",
                "role_acc_on_all_matched_boundaries",
            )},
            "best_threshold": selected["threshold"],
            "pipeline_precision": selected["tuple_precision"],
            "pipeline_recall": selected["tuple_recall"],
            "pipeline_f1": selected["tuple_f1"],
            "pipeline_tp": selected["tuple_tp"],
            "pipeline_predicted": selected["predicted_tuples"],
            "constrained_threshold": constrained["threshold"] if constrained else "",
            "constrained_precision": constrained["tuple_precision"] if constrained else "",
            "constrained_recall": constrained["tuple_recall"] if constrained else "",
            "constrained_f1": constrained["tuple_f1"] if constrained else "",
            "epoch_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda" else 0.0
            ),
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        logger.log(row)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    best_epoch_row = max(
        history, key=lambda item: (item["pipeline_f1"], item["pipeline_precision"])
    )
    final_summary = {
        "model_variant": args.model_variant,
        "initialization": "generic_chinese_bert_wwm_ext_only",
        "best_epoch": best_epoch_row,
        "reference_pipeline": {
            "precision": 0.5822242618938351,
            "recall": args.reference_recall,
            "f1": 0.6097404677987824,
        },
        "checkpoint": str(logger.checkpoint_dir / "task3_verifier_best.bin"),
        "dataset_statistics_file": str(logger.result_dir / "dataset_statistics.json"),
        "run_config": vars(args),
    }
    (logger.result_dir / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
