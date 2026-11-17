#!/usr/bin/env python3
"""Predict Task3 with normalized B/C logits and adaptive Task1 frame prior."""

from __future__ import annotations

import argparse
import gc
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.model import Model as LegacyRoleModel
from tasks.task3.model_efficient import EfficientSpanRoleClassifier
from tasks.task3.predict_soft_frame_prior import SoftPriorDataset, collate


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default=str(REPO_ROOT / 'dataset' / 'cfn-test-B.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset' / 'cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset' / 'frame_info.json'))
    parser.add_argument('--task1_probability_file', required=True)
    parser.add_argument('--task2_prediction_file', required=True)
    parser.add_argument(
        '--b_checkpoint',
        default=str(TASK_DIR / 'experiment_record' /
                    'task3-B-frame-masked-warmstart-2ep-result' /
                    'checkpoints' / 'task3_best.bin'),
    )
    parser.add_argument(
        '--c_checkpoint',
        default=str(TASK_DIR / 'experiment_record' /
                    'task3-C-efficient-span-5ep-result' /
                    'checkpoints' / 'task3_best.bin'),
    )
    parser.add_argument('--b_weight', default=0.45, type=float)
    parser.add_argument('--prior_low', default=0.4, type=float)
    parser.add_argument('--prior_high', default=1.8, type=float)
    parser.add_argument('--prior_power', default=0.5, type=float)
    parser.add_argument(
        '--confidence', default='margin_percentile',
        choices=['margin_percentile', 'negative_entropy_percentile'],
    )
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--span_dim', default=256, type=int)
    parser.add_argument('--role_dim', default=256, type=int)
    parser.add_argument('--distance_dim', default=32, type=int)
    parser.add_argument('--width_dim', default=16, type=int)
    parser.add_argument('--config_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'config.json'))
    parser.add_argument('--vocab_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'vocab.txt'))
    parser.add_argument('--max_rows', default=None, type=int)
    parser.add_argument('--reuse_cache', action='store_true')
    parser.add_argument(
        '--output_dir',
        default=str(TASK_DIR / 'predictions' /
                    'task3-BC-zscore-adaptive-prior'),
    )
    return parser.parse_args()


def zscore_rows(logits):
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / centered.std(axis=1, keepdims=True).clip(min=1e-6)


def percentile_confidence(values):
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.ones_like(values, dtype=np.float32)
    return (
        np.searchsorted(unique, values, side='right').astype(np.float32) - 1.0
    ) / float(len(unique) - 1)


def build_model(args, variant, num_roles, checkpoint, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_roles
    if variant == 'B':
        model = LegacyRoleModel(config)
    else:
        model = EfficientSpanRoleClassifier(
            config, span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
        )
    state = torch.load(checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'{variant} checkpoint mismatch: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    return model.to(device).eval()


def collect_logits(args, variant, checkpoint, loader, num_roles, device):
    model = build_model(args, variant, num_roles, checkpoint, device)
    chunks = []
    with torch.no_grad():
        for input_ids, attention_mask, targets, arguments, rows, frame_probs in tqdm(
                loader, desc=f'task3-ensemble-{variant}'):
            if variant == 'B':
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=arguments, labels=None, device=device, for_test=True,
                )['logits']
            else:
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=targets, argument=arguments,
                )['logits']
            chunks.append(logits.detach().cpu().to(torch.float32).numpy())
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def main():
    args = parse_args()
    if not 0.0 <= args.b_weight <= 1.0:
        raise ValueError(f'b_weight must be in [0,1], got {args.b_weight}')
    if args.prior_high < args.prior_low:
        raise ValueError('prior_high must be >= prior_low')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / 'test_logits.npz'
    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_info, role_names, _, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = SoftPriorDataset(
        args.test_file, args.task1_probability_file, args.task2_prediction_file,
        tokenizer, args.max_rows,
    )
    if dataset.frame_names != frame_names:
        raise ValueError('Task1 sidecar frame label order differs from frame_info.')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(collate, device=device), drop_last=False,
    )
    if args.reuse_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f'--reuse_cache requested but cache is missing: {cache_path}')
        cache = np.load(cache_path)
        b_logits = cache['b_logits'].astype(np.float32)
        c_logits = cache['c_logits'].astype(np.float32)
    else:
        b_logits = collect_logits(
            args, 'B', args.b_checkpoint, loader, len(role_names), device
        )
        c_logits = collect_logits(
            args, 'C', args.c_checkpoint, loader, len(role_names), device
        )
        np.savez_compressed(
            cache_path,
            b_logits=b_logits.astype(np.float32),
            c_logits=c_logits.astype(np.float32),
        )
    if b_logits.shape != (len(dataset), len(role_names)) or c_logits.shape != b_logits.shape:
        raise ValueError(
            f'Cached logits shape mismatch: B={b_logits.shape}, C={c_logits.shape}, '
            f'expected={(len(dataset), len(role_names))}'
        )

    frame_probs = np.stack([row['frame_prob'] for row in dataset.items]).astype(np.float32)
    legal_prior = np.clip(frame_probs @ membership, args.prior_epsilon, 1.0)
    top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    entropy = -(frame_probs * np.log(frame_probs.clip(min=1e-12))).sum(axis=1)
    if args.confidence == 'margin_percentile':
        confidence = percentile_confidence(margin)
    else:
        confidence = percentile_confidence(-entropy)
    prior_weight = args.prior_low + (
        args.prior_high - args.prior_low
    ) * np.power(confidence, args.prior_power)
    combined = (
        args.b_weight * zscore_rows(b_logits)
        + (1.0 - args.b_weight) * zscore_rows(c_logits)
    )
    scores = combined + prior_weight[:, None] * np.log(legal_prior)
    prediction = scores.argmax(axis=1)
    predictions = [
        [row['sentence_id'], row['start'], row['end'], role_names[int(role_id)]]
        for row, role_id in zip(dataset.items, prediction)
    ]
    data_json = '[\n' + ',\n'.join(
        '   ' + json.dumps(row, ensure_ascii=False) for row in predictions
    ) + '\n]'
    for name in ('A_task3_test.json', 'B_task3_test.json'):
        with (output_dir / name).open('w', encoding='utf8', newline='\n') as handle:
            handle.write(data_json)

    summary = {
        'strategy': 'normalized_BC_with_confidence_adaptive_frame_prior',
        'normalization': 'per_sample_zscore_over_1009_roles',
        'b_weight': args.b_weight, 'c_weight': 1.0 - args.b_weight,
        'confidence': args.confidence,
        'prior_low': args.prior_low, 'prior_high': args.prior_high,
        'prior_power': args.prior_power,
        'prior_weight_min': float(prior_weight.min()),
        'prior_weight_mean': float(prior_weight.mean()),
        'prior_weight_max': float(prior_weight.max()),
        'prediction_rows': len(predictions),
        'checkpoint_B': str(Path(args.b_checkpoint).resolve()),
        'checkpoint_C': str(Path(args.c_checkpoint).resolve()),
        'task1_probability_file': str(Path(args.task1_probability_file).resolve()),
        'task2_prediction_file': str(Path(args.task2_prediction_file).resolve()),
        'logits_cache': str(cache_path.resolve()),
        'augmented_frame_role_licenses': augmented,
        'device': str(device),
        'elapsed_seconds': time.perf_counter() - start_time,
        'run_config': vars(args),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
