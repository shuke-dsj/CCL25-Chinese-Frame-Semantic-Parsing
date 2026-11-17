#!/usr/bin/env python3
"""Test-B prediction for the final B + C2 + D2(epoch 1) Task3 ensemble.

The script consumes only cfn-test-B and creates only a newly inferred
``B_task3_test.json``.  For Tianchi's six-file zip, historical A files are
copied unchanged from an existing submission directory; Test-A is never run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
import zipfile
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
from tasks.task3.model_definition import DefinitionPrototypeRoleClassifier
from tasks.task3.model_efficient import EfficientSpanRoleClassifier
from tasks.task3.predict_soft_frame_prior import SoftPriorDataset, collate


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_file', default=str(REPO_ROOT / 'dataset/cfn-test-B.json'))
    p.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    p.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    p.add_argument('--task1_probability_file', default=str(
        REPO_ROOT / 'tasks/task1/predictions/prototype-hybrid-soft-prior/B_task1_frame_probs.npz'))
    p.add_argument('--task1_submission_file', default=str(
        REPO_ROOT / 'tasks/task1/predictions/prototype-hybrid-soft-prior/B_task1_test.json'))
    p.add_argument('--task2_prediction_file', default=str(
        REPO_ROOT / 'tasks/task2/predictions/task2-BDEH-centered-040-020-020-020-qm034/B_task2_test.json'))
    p.add_argument('--b_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-B-frame-masked-warmstart-2ep-result/checkpoints/task3_best.bin'))
    p.add_argument('--c2_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-C2-mixed-objective-a020-3ep-result/checkpoints/task3_best.bin'))
    p.add_argument('--d2_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-D2-pair-marginal-longtail-2ep-result/checkpoints/task3_epoch_1.bin'))
    p.add_argument('--definition_prototype_file', default=str(
        TASK_DIR / 'definition_prototypes/frame_fe_definition_last4_mean.npz'))
    p.add_argument('--definition_frame_topk', default=5, type=int)
    p.add_argument('--b_weight', default=0.585, type=float)
    p.add_argument('--c2_weight', default=0.100, type=float)
    p.add_argument('--d2_weight', default=0.315, type=float)
    p.add_argument('--prior_low', default=0.6, type=float)
    p.add_argument('--prior_high', default=1.8, type=float)
    p.add_argument('--prior_power', default=2.0, type=float)
    p.add_argument('--prior_epsilon', default=1e-8, type=float)
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--span_dim', default=256, type=int)
    p.add_argument('--role_dim', default=256, type=int)
    p.add_argument('--distance_dim', default=32, type=int)
    p.add_argument('--width_dim', default=16, type=int)
    p.add_argument('--config_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext/config.json'))
    p.add_argument('--vocab_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext/vocab.txt'))
    p.add_argument('--max_rows', default=None, type=int)
    p.add_argument('--reuse_cache', action='store_true')
    p.add_argument('--reuse_b_cache', default=str(
        TASK_DIR / 'predictions/task3-BC-zscore-adaptive-prior/test_logits.npz'))
    p.add_argument('--reuse_b_summary', default=str(
        TASK_DIR / 'predictions/task3-BC-zscore-adaptive-prior/summary.json'))
    p.add_argument('--historical_a_dir', default=str(
        TASK_DIR / 'predictions/submission-task3-C-w300'))
    p.add_argument('--output_dir', default=str(
        TASK_DIR / 'predictions/task3-B-C2-D2-e1-zscore-adaptive'))
    return p.parse_args()


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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args, variant, num_roles, checkpoint, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_roles
    if variant == 'B':
        model = LegacyRoleModel(config)
    elif variant == 'C2':
        model = EfficientSpanRoleClassifier(
            config, span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
        )
    elif variant == 'D2':
        model = DefinitionPrototypeRoleClassifier(
            config, prototype_file=args.definition_prototype_file,
            span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
            definition_aggregation='pair_logsumexp',
        )
    else:
        raise ValueError(variant)
    state = torch.load(checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'{variant} checkpoint mismatch: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    return model.to(device).eval()


def collect_logits(args, variant, checkpoint, loader, num_roles, device):
    model = load_model(args, variant, num_roles, checkpoint, device)
    chunks = []
    with torch.no_grad():
        for input_ids, attention_mask, targets, arguments, rows, frame_probs in tqdm(
                loader, desc=f'task3-final-{variant}'):
            if variant == 'B':
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=arguments, labels=None, device=device, for_test=True,
                )['logits']
            elif variant == 'C2':
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=targets, argument=arguments,
                )['logits']
            else:
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=targets, argument=arguments,
                    frame_probabilities=frame_probs,
                    frame_topk=args.definition_frame_topk,
                )['logits']
            chunks.append(logits.detach().cpu().to(torch.float32).numpy())
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def validate_reused_b(args, expected_shape):
    cache_path, summary_path = Path(args.reuse_b_cache), Path(args.reuse_b_summary)
    if args.max_rows is not None:
        raise ValueError('B-logit reuse is disabled when --max_rows is set.')
    if not cache_path.exists() or not summary_path.exists():
        raise FileNotFoundError('Reusable B cache or its provenance summary is missing.')
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    checks = {
        'checkpoint_B': Path(args.b_checkpoint).resolve(),
        'task1_probability_file': Path(args.task1_probability_file).resolve(),
        'task2_prediction_file': Path(args.task2_prediction_file).resolve(),
    }
    for key, expected in checks.items():
        if Path(summary[key]).resolve() != expected:
            raise ValueError(f'Reused B cache provenance mismatch in {key}.')
    cache = np.load(cache_path, allow_pickle=False)
    logits = cache['b_logits'].astype(np.float32)
    if logits.shape != expected_shape:
        raise ValueError(f'Reused B logits have shape {logits.shape}, expected {expected_shape}.')
    return logits


def write_submission_zip(args, output_dir, task3_file):
    submission_dir = output_dir / 'submission_files'
    submission_dir.mkdir(parents=True, exist_ok=True)
    split = 'A' if Path(args.test_file).name == 'cfn-test-A.json' else 'B'
    if split == 'A':
        sources = {
            'A_task1_test.json': Path(args.task1_submission_file),
            'A_task2_test.json': Path(args.task2_prediction_file),
            'A_task3_test.json': task3_file,
        }
        zip_path = output_dir / 'submission-task3-testA-role-ensemble.zip'
    else:
        # The evaluator historically expects all six names for a B submission.
        # Reuse historical A files byte-for-byte; no A model/data is loaded.
        historical_a = Path(args.historical_a_dir)
        sources = {
            'A_task1_test.json': historical_a / 'A_task1_test.json',
            'A_task2_test.json': historical_a / 'A_task2_test.json',
            'A_task3_test.json': historical_a / 'A_task3_test.json',
            'B_task1_test.json': Path(args.task1_submission_file),
            'B_task2_test.json': Path(args.task2_prediction_file),
            'B_task3_test.json': task3_file,
        }
        zip_path = output_dir / 'submission-task3-B-C2-D2-e1.zip'
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f'Submission component missing: {source}')
        shutil.copyfile(source, submission_dir / name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(submission_dir / name, arcname=name)
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(archive.namelist())
        expected = sorted(sources)
        if names != expected or archive.testzip() is not None:
            raise RuntimeError(f'Invalid submission zip members: {names}')
    return submission_dir, zip_path, sorted(sources)


def main():
    args = parse_args()
    split = 'A' if Path(args.test_file).name == 'cfn-test-A.json' else 'B'
    if Path(args.test_file).name not in {'cfn-test-A.json', 'cfn-test-B.json'}:
        raise ValueError('test_file must be cfn-test-A.json or cfn-test-B.json')
    weights = np.asarray([args.b_weight, args.c2_weight, args.d2_weight], np.float32)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f'B/C2/D2 weights must be non-negative and sum to one: {weights}')
    if args.prior_high < args.prior_low:
        raise ValueError('prior_high must be >= prior_low.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f'test_logits_{split.lower()}.npz'
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_info, role_names, _, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = SoftPriorDataset(
        args.test_file, args.task1_probability_file, args.task2_prediction_file,
        tokenizer, args.max_rows,
    )
    if dataset.frame_names != frame_names:
        raise ValueError('Task1 frame order differs from frame_info.json.')
    expected_shape = (len(dataset), len(role_names))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(collate, device=device), drop_last=False,
    )

    if args.reuse_cache:
        cache = np.load(cache_path, allow_pickle=False)
        b_logits = cache['b_logits'].astype(np.float32)
        c2_logits = cache['c2_logits'].astype(np.float32)
        d2_logits = cache['d2_logits'].astype(np.float32)
    else:
        # Test-B can reuse the previously validated B cache. Test-A must
        # compute the legacy B logits on the A rows; reusing a B cache would
        # silently misalign candidates.
        if split == 'B':
            b_logits = validate_reused_b(args, expected_shape)
        else:
            b_logits = collect_logits(
                args, 'B', args.b_checkpoint, loader, len(role_names), device
            )
        c2_logits = collect_logits(
            args, 'C2', args.c2_checkpoint, loader, len(role_names), device
        )
        d2_logits = collect_logits(
            args, 'D2', args.d2_checkpoint, loader, len(role_names), device
        )
        np.savez_compressed(
            cache_path, b_logits=b_logits, c2_logits=c2_logits, d2_logits=d2_logits,
        )
    for name, logits in (('B', b_logits), ('C2', c2_logits), ('D2', d2_logits)):
        if logits.shape != expected_shape or not np.isfinite(logits).all():
            raise ValueError(f'{name} logits invalid: shape={logits.shape}')

    frame_probs = np.stack([row['frame_prob'] for row in dataset.items]).astype(np.float32)
    legal_prior = np.clip(frame_probs @ membership, args.prior_epsilon, 1.0)
    top2 = np.sort(np.partition(frame_probs, -2, axis=1)[:, -2:], axis=1)
    confidence = percentile_confidence(top2[:, 1] - top2[:, 0])
    prior_weight = args.prior_low + (
        args.prior_high - args.prior_low
    ) * np.power(confidence, args.prior_power)
    combined = (
        args.b_weight * zscore_rows(b_logits)
        + args.c2_weight * zscore_rows(c2_logits)
        + args.d2_weight * zscore_rows(d2_logits)
    )
    prediction = (combined + prior_weight[:, None] * np.log(legal_prior)).argmax(1)
    rows = [
        [item['sentence_id'], item['start'], item['end'], role_names[int(role_id)]]
        for item, role_id in zip(dataset.items, prediction)
    ]
    task3_file = output_dir / f'{split}_task3_test.json'
    with task3_file.open('w', encoding='utf8', newline='\n') as handle:
        handle.write(
            '[\n' + ',\n'.join(
                '   ' + json.dumps(row, ensure_ascii=False) for row in rows
            ) + '\n]'
        )
    # Read it back before packaging, catching truncation/encoding mistakes.
    decoded = json.loads(task3_file.read_text(encoding='utf8'))
    if decoded != rows or len(decoded) != len(dataset):
        raise RuntimeError(f'{split}_task3_test.json round-trip validation failed.')
    task2_rows = json.loads(Path(args.task2_prediction_file).read_text(encoding='utf8'))
    if len(task2_rows) != len(decoded) or any(
        list(task2[:3]) != list(task3[:3])
        for task2, task3 in zip(task2_rows, decoded)
    ):
        raise RuntimeError('Task3 rows do not align one-to-one with Task2 boundaries.')
    submission_dir, zip_path, zip_members = write_submission_zip(
        args, output_dir, task3_file
    )

    previous_file = TASK_DIR / 'predictions/task3-BC-zscore-adaptive-prior/B_task3_test.json'
    changed = None
    if previous_file.exists():
        previous = json.loads(previous_file.read_text(encoding='utf8'))
        if len(previous) == len(rows):
            changed = sum(a[3] != b[3] for a, b in zip(previous, rows))
    summary = {
        'strategy': 'B_C2_D2_epoch1_zscore_margin_adaptive_prior',
        'test_split': split,
        'weights': {'B': args.b_weight, 'C2': args.c2_weight, 'D2': args.d2_weight},
        'prior': {'confidence': 'margin_percentile', 'low': args.prior_low,
                  'high': args.prior_high, 'power': args.prior_power},
        'prediction_rows': len(rows),
        'changed_roles_vs_previous_BC_submission': changed,
        'prior_weight_min': float(prior_weight.min()),
        'prior_weight_mean': float(prior_weight.mean()),
        'prior_weight_max': float(prior_weight.max()),
        'checkpoints': {'B': str(Path(args.b_checkpoint).resolve()),
                        'C2': str(Path(args.c2_checkpoint).resolve()),
                        'D2_epoch1': str(Path(args.d2_checkpoint).resolve())},
        'task1_probability_file': str(Path(args.task1_probability_file).resolve()),
        'task2_prediction_file': str(Path(args.task2_prediction_file).resolve()),
        'new_task3_file': str(task3_file.resolve()),
        'submission_dir': str(submission_dir.resolve()),
        'submission_zip': str(zip_path.resolve()),
        'submission_zip_members': zip_members,
        'integrity': {
            'task2_task3_row_alignment': True,
            'task3_json_round_trip': True,
            'task3_sha256': sha256(task3_file),
            'submission_zip_sha256': sha256(zip_path),
            'submission_component_sha256': {
                name: sha256(submission_dir / name) for name in zip_members
            },
        },
        'historical_a_files_reused_without_inference': True,
        'logits_cache': str(cache_path.resolve()),
        'device': str(device),
        'elapsed_seconds': time.perf_counter() - start,
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
