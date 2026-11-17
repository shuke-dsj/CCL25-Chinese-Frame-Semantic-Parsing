#!/usr/bin/env python3
"""Instance-safe Task1+Task2+Task3 Dev pipeline evaluation.

Task2's official tuple does not contain a predicate-instance identifier.  Dev
contains duplicate sentence IDs with different targets, so this evaluator
consumes the internal ``task2_instance_predictions.json`` sidecar produced by
``tasks.task2.predict_centered_ensemble_h``.  Metrics are exact Task3 tuples
``(instance_index, start, end, FE)`` and therefore cannot silently mix two
predicate instances from the same sentence.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from .dataset_experiments import build_augmented_frame_roles, shifted_character_position
from .model import Model as LegacyRoleModel
from .model_c3 import UncertaintyAwareRoleClassifier
from .model_definition import DefinitionPrototypeRoleClassifier
from .model_efficient import EfficientSpanRoleClassifier


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset' / 'cfn-dev.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset' / 'cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset' / 'frame_info.json'))
    parser.add_argument(
        '--task1_dev_logits',
        default=str(REPO_ROOT / 'tasks' / 'task1' / 'analysis' /
                    'baseline-prototype-10ep' / 'dev_logits.npz'),
    )
    parser.add_argument('--task1_baseline_weight', default=0.5, type=float)
    parser.add_argument('--task1_prototype_gamma', default=2.7, type=float)
    parser.add_argument('--task2_instance_file', required=True)
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
    parser.add_argument(
        '--c_model_variant', choices=[
            'efficient_span', 'uncertainty_aware_c3',
            'definition_prototype', 'definition_pair_marginal'
        ],
        default='efficient_span',
        help='The second pipeline model; definition_prototype is reported as D.',
    )
    parser.add_argument(
        '--definition_prototype_file',
        default=str(TASK_DIR / 'definition_prototypes' /
                    'frame_fe_definition_last4_mean.npz'),
    )
    parser.add_argument(
        '--definition_frame_condition', choices=['top1', 'soft_topk'],
        default='top1',
    )
    parser.add_argument('--definition_frame_topk', default=5, type=int)
    parser.add_argument('--b_prior_weight', default=1.0, type=float)
    parser.add_argument('--c_prior_weight', default=3.0, type=float)
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--span_dim', default=256, type=int)
    parser.add_argument('--role_dim', default=256, type=int)
    parser.add_argument('--distance_dim', default=32, type=int)
    parser.add_argument('--width_dim', default=16, type=int)
    parser.add_argument('--config_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'config.json'))
    parser.add_argument('--vocab_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'vocab.txt'))
    parser.add_argument('--max_rows', default=0, type=int, help='Smoke-test limit; zero uses all Task2 spans.')
    parser.add_argument(
        '--output_dir',
        default=str(TASK_DIR / 'analysis' / 'task3-full-pipeline-BC'),
    )
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_task1_probabilities(args, dev_rows, frame_info):
    cache = np.load(args.task1_dev_logits)
    required = {'baseline_logits', 'prototype_logits', 'sentence_ids'}
    missing = required.difference(cache.files)
    if missing:
        raise ValueError(f'Task1 Dev cache missing fields: {sorted(missing)}')
    baseline = cache['baseline_logits'].astype(np.float32)
    prototype = cache['prototype_logits'].astype(np.float32).copy()
    frame_names = [item['frame_name'] for item in frame_info]
    if baseline.shape != (len(dev_rows), len(frame_names)) or prototype.shape != baseline.shape:
        raise ValueError(
            f'Task1 logits shape mismatch: baseline={baseline.shape}, '
            f'expected={(len(dev_rows), len(frame_names))}'
        )
    train_rows = read_json(args.train_file)
    seen_frames = {row['frame'] for row in train_rows}
    seen_mask = np.asarray([name in seen_frames for name in frame_names], dtype=bool)
    prototype[:, seen_mask] -= args.task1_prototype_gamma
    weight = args.task1_baseline_weight
    probabilities = weight * softmax(baseline) + (1.0 - weight) * softmax(prototype)
    expected_sids = np.asarray([int(row['sentence_id']) for row in dev_rows], dtype=np.int64)
    if not np.array_equal(cache['sentence_ids'].astype(np.int64), expected_sids):
        raise ValueError('Task1 Dev cache does not follow cfn-dev instance order.')
    return probabilities.astype(np.float32)


class PredictedSpanDataset(Dataset):
    def __init__(self, dev_rows, task2_rows, frame_probabilities, tokenizer):
        self.dev_rows = dev_rows
        self.frame_probabilities = frame_probabilities
        self.tokenizer = tokenizer
        self.target_start_id = tokenizer.convert_tokens_to_ids('[unused1]')
        self.target_end_id = tokenizer.convert_tokens_to_ids('[unused2]')
        self.items = []
        seen = set()
        for row in task2_rows:
            instance_index = int(row['instance_index'])
            if not 0 <= instance_index < len(dev_rows):
                raise IndexError(f'Invalid Task2 instance_index={instance_index}')
            source = dev_rows[instance_index]
            sentence_id = int(row['sentence_id'])
            if sentence_id != int(source['sentence_id']):
                raise ValueError(
                    f'Task2 sidecar SID mismatch at instance {instance_index}: '
                    f'{sentence_id} != {source["sentence_id"]}'
                )
            start, end = int(row['start']), int(row['end'])
            if not (0 <= start <= end < len(source['text'])):
                raise ValueError(
                    f'Invalid predicted span instance={instance_index}: [{start},{end}]'
                )
            key = (instance_index, start, end)
            if key in seen:
                raise ValueError(f'Duplicate Task2 predicted span: {key}')
            seen.add(key)
            target = source['target'][-1]
            self.items.append({
                'instance_index': instance_index,
                'sentence_id': sentence_id,
                'text': source['text'],
                'target_start': int(target['start']),
                'target_end': int(target['end']),
                'argument_start': start,
                'argument_end': end,
                'frame_prob': frame_probabilities[instance_index],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']
        raw_ts, raw_te = row['target_start'], row['target_end']
        ts, te = raw_ts + 1, raw_te + 1
        input_ids = (
            input_ids[:ts] + [self.target_start_id]
            + input_ids[ts:te + 1] + [self.target_end_id]
            + input_ids[te + 1:]
        )
        attention_mask = (
            attention_mask[:ts] + [1]
            + attention_mask[ts:te + 1] + [1]
            + attention_mask[te + 1:]
        )
        target = [ts + 1, te + 1]
        argument = [
            shifted_character_position(row['argument_start'], raw_ts, raw_te),
            shifted_character_position(row['argument_end'], raw_ts, raw_te),
        ]
        return input_ids, attention_mask, target, argument, row


def collate(batch, device, pad_token_id):
    max_length = max(len(item[0]) for item in batch)
    pad_ids = lambda values: values + [pad_token_id] * (max_length - len(values))
    pad_mask = lambda values: values + [0] * (max_length - len(values))
    return {
        'input_ids': torch.tensor([pad_ids(item[0]) for item in batch], dtype=torch.long, device=device),
        'attention_mask': torch.tensor([pad_mask(item[1]) for item in batch], dtype=torch.long, device=device),
        'target': [item[2] for item in batch],
        'argument': [item[3] for item in batch],
        'rows': [item[4] for item in batch],
    }


def build_model(args, variant, num_roles, checkpoint, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_roles
    if variant == 'B':
        model = LegacyRoleModel(config)
    elif variant == 'C':
        model = EfficientSpanRoleClassifier(
            config, span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
        )
    elif variant == 'C3':
        frame_info, _, _, frame_to_role_ids, _ = build_augmented_frame_roles(
            args.frame_info_file, args.train_file
        )
        frame_names = [row['frame_name'] for row in frame_info]
        frame2idx = {name: index for index, name in enumerate(frame_names)}
        legal = np.zeros((len(frame_names), num_roles), dtype=np.float32)
        for frame, role_ids in frame_to_role_ids.items():
            legal[frame2idx[frame], role_ids] = 1.0
        train_rows = read_json(args.train_file)
        frame_counts = {name: 0 for name in frame_names}
        for row in train_rows:
            frame_counts[row['frame']] = frame_counts.get(row['frame'], 0) + 1
        log_frequency = np.log1p(np.asarray([
            frame_counts[name] for name in frame_names
        ], dtype=np.float32))
        if log_frequency.max() > 0:
            log_frequency /= log_frequency.max()
        model = UncertaintyAwareRoleClassifier(
            config,
            num_frames=len(frame_names),
            frame_role_legal=legal,
            frame_log_frequency=log_frequency,
            span_dim=args.span_dim,
            role_dim=args.role_dim,
            distance_dim=args.distance_dim,
            width_dim=args.width_dim,
        )
    elif variant in ('D', 'D2'):
        model = DefinitionPrototypeRoleClassifier(
            config, prototype_file=args.definition_prototype_file,
            span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
            definition_aggregation=(
                'pair_logsumexp' if variant == 'D2' else 'residual'
            ),
        )
    else:
        raise ValueError(variant)
    state = torch.load(checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'Task3 {variant} checkpoint mismatch: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    return model.to(device).eval()


def collect_logits(args, variant, checkpoint, loader, num_roles, device):
    model = build_model(args, variant, num_roles, checkpoint, device)
    output = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'full-pipeline-{variant}'):
            if variant == 'B':
                logits = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    target=batch['argument'], labels=None, device=device, for_test=True,
                )['logits']
            elif variant == 'C':
                logits = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    target=batch['target'], argument=batch['argument'],
                )['logits']
            elif variant == 'C3':
                frame_probabilities = torch.tensor(
                    np.stack([row['frame_prob'] for row in batch['rows']]),
                    dtype=torch.float32, device=device,
                )
                logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target=batch['target'],
                    argument=batch['argument'],
                    frame_probabilities=frame_probabilities,
                )['logits']
            else:
                common = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'target': batch['target'],
                    'argument': batch['argument'],
                }
                if variant == 'D':
                    # D was trained with the gold frame selecting the local
                    # Frame--FE prototype.  At pipeline inference that frame
                    # is unavailable, so condition only on Task1 predictions.
                    if args.definition_frame_condition == 'soft_topk':
                        frame_probabilities = torch.tensor(
                            np.stack([row['frame_prob'] for row in batch['rows']]),
                            dtype=torch.float32, device=device,
                        )
                        logits = model(
                            **common,
                            frame_probabilities=frame_probabilities,
                            frame_topk=args.definition_frame_topk,
                        )['logits']
                    else:
                        frame_ids = torch.tensor(
                            [int(np.argmax(row['frame_prob'])) for row in batch['rows']],
                            dtype=torch.long, device=device,
                        )
                        logits = model(**common, frame_ids=frame_ids)['logits']
                else:
                    # D2 is trained to marginalise Frame--FE pair definitions
                    # and therefore always consumes Task1 probabilities.
                    frame_probabilities = torch.tensor(
                        np.stack([row['frame_prob'] for row in batch['rows']]),
                        dtype=torch.float32, device=device,
                    )
                    logits = model(
                        **common,
                        frame_probabilities=frame_probabilities,
                        frame_topk=args.definition_frame_topk,
                    )['logits']
            output.append(logits.detach().cpu().to(torch.float32).numpy())
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(output, axis=0) if output else np.zeros((0, num_roles), np.float32)


def prf(true_positive, predicted, gold):
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(true_positive), 'predicted': int(predicted), 'gold': int(gold),
    }


def gold_sets(dev_rows, role2idx):
    tuples = set()
    boundaries = set()
    for instance_index, row in enumerate(dev_rows):
        for span in row.get('cfn_spans', []):
            boundary = (instance_index, int(span['start']), int(span['end']))
            boundaries.add(boundary)
            tuples.add(boundary + (int(role2idx[span['fe_name']]),))
    return tuples, boundaries


def evaluate_predictions(dataset, predicted_roles, gold_tuples, gold_boundaries):
    predicted_boundaries = {
        (row['instance_index'], row['argument_start'], row['argument_end'])
        for row in dataset.items
    }
    predicted_tuples = {
        (row['instance_index'], row['argument_start'], row['argument_end'], int(role))
        for row, role in zip(dataset.items, predicted_roles)
    }
    tuple_tp = len(predicted_tuples & gold_tuples)
    boundary_tp = len(predicted_boundaries & gold_boundaries)
    result = {
        'task3_exact_tuple': prf(tuple_tp, len(predicted_tuples), len(gold_tuples)),
        'task2_exact_boundary': prf(
            boundary_tp, len(predicted_boundaries), len(gold_boundaries)
        ),
        'role_accuracy_on_matched_predicted_boundaries': (
            tuple_tp / boundary_tp if boundary_tp else 0.0
        ),
    }
    return result


def main():
    args = parse_args()
    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_rows = read_json(args.dev_file)
    task2_rows = read_json(args.task2_instance_file)
    if args.max_rows > 0:
        task2_rows = task2_rows[:args.max_rows]

    frame_info, role_names, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0

    frame_probabilities = load_task1_probabilities(args, dev_rows, frame_info)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = PredictedSpanDataset(
        dev_rows, task2_rows, frame_probabilities, tokenizer
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(
            collate, device=device, pad_token_id=tokenizer.pad_token_id
        ),
        drop_last=False,
    )
    b_logits = collect_logits(
        args, 'B', args.b_checkpoint, loader, len(role_names), device
    )
    second_name = {
        'efficient_span': 'C',
        'uncertainty_aware_c3': 'C3',
        'definition_prototype': 'D',
        'definition_pair_marginal': 'D2',
    }[args.c_model_variant]
    c_logits = collect_logits(
        args, second_name, args.c_checkpoint, loader, len(role_names), device
    )
    sample_frame_probs = np.stack([row['frame_prob'] for row in dataset.items])
    legal_prior = np.clip(sample_frame_probs @ membership, args.prior_epsilon, 1.0)
    log_prior = np.log(legal_prior)
    gold_tuples, gold_boundaries = gold_sets(dev_rows, role2idx)

    results = {}
    for name, logits, prior_weight in (
        ('B', b_logits, args.b_prior_weight),
        (second_name, c_logits, args.c_prior_weight),
    ):
        global_prediction = logits.argmax(axis=1)
        soft_prediction = (logits + prior_weight * log_prior).argmax(axis=1)
        results[name] = {
            'global': evaluate_predictions(
                dataset, global_prediction, gold_tuples, gold_boundaries
            ),
            'soft_frame_prior': {
                'prior_weight': prior_weight,
                **evaluate_predictions(
                    dataset, soft_prediction, gold_tuples, gold_boundaries
                ),
            },
        }
        prediction_rows = []
        for row, role_id in zip(dataset.items, soft_prediction):
            prediction_rows.append({
                'instance_index': row['instance_index'],
                'sentence_id': row['sentence_id'],
                'start': row['argument_start'], 'end': row['argument_end'],
                'role_id': int(role_id), 'role': role_names[int(role_id)],
            })
        with (output_dir / f'{name}_soft_predictions.json').open(
                'w', encoding='utf8') as handle:
            json.dump(prediction_rows, handle, ensure_ascii=False, indent=2)

    np.savez_compressed(
        output_dir / 'pipeline_logits.npz',
        b_logits=b_logits.astype(np.float32),
        c_logits=c_logits.astype(np.float32),
        frame_probs=sample_frame_probs.astype(np.float32),
        instance_indices=np.asarray(
            [row['instance_index'] for row in dataset.items], dtype=np.int32
        ),
        sentence_ids=np.asarray(
            [row['sentence_id'] for row in dataset.items], dtype=np.int64
        ),
        starts=np.asarray(
            [row['argument_start'] for row in dataset.items], dtype=np.int32
        ),
        ends=np.asarray(
            [row['argument_end'] for row in dataset.items], dtype=np.int32
        ),
        role_names=np.asarray(role_names),
    )
    summary = {
        'device': str(device),
        'num_dev_instances': len(dev_rows),
        'num_task2_predicted_spans': len(dataset),
        'num_gold_task3_tuples': len(gold_tuples),
        'num_gold_unique_boundaries': len(gold_boundaries),
        'task1_sentence_frame_acc': float(np.mean(
            frame_probabilities.argmax(axis=1)
            == np.asarray([frame2idx[row['frame']] for row in dev_rows])
        )),
        'results': results,
        'augmented_frame_role_licenses': augmented,
        'elapsed_seconds': time.perf_counter() - start_time,
        'run_config': vars(args),
        'note': (
            'Task3 exact tuples use instance_index rather than sentence_id so Dev '
            'predicate instances with duplicate sentence IDs cannot collide.'
        ),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
