#!/usr/bin/env python3
"""A2: Task3 prediction with a soft legal-role prior from Task1 probabilities."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from .dataset_experiments import build_augmented_frame_roles
from .model import Model
from .model_efficient import EfficientSpanRoleClassifier
from tasks.common.experiment_utils import REPO_ROOT


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_file', default=str(REPO_ROOT / 'dataset' / 'cfn-test-B.json'))
    p.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset' / 'frame_info.json'))
    p.add_argument('--train_file', default=str(REPO_ROOT / 'dataset' / 'cfn-train.json'))
    p.add_argument('--task1_probability_file', required=True)
    p.add_argument('--task2_prediction_file', required=True)
    p.add_argument('--checkpoint_path', default=str(TASK_DIR / 'experiment_record' / 'legacy-saves' / 'checkpoints' / 'model_task3_best.bin'))
    p.add_argument(
        '--model_variant', default='legacy',
        choices=['legacy', 'frame_masked_baseline', 'efficient_span'],
    )
    p.add_argument('--config_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'config.json'))
    p.add_argument('--vocab_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext' / 'vocab.txt'))
    p.add_argument('--prior_weight', default=1.5, type=float)
    p.add_argument('--prior_epsilon', default=1e-8, type=float)
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--span_dim', default=256, type=int)
    p.add_argument('--role_dim', default=256, type=int)
    p.add_argument('--distance_dim', default=32, type=int)
    p.add_argument('--width_dim', default=16, type=int)
    p.add_argument('--output_dir', default=str(TASK_DIR / 'predictions' / 'task3-A2-soft-frame-prior-w150'))
    p.add_argument('--max_rows', default=None, type=int, help='smoke-test limit for Task2 rows')
    return p.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def shifted_character_position(pos, target_start, target_end):
    """Map a raw character position after [CLS] and two target markers."""
    mapped = int(pos) + 1
    if pos >= target_start:
        mapped += 1
    if pos > target_end:
        mapped += 1
    return mapped


class SoftPriorDataset(torch.utils.data.Dataset):
    def __init__(self, test_file, task1_probability_file, task2_prediction_file, tokenizer,
                 max_rows=None):
        self.raw_data = read_json(test_file)
        self.task2_rows = read_json(task2_prediction_file)
        if max_rows is not None:
            self.task2_rows = self.task2_rows[:max_rows]
        self.tokenizer = tokenizer

        sidecar = np.load(task1_probability_file, allow_pickle=False)
        required = {'frame_probs','frame_names','sentence_ids','target_starts','target_ends','instance_indices'}
        missing = required.difference(sidecar.files)
        if missing:
            raise ValueError(f'Task1 probability sidecar missing fields: {sorted(missing)}')
        self.frame_probs = sidecar['frame_probs'].astype(np.float32)
        self.frame_names = sidecar['frame_names'].tolist()
        self.sentence_ids = sidecar['sentence_ids'].astype(np.int64)
        self.target_starts = sidecar['target_starts'].astype(np.int32)
        self.target_ends = sidecar['target_ends'].astype(np.int32)
        self.instance_indices = sidecar['instance_indices'].astype(np.int32)

        n = len(self.raw_data)
        if not all(len(x) == n for x in (
            self.frame_probs, self.sentence_ids, self.target_starts,
            self.target_ends, self.instance_indices,
        )):
            raise ValueError('Task1 probability sidecar length differs from test instances.')
        if self.frame_probs.ndim != 2:
            raise ValueError(f'frame_probs must be [instances,frames], got {self.frame_probs.shape}')
        raw_ids = np.asarray([int(x['sentence_id']) for x in self.raw_data], dtype=np.int64)
        raw_starts = np.asarray([int(x['target'][-1]['start']) for x in self.raw_data], dtype=np.int32)
        raw_ends = np.asarray([int(x['target'][-1]['end']) for x in self.raw_data], dtype=np.int32)
        if not np.array_equal(self.instance_indices, np.arange(n, dtype=np.int32)):
            raise ValueError('Task1 instance_indices must follow the original test row order.')
        if not (np.array_equal(raw_ids, self.sentence_ids)
                and np.array_equal(raw_starts, self.target_starts)
                and np.array_equal(raw_ends, self.target_ends)):
            raise ValueError('Task1 sidecar does not align with test sentence/target rows.')
        prob_sums = self.frame_probs.sum(axis=1)
        if not np.all(np.isfinite(self.frame_probs)) or not np.allclose(prob_sums, 1.0, atol=2e-4):
            raise ValueError('Task1 frame probabilities are non-finite or do not sum to one.')

        by_sid = defaultdict(list)
        for i, row in enumerate(self.raw_data):
            by_sid[int(row['sentence_id'])].append(i)
        self.duplicate_sentence_ids = sum(len(v) > 1 for v in by_sid.values())
        self.collapsed_identical_duplicate_rows = 0
        self.target_overlapping_task2_spans = 0
        self.items = []
        for task2_row in self.task2_rows:
            sid, start, end = int(task2_row[0]), int(task2_row[1]), int(task2_row[2])
            candidates = by_sid.get(sid, [])
            if not candidates:
                raise KeyError(f'Task2 sentence_id {sid} does not exist in test data.')

            signatures = {
                (self.raw_data[i]['text'], int(self.target_starts[i]), int(self.target_ends[i]))
                for i in candidates
            }
            if len(signatures) != 1:
                raise ValueError(
                    f'Ambiguous Task2 row for sentence_id={sid}: the official tuple lacks target '
                    f'identity but {len(signatures)} distinct target instances exist. '
                    'Generate a Task2 instance sidecar before continuing.'
                )
            if len(candidates) > 1:
                self.collapsed_identical_duplicate_rows += 1
            source = candidates[0]
            raw = self.raw_data[source]
            target_start, target_end = int(self.target_starts[source]), int(self.target_ends[source])
            if not (end < target_start or start > target_end):
                self.target_overlapping_task2_spans += 1
            # Identical duplicate test instances have the same text/target. Average
            # their probabilities rather than selecting one by sentence_id.
            frame_prob = self.frame_probs[candidates].mean(axis=0)
            self.items.append({
                'text': raw['text'], 'sentence_id': sid, 'start': start, 'end': end,
                'target_start': target_start, 'target_end': target_end,
                'frame_prob': frame_prob,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']
        ts, te = row['target_start'], row['target_end']
        target_token_start, target_token_end = ts + 1, te + 1
        input_ids = (
            input_ids[:target_token_start] + [1]
            + input_ids[target_token_start:target_token_end + 1] + [2]
            + input_ids[target_token_end + 1:]
        )
        attention_mask = attention_mask + [1, 1]
        target = [target_token_start + 1, target_token_end + 1]
        argument = [
            shifted_character_position(row['start'], ts, te),
            shifted_character_position(row['end'], ts, te),
        ]
        return input_ids, attention_mask, target, argument, row


def collate(batch, device):
    max_len = max(len(x[0]) for x in batch)
    pad = lambda x: x + [0] * (max_len - len(x))
    input_ids = torch.tensor([pad(x[0]) for x in batch], dtype=torch.long, device=device)
    attention_mask = torch.tensor([pad(x[1]) for x in batch], dtype=torch.long, device=device)
    targets = [x[2] for x in batch]
    arguments = [x[3] for x in batch]
    rows = [x[4] for x in batch]
    frame_probs = torch.tensor(
        np.stack([x['frame_prob'] for x in rows]), dtype=torch.float32, device=device
    )
    return input_ids, attention_mask, targets, arguments, rows, frame_probs


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_info, role_names, _, frame_to_role_ids, augmented_licenses = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [x['frame_name'] for x in frame_info]
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = SoftPriorDataset(
        args.test_file, args.task1_probability_file, args.task2_prediction_file,
        tokenizer, args.max_rows,
    )
    if dataset.frame_names != frame_names:
        raise ValueError('Task1 sidecar frame label order differs from frame_info.json.')

    membership = torch.zeros((len(frame_names), len(role_names)), dtype=torch.float32, device=device)
    frame_to_index = {name: i for i, name in enumerate(frame_names)}
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame_to_index[frame], torch.tensor(role_ids, device=device)] = 1.0

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = len(role_names)
    if args.model_variant == 'efficient_span':
        model = EfficientSpanRoleClassifier(
            config, span_dim=args.span_dim, role_dim=args.role_dim,
            distance_dim=args.distance_dim, width_dim=args.width_dim,
        )
    else:
        model = Model(config)
    state = torch.load(args.checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f'Task3 checkpoint mismatch: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}'
        )
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(collate, device=device), drop_last=False,
    )

    predictions = []
    global_changed = 0
    start_time = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for input_ids, attention_mask, targets, arguments, rows, frame_probs in tqdm(loader, desc='task3-A2'):
            if args.model_variant == 'efficient_span':
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    target=targets, argument=arguments,
                )
            else:
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, target=arguments,
                    labels=None, device=device, for_test=True,
                )
            logits = output['logits']
            legal_prior = torch.matmul(frame_probs, membership).clamp_min(args.prior_epsilon)
            scores = logits + args.prior_weight * torch.log(legal_prior)
            global_pred = logits.argmax(dim=-1)
            pred = scores.argmax(dim=-1)
            global_changed += (pred != global_pred).sum().item()
            for i, row in enumerate(rows):
                predictions.append([
                    row['sentence_id'], row['start'], row['end'], role_names[int(pred[i])]
                ])

    data_json = '[\n' + ',\n'.join(
        '   ' + json.dumps(x, ensure_ascii=False) for x in predictions
    ) + '\n]'
    for name in ('A_task3_test.json', 'B_task3_test.json'):
        with (output_dir / name).open('w', encoding='utf8', newline='\n') as f:
            f.write(data_json)

    elapsed = time.perf_counter() - start_time
    summary = {
        'strategy': 'task1_marginal_legal_role_prior',
        'model_variant': args.model_variant,
        'prior_weight': args.prior_weight,
        'prior_epsilon': args.prior_epsilon,
        'task1_probability_file': str(Path(args.task1_probability_file).resolve()),
        'task2_prediction_file': str(Path(args.task2_prediction_file).resolve()),
        'task3_checkpoint': str(Path(args.checkpoint_path).resolve()),
        'test_file': str(Path(args.test_file).resolve()),
        'prediction_rows': len(predictions),
        'predictions_changed_vs_global_task3': global_changed,
        'changed_ratio': global_changed / max(len(predictions), 1),
        'duplicate_sentence_id_groups': dataset.duplicate_sentence_ids,
        'task2_rows_using_identical_duplicate_instances': dataset.collapsed_identical_duplicate_rows,
        'target_overlapping_task2_spans': dataset.target_overlapping_task2_spans,
        'checkpoint_missing': len(msg.missing_keys),
        'checkpoint_unexpected': len(msg.unexpected_keys),
        'augmented_frame_role_licenses': augmented_licenses,
        'device': str(device),
        'elapsed_seconds': elapsed,
        'examples_per_second': len(predictions) / max(elapsed, 1e-9),
        'peak_gpu_memory_mb': (
            torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else 0.0
        ),
        'output_dir': str(output_dir.resolve()),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
